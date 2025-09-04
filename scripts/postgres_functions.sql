CREATE OR REPLACE FUNCTION edit_distance(str1 text, str2 text)
RETURNS float AS $$
DECLARE
    m integer;
    n integer;
    dp float[];
    i integer;
    j integer;
    width integer;
BEGIN
    IF str1 IS NULL AND str2 IS NULL THEN
        RETURN -1;
    ELSIF str1 IS NULL OR str2 IS NULL THEN
        RETURN 100;
    END IF;
    
    m := LENGTH(str1);
    n := LENGTH(str2);
    
    IF m = 0 AND n = 0 THEN
        RETURN 1.0;
    END IF;
    
    width := n + 1;
    dp := array_fill(0::float, ARRAY[(m + 1) * (n + 1)]);
    
    FOR i IN 0..m LOOP
        dp[i * width] := i::float;
    END LOOP;
    
    FOR j IN 0..n LOOP
        dp[j] := j::float;
    END LOOP;
    
    FOR i IN 1..m LOOP
        FOR j IN 1..n LOOP
            IF SUBSTRING(str1, i, 1) = SUBSTRING(str2, j, 1) THEN
                dp[i * width + j] := dp[(i-1) * width + (j-1)];
            ELSE
                dp[i * width + j] := 1 + LEAST(
                    dp[(i-1) * width + j],     
                    dp[i * width + (j-1)],      
                    dp[(i-1) * width + (j-1)]   
                );
            END IF;
        END LOOP;
    END LOOP;
    
    RETURN dp[m * width + n] / GREATEST(m, n)::float;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION age_difference(dob1 date, dob2 date)
RETURNS double precision AS $$
DECLARE
    days_diff float;
    age_diff_years float;
BEGIN
    IF dob1 IS NULL AND dob2 IS NULL THEN
        RETURN 10;
    ELSIF dob1 IS NULL OR dob2 IS NULL THEN
        RETURN 100;
    END IF;
  
    days_diff := (dob1 - dob2);
    age_diff_years := days_diff / 365.25;
    
    RETURN GREATEST(LEAST(age_diff_years / 100.0, 1.0), -1.0);
EXCEPTION
    WHEN OTHERS THEN
        RETURN 50;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION soundex(input_string text)
RETURNS text AS $$
DECLARE
    soundex_code TEXT;
    current_char CHAR;
    previous_digit CHAR;
    current_digit CHAR;
    i INT;
BEGIN
    IF input_string IS NULL OR input_string = '' THEN
        RETURN NULL;
    END IF;

    soundex_code := LEFT(input_string, 1);
    previous_digit := '0';

    FOR i IN 2..COALESCE(LENGTH(input_string), 1) LOOP
        current_char := SUBSTRING(input_string FROM i FOR 1);
        CASE current_char
            WHEN 'B', 'F', 'P', 'V' THEN current_digit := '1';
            WHEN 'C', 'G', 'J', 'K', 'Q', 'S', 'X', 'Z' THEN current_digit := '2';
            WHEN 'D', 'T' THEN current_digit := '3';
            WHEN 'L' THEN current_digit := '4';
            WHEN 'M', 'N' THEN current_digit := '5';
            WHEN 'R' THEN current_digit := '6';
            ELSE current_digit := '0';
        END CASE;

        IF current_digit <> previous_digit AND current_digit <> '0' THEN
            soundex_code := soundex_code || current_digit;
        END IF;

        previous_digit := current_digit;
    END LOOP;

    soundex_code := soundex_code || '0000';
    RETURN LEFT(soundex_code, 4);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create blocking keys 
CREATE OR REPLACE FUNCTION create_blocking_keys(
    batch_size integer DEFAULT 100000,
    job_schema text DEFAULT 'public',
    records_table text DEFAULT 'records'
) RETURNS void AS $$
DECLARE
    total_records integer;
    processed_records integer := 0;
    start_time timestamp;
    full_table_name text;
BEGIN
    start_time := clock_timestamp();
    
    -- Build full table name
    full_table_name := quote_ident(job_schema) || '.' || quote_ident(records_table);
    
    -- Drop and recreate record_blocks table
    EXECUTE format('DROP TABLE IF EXISTS %I.record_blocks CASCADE', job_schema);
    EXECUTE format('CREATE UNLOGGED TABLE %I.record_blocks (
        id TEXT,
        block_key TEXT,
        sort_key TEXT
    )', job_schema);
    
    -- Get total record count
    EXECUTE format('SELECT COUNT(*) FROM %s', full_table_name) INTO total_records;
    
    RAISE NOTICE 'Processing % records from % for blocking keys', total_records, full_table_name;
    
    -- Create blocking keys
    EXECUTE format('
        INSERT INTO %I.record_blocks (id, block_key, sort_key)
        WITH record_data AS (
            SELECT 
                r.id, 
                r.last_name, 
                r.zip, 
                r.address,
                r.dob,
                r.sex,
                EXTRACT(YEAR FROM r.dob) as birth_year,
                soundex(r.last_name) as ln_soundex,
                LEFT(r.last_name, 6) as ln_prefix
            FROM %s r
        )
        SELECT id, block_key, sort_key FROM (
            SELECT 
                id,
                ''N_'' || COALESCE(ln_soundex, ''NULL'') || ''_'' || COALESCE(ln_prefix, ''NULL'') as block_key,
                COALESCE(last_name, '''') || ''|'' || COALESCE(address, '''') as sort_key
            FROM record_data
            
            UNION ALL
            
            SELECT 
                id,
                ''A_'' || COALESCE(LEFT(zip, 3), ''NA'') || ''_'' || COALESCE(LEFT(address, 6), ''NA'') as block_key,
                COALESCE(address, '''') || ''|'' || COALESCE(last_name, '''') as sort_key
            FROM record_data
            
            UNION ALL
            
            SELECT 
                id,
                ''D_'' || COALESCE(birth_year::TEXT, ''NA'') || ''_'' || COALESCE(sex, ''NA'') || ''_'' || COALESCE(ln_soundex, ''NULL'') as block_key,
                COALESCE(birth_year::TEXT, ''NA'') || ''|'' || COALESCE(sex, '''') || ''|'' || COALESCE(last_name, '''') as sort_key
            FROM record_data
            WHERE birth_year IS NOT NULL
        ) blocks
    ', job_schema, full_table_name);
    
    GET DIAGNOSTICS processed_records = ROW_COUNT;
    
    -- Create indexes
    EXECUTE format('CREATE INDEX idx_%I_record_blocks_block_key ON %I.record_blocks(block_key)', 
                   job_schema, job_schema);
    EXECUTE format('CREATE INDEX idx_%I_record_blocks_gid ON %I.record_blocks(id)', 
                   job_schema, job_schema);
    
    -- Create block_sizes table
    EXECUTE format('DROP TABLE IF EXISTS %I.block_sizes CASCADE', job_schema);
    EXECUTE format('CREATE TABLE %I.block_sizes AS
        SELECT block_key, COUNT(*) AS block_size
        FROM %I.record_blocks
        GROUP BY block_key', job_schema, job_schema, job_schema);
    
    EXECUTE format('CREATE INDEX idx_%I_block_sizes ON %I.block_sizes(block_key)', 
                   job_schema, job_schema);
    
    RAISE NOTICE 'Blocking keys created: % entries in % for schema %', 
        processed_records, clock_timestamp() - start_time, job_schema;
END;
$$ LANGUAGE plpgsql;

-- Create blocking keys for parallel workers
CREATE OR REPLACE FUNCTION create_blocking_keys_test(
    batch_size integer DEFAULT 10000,
    gid_start text DEFAULT NULL,
    gid_end text DEFAULT NULL,
    table_suffix text DEFAULT '',
    job_schema text DEFAULT 'public',
    records_table text DEFAULT 'records'
) RETURNS void AS $$
DECLARE
    total_records integer;
    processed_records integer := 0;
    start_time timestamp;
    table_name text;
    full_table_name text;
BEGIN
    start_time := clock_timestamp();
    
    -- Build full table names
    full_table_name := quote_ident(job_schema) || '.' || quote_ident(records_table);
    
    IF table_suffix IS NULL OR table_suffix = '' THEN
        table_name := quote_ident(job_schema) || '.record_blocks';
    ELSE
        table_suffix := regexp_replace(table_suffix, '[^a-zA-Z0-9_]', '', 'g');
        table_name := quote_ident(job_schema) || '.record_blocks_' || table_suffix;
    END IF;
    
    EXECUTE format('DROP TABLE IF EXISTS %s CASCADE', table_name);
    EXECUTE format('CREATE UNLOGGED TABLE %s (
        id TEXT,
        block_key TEXT,
        sort_key TEXT
    )', table_name);
    
    EXECUTE format('SELECT COUNT(*) FROM %s WHERE ($1 IS NULL OR id >= $1) AND ($2 IS NULL OR id <= $2)', 
                   full_table_name) 
    INTO total_records USING gid_start, gid_end;
    
    RAISE NOTICE 'Processing % records for table % (range: % to %)', 
        total_records, table_name, COALESCE(gid_start, 'MIN'), COALESCE(gid_end, 'MAX');
    
    -- Create blocking keys
    EXECUTE format('
        INSERT INTO %s (id, block_key, sort_key)
        WITH record_data AS (
            SELECT 
                r.id, 
                r.last_name, 
                r.zip, 
                r.address,
                r.dob,
                r.sex,
                EXTRACT(YEAR FROM r.dob) as birth_year,
                soundex(r.last_name) as ln_soundex,
                LEFT(r.last_name, 6) as ln_prefix
            FROM %s r
            WHERE ($1 IS NULL OR r.id >= $1)
              AND ($2 IS NULL OR r.id <= $2)
        )
        SELECT id, block_key, sort_key FROM (
            SELECT 
                id,
                ''N_'' || COALESCE(ln_soundex, ''NULL'') || ''_'' || COALESCE(ln_prefix, ''NULL'') as block_key,
                COALESCE(last_name, '''') || ''|'' || COALESCE(address, '''') as sort_key
            FROM record_data
            
            UNION ALL
            
            SELECT 
                id,
                ''A_'' || COALESCE(LEFT(zip, 3), ''NA'') || ''_'' || COALESCE(LEFT(address, 6), ''NA'') as block_key,
                COALESCE(address, '''') || ''|'' || COALESCE(last_name, '''') as sort_key
            FROM record_data
            
            UNION ALL
            
            SELECT 
                id,
                ''D_'' || COALESCE(birth_year::TEXT, ''NA'') || ''_'' || COALESCE(sex, ''NA'') || ''_'' || COALESCE(ln_soundex, ''NULL'') as block_key,
                COALESCE(birth_year::TEXT, ''NA'') || ''|'' || COALESCE(sex, '''') || ''|'' || COALESCE(last_name, '''') as sort_key
            FROM record_data
            WHERE birth_year IS NOT NULL
        ) blocks
    ', table_name, full_table_name) USING gid_start, gid_end;
    
    GET DIAGNOSTICS processed_records = ROW_COUNT;
    
    EXECUTE format('CREATE INDEX %I ON %s(block_key)', 
                   'idx_' || replace(table_name, '.', '_') || '_block_key', table_name);
    EXECUTE format('CREATE INDEX %I ON %s(id)', 
                   'idx_' || replace(table_name, '.', '_') || '_gid', table_name);
    
    RAISE NOTICE 'Table % created with % entries in %', 
        table_name, processed_records, clock_timestamp() - start_time;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION compare_records_optimized(
    max_block_size integer DEFAULT 500,
    window_size integer DEFAULT 100, 
    overlap integer DEFAULT 50,
    chunk_size integer DEFAULT 1000,
    job_schema text DEFAULT 'public',
    records_table text DEFAULT 'records'
) RETURNS void AS $$
DECLARE
    total_blocks integer;
    processed_blocks integer := 0;
    processed_records integer := 0;
    block_key_val text;
    block_size_val integer;
    start_time timestamp;
    total_comparisons integer := 0;
    full_table_name text;
    processed_table_name text;
BEGIN
    start_time := clock_timestamp();
    
    -- Build table names
    full_table_name := quote_ident(job_schema) || '.' || quote_ident(records_table);
    processed_table_name := quote_ident(job_schema) || '.processed_records';
    
    -- Create processed_records table
    EXECUTE format('DROP TABLE IF EXISTS %s CASCADE', processed_table_name);
    EXECUTE format('CREATE UNLOGGED TABLE %s (
        from_id TEXT,
        to_id TEXT,
        edit_dist_ln FLOAT,
        edit_dist_phone_num FLOAT,
        edit_dist_mn FLOAT,
        edit_dist_zip FLOAT,
        edit_dist_city FLOAT,
        age_diff FLOAT,
        sex_diff INTEGER,
        ssn_match INTEGER,
        state_match INTEGER,
        edit_dist_mail_address FLOAT,
        record1_sex INTEGER,
        record2_sex INTEGER,
        record1_agecategory INTEGER,
        record2_agecategory INTEGER,
        PRIMARY KEY (from_id, to_id)
    )', processed_table_name);
    
    -- Get total blocks to process
    EXECUTE format('SELECT COUNT(*) FROM %I.block_sizes 
                    WHERE block_size <= %s AND block_size > 3', 
                   job_schema, max_block_size) INTO total_blocks;
    
    RAISE NOTICE 'Processing % blocks in schema % with max size %', 
        total_blocks, job_schema, max_block_size;
    
    -- Process each block
    FOR block_key_val, block_size_val IN 
        EXECUTE format('SELECT bs.block_key, bs.block_size
                       FROM %I.block_sizes bs
                       WHERE bs.block_size <= %s AND bs.block_size > 3
                       ORDER BY bs.block_size', job_schema, max_block_size)
    LOOP
        processed_blocks := processed_blocks + 1;
        
        -- Insert comparisons
        EXECUTE format('
            INSERT INTO %s
            WITH block_records AS (
                SELECT 
                    r.*,
                    ROW_NUMBER() OVER (ORDER BY rb.sort_key, r.id) AS row_num,
                    EXTRACT(YEAR FROM AGE(CURRENT_DATE, r.dob)) as age_years
                FROM %I.record_blocks rb
                JOIN %s r ON rb.id = r.id
                WHERE rb.block_key = $1
            ),
            record_pairs AS (
                SELECT 
                    r1.id AS from_id,
                    r2.id AS to_id,
                    r1.last_name AS r1_ln, r2.last_name AS r2_ln,
                    r1.middle_name AS r1_mi, r2.middle_name AS r2_mi,
                    r1.dob AS r1_dob, r2.dob AS r2_dob,
                    r1.city AS r1_city, r2.city AS r2_city,
                    r1.zip AS r1_zip, r2.zip AS r2_zip,
                    r1.phone AS r1_tel, r2.phone AS r2_tel,
                    r1.state AS r1_st, r2.state AS r2_st,
                    r1.address AS r1_adr, r2.address AS r2_adr,
                    r1.sex AS r1_sex, r2.sex AS r2_sex,
                    r1.ssn AS r1_ssn, r2.ssn AS r2_ssn,
                    r1.age_years AS r1_age_years, r2.age_years AS r2_age_years
                FROM block_records r1
                JOIN block_records r2 ON r1.row_num < r2.row_num
            )
            SELECT 
                from_id, to_id,
                edit_distance(r1_ln, r2_ln),
                edit_distance(r1_tel, r2_tel),
                edit_distance(r1_mi, r2_mi),
                edit_distance(r1_zip, r2_zip),
                edit_distance(r1_city, r2_city),
                age_difference(r1_dob, r2_dob),
                CASE 
                    WHEN r1_sex IS NULL OR r2_sex IS NULL THEN -1
                    WHEN r1_sex != r2_sex THEN 1
                    ELSE 0 
                END,
                CASE 
                    WHEN r1_ssn IS NULL OR r2_ssn IS NULL THEN -1
                    WHEN r1_ssn = r2_ssn THEN 1
                    ELSE 0 
                END,
                CASE 
                    WHEN r1_st IS NULL OR r2_st IS NULL THEN -1
                    WHEN r1_st = r2_st THEN 1
                    ELSE 0 
                END,
                edit_distance(r1_adr, r2_adr),
                CASE 
                    WHEN r1_sex IS NULL THEN -1
                    WHEN r1_sex = ''M'' THEN 0
                    WHEN r1_sex = ''F'' THEN 1
                    ELSE -1
                END,
                CASE 
                    WHEN r2_sex IS NULL THEN -1
                    WHEN r2_sex = ''M'' THEN 0
                    WHEN r2_sex = ''F'' THEN 1
                    ELSE -1
                END,
                CASE 
                    WHEN r1_age_years IS NULL THEN -1
                    WHEN r1_age_years < 18 THEN 0
                    WHEN r1_age_years BETWEEN 18 AND 50 THEN 1
                    ELSE 2
                END,
                CASE 
                    WHEN r2_age_years IS NULL THEN -1
                    WHEN r2_age_years < 18 THEN 0
                    WHEN r2_age_years BETWEEN 18 AND 50 THEN 1
                    ELSE 2
                END
            FROM record_pairs
            ON CONFLICT (from_id, to_id) DO NOTHING
        ', processed_table_name, job_schema, full_table_name) USING block_key_val;
        
        processed_records := processed_records + block_size_val;
        
        IF processed_blocks % 100 = 0 THEN
            RAISE NOTICE 'Progress: % of % blocks processed', processed_blocks, total_blocks;
        END IF;
    END LOOP;
    
    EXECUTE format('SELECT COUNT(*) FROM %s', processed_table_name) INTO total_comparisons;
    
    RAISE NOTICE 'Completed: % blocks, % comparisons in %',
        processed_blocks, total_comparisons, clock_timestamp() - start_time;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION compare_records_exhaustive(
    job_schema text DEFAULT 'public',
    records_table text DEFAULT 'records'
) RETURNS void AS $$
DECLARE
    total_records integer;
    start_time timestamp;
    total_comparisons integer := 0;
    full_table_name text;
    processed_table_name text;
BEGIN
    start_time := clock_timestamp();
    
    -- Build table names
    full_table_name := quote_ident(job_schema) || '.' || quote_ident(records_table);
    processed_table_name := quote_ident(job_schema) || '.processed_records';
    
    EXECUTE format('SELECT COUNT(*) FROM %s', full_table_name) INTO total_records;
    
    RAISE NOTICE 'Starting exhaustive comparison of % records from %', total_records, full_table_name;
    
    -- Create processed_records table
    EXECUTE format('DROP TABLE IF EXISTS %s CASCADE', processed_table_name);
    EXECUTE format('CREATE UNLOGGED TABLE %s (
        from_id TEXT,
        to_id TEXT,
        edit_dist_ln FLOAT,
        edit_dist_phone_num FLOAT,
        edit_dist_mn FLOAT,
        edit_dist_zip FLOAT,
        edit_dist_city FLOAT,
        age_diff FLOAT,
        sex_diff INTEGER,
        ssn_match INTEGER,
        state_match INTEGER,
        edit_dist_mail_address FLOAT,
        record1_sex INTEGER,
        record2_sex INTEGER,
        record1_agecategory INTEGER,
        record2_agecategory INTEGER,
        PRIMARY KEY (from_id, to_id)
    )', processed_table_name);
    
    -- Create exhaustive comparisons
    EXECUTE format('
        INSERT INTO %s
        WITH age_data AS (
            SELECT 
                r.*,
                EXTRACT(YEAR FROM AGE(CURRENT_DATE, r.dob)) as age_years
            FROM %s r
        )
        SELECT 
            r1.id, 
            r2.id,
            edit_distance(r1.last_name, r2.last_name),
            edit_distance(r1.phone, r2.phone),
            edit_distance(r1.middle_name, r2.middle_name),
            edit_distance(r1.zip, r2.zip),
            edit_distance(r1.city, r2.city),
            age_difference(r1.dob, r2.dob),
            CASE 
                WHEN r1.sex IS NULL OR r2.sex IS NULL THEN -1
                WHEN r1.sex != r2.sex THEN 1
                ELSE 0 
            END,
            CASE 
                WHEN r1.ssn IS NULL OR r2.ssn IS NULL THEN -1
                WHEN r1.ssn = r2.ssn THEN 1
                ELSE 0 
            END,
            CASE 
                WHEN r1.state IS NULL OR r2.state IS NULL THEN -1
                WHEN r1.state = r2.state THEN 1
                ELSE 0 
            END,
            edit_distance(r1.address, r2.address),
            CASE 
                WHEN r1.sex IS NULL THEN -1
                WHEN r1.sex = ''M'' THEN 0
                WHEN r1.sex = ''F'' THEN 1
                ELSE -1
            END,
            CASE 
                WHEN r2.sex IS NULL THEN -1
                WHEN r2.sex = ''M'' THEN 0
                WHEN r2.sex = ''F'' THEN 1
                ELSE -1
            END,
            CASE 
                WHEN r1.age_years IS NULL THEN -1
                WHEN r1.age_years < 18 THEN 0
                WHEN r1.age_years BETWEEN 18 AND 50 THEN 1
                ELSE 2
            END,
            CASE 
                WHEN r2.age_years IS NULL THEN -1
                WHEN r2.age_years < 18 THEN 0
                WHEN r2.age_years BETWEEN 18 AND 50 THEN 1
                ELSE 2
            END
        FROM age_data r1
        JOIN age_data r2 ON r1.id < r2.id
        ON CONFLICT (from_id, to_id) DO NOTHING
    ', processed_table_name, full_table_name);
    
    GET DIAGNOSTICS total_comparisons = ROW_COUNT;
    
    RAISE NOTICE 'Exhaustive processing complete: % matches found in %',
        total_comparisons, clock_timestamp() - start_time;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION compare_records_optimized_parallel(
    block_key_start text,
    block_key_end text,
    max_block_size integer DEFAULT 500,
    window_size integer DEFAULT 100,
    overlap integer DEFAULT 50,
    similarity_threshold float DEFAULT 2.0,
    table_suffix text DEFAULT '',
    job_schema text DEFAULT 'public',
    records_table text DEFAULT 'records'
) RETURNS void AS $$
DECLARE
    temp_table_name text;
    block_key_val text;
    block_size_val integer;
    window_start integer;
    window_end integer;
    total_comparisons integer := 0;
    curr_window_matches integer := 0;
    full_table_name text;
BEGIN
    -- Build table names
    full_table_name := quote_ident(job_schema) || '.' || quote_ident(records_table);
    
    IF table_suffix IS NULL OR table_suffix = '' THEN
        temp_table_name := quote_ident(job_schema) || '.processed_records_temp';
    ELSE
        table_suffix := regexp_replace(table_suffix, '[^a-zA-Z0-9_]', '', 'g');
        temp_table_name := quote_ident(job_schema) || '.processed_records_worker_' || table_suffix;
    END IF;
    
    EXECUTE format('DROP TABLE IF EXISTS %s CASCADE', temp_table_name);
    
    EXECUTE format('CREATE UNLOGGED TABLE %s (
        from_id TEXT,
        to_id TEXT,
        similarity_score FLOAT,
        edit_dist_ln FLOAT,
        edit_dist_phone_num FLOAT,
        edit_dist_mn FLOAT,
        edit_dist_zip FLOAT,
        edit_dist_city FLOAT,
        age_diff FLOAT,
        sex_diff INTEGER,
        ssn_match INTEGER,
        state_match INTEGER,
        edit_dist_mail_address FLOAT,
        record1_sex INTEGER,
        record2_sex INTEGER,
        record1_agecategory INTEGER,
        record2_agecategory INTEGER,
        PRIMARY KEY (from_id, to_id)
    )', temp_table_name);
    
    FOR block_key_val, block_size_val IN 
        EXECUTE format('SELECT bs.block_key, bs.block_size
                       FROM %I.block_sizes bs
                       WHERE bs.block_size <= %s 
                         AND bs.block_size > 3
                         AND bs.block_key BETWEEN $1 AND $2
                       ORDER BY bs.block_size', job_schema, max_block_size)
        USING block_key_start, block_key_end
    LOOP
        window_start := 1;
        
        WHILE window_start <= block_size_val LOOP
            window_end := LEAST(window_start + window_size - 1, block_size_val);
            
            EXECUTE format('
                INSERT INTO %s (
                    from_id, to_id, similarity_score, edit_dist_ln, edit_dist_phone_num,
                    edit_dist_mn, edit_dist_zip, edit_dist_city, age_diff,
                    sex_diff, ssn_match, state_match, edit_dist_mail_address,
                    record1_sex, record2_sex, record1_agecategory, record2_agecategory
                )
                WITH block_records AS (
                    SELECT 
                        r.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY rb.block_key 
                            ORDER BY rb.sort_key, r.id
                        ) AS row_num,
                        EXTRACT(YEAR FROM AGE(CURRENT_DATE, r.dob)) as age_years
                    FROM %I.record_blocks rb
                    JOIN %s r ON rb.id = r.id
                    WHERE rb.block_key = $1
                ),
                record_pairs AS (
                    SELECT 
                        r1.id AS r1_gid,
                        r1.last_name AS r1_ln, r1.middle_name AS r1_mi, r1.dob AS r1_dob,
                        r1.city AS r1_city, r1.zip AS r1_zip, r1.phone AS r1_tel,
                        r1.state AS r1_st, r1.address AS r1_adr, r1.sex AS r1_sex, r1.ssn AS r1_ssn,
                        r1.age_years AS r1_age_years,
                        r2.id AS r2_gid,
                        r2.last_name AS r2_ln, r2.middle_name AS r2_mi, r2.dob AS r2_dob,
                        r2.city AS r2_city, r2.zip AS r2_zip, r2.phone AS r2_tel,
                        r2.state AS r2_st, r2.address AS r2_adr, r2.sex AS r2_sex, r2.ssn AS r2_ssn,
                        r2.age_years AS r2_age_years
                    FROM block_records r1
                    JOIN block_records r2 ON r1.row_num < r2.row_num
                        AND r1.row_num BETWEEN $2 AND $3
                        AND r2.row_num BETWEEN $2 AND $3
                )
                SELECT
                r1_gid, r2_gid,
                   (
                       CASE WHEN edit_distance(r1_ln, r2_ln) <= 0.5 THEN 1 ELSE 0 END +
                       CASE WHEN edit_distance(r1_mi, r2_mi) <= 0.5 THEN 1 ELSE 0 END +
                       CASE WHEN age_difference(r1_dob, r2_dob) BETWEEN -0.55 AND 0.55 THEN 1 ELSE 0 END +
                       CASE WHEN r1_st = r2_st THEN 1 ELSE 0 END +
                       CASE WHEN r1_city = r2_city THEN 1 ELSE 0 END
                   )::float,
                   edit_distance(r1_ln, r2_ln),
                   edit_distance(r1_tel, r2_tel),
                   edit_distance(r1_mi, r2_mi),
                   edit_distance(r1_zip, r2_zip),
                   edit_distance(r1_city, r2_city),
                   age_difference(r1_dob, r2_dob),
                   CASE 
                       WHEN r1_sex IS NULL OR r2_sex IS NULL THEN -1
                       WHEN r1_sex != r2_sex THEN 1
                       ELSE 0 
                   END,
                   CASE 
                       WHEN r1_ssn IS NULL OR r2_ssn IS NULL THEN -1
                       WHEN r1_ssn = r2_ssn THEN 1
                       ELSE 0 
                   END,
                   CASE 
                       WHEN r1_st IS NULL OR r2_st IS NULL THEN -1
                       WHEN r1_st = r2_st THEN 1
                       ELSE 0 
                   END,
                   edit_distance(r1_adr, r2_adr),
                   CASE 
                       WHEN r1_sex IS NULL THEN -1
                       WHEN r1_sex = ''M'' THEN 0
                       WHEN r1_sex = ''F'' THEN 1
                       ELSE -1
                   END,
                   CASE 
                       WHEN r2_sex IS NULL THEN -1
                       WHEN r2_sex = ''M'' THEN 0
                       WHEN r2_sex = ''F'' THEN 1
                       ELSE -1
                   END,
                   CASE 
                       WHEN r1_age_years IS NULL THEN -1
                       WHEN r1_age_years < 18 THEN 0
                       WHEN r1_age_years BETWEEN 18 AND 50 THEN 1
                       ELSE 2
                   END,
                   CASE 
                       WHEN r2_age_years IS NULL THEN -1
                       WHEN r2_age_years < 18 THEN 0
                       WHEN r2_age_years BETWEEN 18 AND 50 THEN 1
                       ELSE 2
                   END
               FROM record_pairs
               WHERE (
                   CASE WHEN edit_distance(r1_ln, r2_ln) <= 0.5 THEN 1 ELSE 0 END +
                   CASE WHEN edit_distance(r1_mi, r2_mi) <= 0.5 THEN 1 ELSE 0 END +
                   CASE WHEN age_difference(r1_dob, r2_dob) BETWEEN -0.55 AND 0.55 THEN 1 ELSE 0 END +
                   CASE WHEN r1_st = r2_st THEN 1 ELSE 0 END +
                   CASE WHEN r1_city = r2_city THEN 1 ELSE 0 END
               )::float >= $4
               ON CONFLICT (from_id, to_id) DO NOTHING
           ', temp_table_name, job_schema, full_table_name) 
           USING block_key_val, window_start, window_end, similarity_threshold;
           
           GET DIAGNOSTICS curr_window_matches = ROW_COUNT;
           total_comparisons := total_comparisons + curr_window_matches;
           
           window_start := window_start + window_size - overlap;
       END LOOP;
   END LOOP;
   
   RAISE NOTICE 'Created worker table % with % comparisons using sliding window', temp_table_name, total_comparisons;
END;
$$ LANGUAGE plpgsql;

-- Parallel exhaustive comparison for workers
CREATE OR REPLACE FUNCTION compare_records_exhaustive_parallel(
   gid_start text,
   gid_end text,
   table_suffix text DEFAULT '',
   job_schema text DEFAULT 'public',
   records_table text DEFAULT 'records'
) RETURNS void AS $$
DECLARE
   temp_table_name text;
   full_table_name text;
BEGIN
   -- Build table names
   full_table_name := quote_ident(job_schema) || '.' || quote_ident(records_table);
   
   IF table_suffix IS NULL OR table_suffix = '' THEN
       temp_table_name := quote_ident(job_schema) || '.processed_records_temp';
   ELSE
       table_suffix := regexp_replace(table_suffix, '[^a-zA-Z0-9_]', '', 'g');
       temp_table_name := quote_ident(job_schema) || '.processed_records_worker_' || table_suffix;
   END IF;
   
   EXECUTE format('DROP TABLE IF EXISTS %s CASCADE', temp_table_name);
   
   EXECUTE format('
       CREATE UNLOGGED TABLE %s AS
       WITH age_data AS (
           SELECT 
               r.*,
               EXTRACT(YEAR FROM AGE(CURRENT_DATE, r.dob)) as age_years
           FROM %s r
       )
       SELECT 
           r1.id as from_id, 
           r2.id as to_id,
           (
               CASE WHEN edit_distance(r1.last_name, r2.last_name) <= 0.5 THEN 1 ELSE 0 END +
               CASE WHEN edit_distance(r1.middle_name, r2.middle_name) <= 0.5 THEN 1 ELSE 0 END +
               CASE WHEN age_difference(r1.dob, r2.dob) BETWEEN -0.55 AND 0.55 THEN 1 ELSE 0 END +
               CASE WHEN r1.state = r2.state THEN 1 ELSE 0 END +
               CASE WHEN r1.city = r2.city THEN 1 ELSE 0 END
           )::float AS similarity_score,
           edit_distance(r1.last_name, r2.last_name) as edit_dist_ln,
           edit_distance(r1.phone, r2.phone) as edit_dist_phone_num,
           edit_distance(r1.middle_name, r2.middle_name) as edit_dist_mn,
           edit_distance(r1.zip, r2.zip) as edit_dist_zip,
           edit_distance(r1.city, r2.city) as edit_dist_city,
           age_difference(r1.dob, r2.dob) as age_diff,
           CASE 
               WHEN r1.sex IS NULL OR r2.sex IS NULL THEN -1
               WHEN r1.sex != r2.sex THEN 1
               ELSE 0 
           END as sex_diff,
           CASE 
               WHEN r1.ssn IS NULL OR r2.ssn IS NULL THEN -1
               WHEN r1.ssn = r2.ssn THEN 1
               ELSE 0 
           END as ssn_match,
           CASE 
               WHEN r1.state IS NULL OR r2.state IS NULL THEN -1
               WHEN r1.state = r2.state THEN 1
               ELSE 0 
           END as state_match,
           edit_distance(r1.address, r2.address) as edit_dist_mail_address,
           CASE 
               WHEN r1.sex IS NULL THEN -1
               WHEN r1.sex = ''M'' THEN 0
               WHEN r1.sex = ''F'' THEN 1
               ELSE -1
           END as record1_sex,
           CASE 
               WHEN r2.sex IS NULL THEN -1
               WHEN r2.sex = ''M'' THEN 0
               WHEN r2.sex = ''F'' THEN 1
               ELSE -1
           END as record2_sex,
           CASE 
               WHEN r1.age_years IS NULL THEN -1
               WHEN r1.age_years < 18 THEN 0
               WHEN r1.age_years BETWEEN 18 AND 50 THEN 1
               ELSE 2
           END as record1_agecategory,
           CASE 
               WHEN r2.age_years IS NULL THEN -1
               WHEN r2.age_years < 18 THEN 0
               WHEN r2.age_years BETWEEN 18 AND 50 THEN 1
               ELSE 2
           END as record2_agecategory
       FROM age_data r1
       JOIN age_data r2 ON r1.id < r2.id
       WHERE r1.id BETWEEN $1 AND $2
   ', temp_table_name, full_table_name) 
   USING gid_start, gid_end;
   
   RAISE NOTICE 'Created worker table % with exhaustive comparison', temp_table_name;
END;
$$ LANGUAGE plpgsql;

-- Merge blocking tables from workers
CREATE OR REPLACE FUNCTION merge_blocking_tables(job_schema text DEFAULT 'public')
RETURNS void AS $$
DECLARE
   table_record record;
   total_records bigint := 0;
   merged_count integer := 0;
BEGIN
   EXECUTE format('DROP TABLE IF EXISTS %I.record_blocks CASCADE', job_schema);
   EXECUTE format('CREATE UNLOGGED TABLE %I.record_blocks (
       id TEXT,
       block_key TEXT,
       sort_key TEXT
   )', job_schema);
   
   FOR table_record IN 
       EXECUTE format('SELECT tablename 
                      FROM pg_tables 
                      WHERE schemaname = %L 
                      AND tablename LIKE ''record_blocks_worker_%%''',
                      job_schema)
   LOOP
       EXECUTE format('INSERT INTO %I.record_blocks SELECT * FROM %I.%I', 
                     job_schema, job_schema, table_record.tablename);
       
       EXECUTE format('SELECT COUNT(*) FROM %I.%I', job_schema, table_record.tablename) 
       INTO total_records;
       RAISE NOTICE 'Merged % records from %', total_records, table_record.tablename;
       
       EXECUTE format('DROP TABLE IF EXISTS %I.%I CASCADE', job_schema, table_record.tablename);
       
       merged_count := merged_count + 1;
   END LOOP;
   
   EXECUTE format('CREATE INDEX IF NOT EXISTS idx_block_key ON %I.record_blocks(block_key)', job_schema);
   EXECUTE format('CREATE INDEX IF NOT EXISTS idx_gid_blocks ON %I.record_blocks(id)', job_schema);
   
   EXECUTE format('DROP TABLE IF EXISTS %I.block_sizes CASCADE', job_schema);
   EXECUTE format('CREATE TABLE %I.block_sizes AS
                  SELECT block_key, COUNT(*) AS block_size
                  FROM %I.record_blocks
                  GROUP BY block_key', job_schema, job_schema, job_schema);
   
   EXECUTE format('CREATE INDEX idx_block_sizes ON %I.block_sizes(block_key)', job_schema);
   
   EXECUTE format('SELECT COUNT(*) FROM %I.record_blocks', job_schema) INTO total_records;
   RAISE NOTICE 'Merged % worker tables. Final record_blocks table has % records', merged_count, total_records;
   
   EXECUTE format('SELECT COUNT(*) FROM %I.block_sizes', job_schema) INTO total_records;
   RAISE NOTICE 'Created % unique blocks in block_sizes table', total_records;
END;
$$ LANGUAGE plpgsql;

-- Merge processed records from workers
CREATE OR REPLACE FUNCTION merge_processed_records(job_schema text DEFAULT 'public')
RETURNS void AS $$
DECLARE
   table_record record;
   total_records bigint := 0;
   merged_count integer := 0;
BEGIN
   -- Create processed_records if it doesn't exist
   EXECUTE format('CREATE TABLE IF NOT EXISTS %I.processed_records (
       from_id TEXT,
       to_id TEXT,
       similarity_score FLOAT,
       edit_dist_ln FLOAT,
       edit_dist_phone_num FLOAT,
       edit_dist_mn FLOAT,
       edit_dist_zip FLOAT,
       edit_dist_city FLOAT,
       age_diff FLOAT,
       sex_diff INTEGER,
       ssn_match INTEGER,
       state_match INTEGER,
       edit_dist_mail_address FLOAT,
       record1_sex INTEGER,
       record2_sex INTEGER,
       record1_agecategory INTEGER,
       record2_agecategory INTEGER,
       PRIMARY KEY (from_id, to_id)
   )', job_schema);
   
   EXECUTE format('TRUNCATE %I.processed_records', job_schema);
   
   FOR table_record IN 
       EXECUTE format('SELECT tablename 
                      FROM pg_tables 
                      WHERE schemaname = %L 
                      AND tablename LIKE ''processed_records_worker_%%''',
                      job_schema)
   LOOP
       EXECUTE format('INSERT INTO %I.processed_records SELECT * FROM %I.%I ON CONFLICT (from_id, to_id) DO NOTHING', 
                     job_schema, job_schema, table_record.tablename);
       
       EXECUTE format('SELECT COUNT(*) FROM %I.%I', job_schema, table_record.tablename) 
       INTO total_records;
       RAISE NOTICE 'Merged % records from %', total_records, table_record.tablename;
       
       EXECUTE format('DROP TABLE IF EXISTS %I.%I CASCADE', job_schema, table_record.tablename);
       
       merged_count := merged_count + 1;
   END LOOP;
   
   EXECUTE format('SELECT COUNT(*) FROM %I.processed_records', job_schema) INTO total_records;
   RAISE NOTICE 'Merged % worker tables. Final processed_records table has % record pairs', 
       merged_count, total_records;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION process_positive_record_pairs() 
RETURNS void AS $$
DECLARE
   total_pairs integer;
   start_time timestamp;
BEGIN
   start_time := clock_timestamp();
   
   SELECT COUNT(*) INTO total_pairs FROM labels;
   
   RAISE NOTICE 'Starting processing of % record pairs', total_pairs;
   
   DROP TABLE IF EXISTS processed_positive_records;
   CREATE UNLOGGED TABLE processed_positive_records (
       from_id TEXT,
       to_id TEXT,
       edit_dist_ln FLOAT,
       edit_dist_phone_num FLOAT,
       edit_dist_mn FLOAT,
       edit_dist_zip FLOAT,
       edit_dist_city FLOAT,
       age_diff FLOAT,
       sex_diff INTEGER,
       ssn_match INTEGER,
       state_match INTEGER,
       edit_dist_mail_address FLOAT,
       record1_sex INTEGER,
       record2_sex INTEGER,
       record1_agecategory INTEGER,
       record2_agecategory INTEGER,
       PRIMARY KEY (from_id, to_id)
   );
   
   -- Process positive pairs from labels
   INSERT INTO processed_positive_records
   WITH age_data AS (
       SELECT 
           r.*,
           EXTRACT(YEAR FROM AGE(CURRENT_DATE, r.dob)) as age_years
       FROM records r
   )
   SELECT 
       l.from_id,
       l.to_id,
       edit_distance(r1.last_name, r2.last_name),
       edit_distance(r1.phone, r2.phone),
       edit_distance(r1.middle_name, r2.middle_name),
       edit_distance(r1.zip, r2.zip),
       edit_distance(r1.city, r2.city),
       age_difference(r1.dob, r2.dob),
       CASE 
           WHEN r1.sex IS NULL OR r2.sex IS NULL THEN -1
           WHEN r1.sex != r2.sex THEN 1
           ELSE 0 
       END,
       CASE 
           WHEN r1.ssn IS NULL OR r2.ssn IS NULL THEN -1
           WHEN r1.ssn = r2.ssn THEN 1
           ELSE 0 
       END,
       CASE 
           WHEN r1.state IS NULL OR r2.state IS NULL THEN -1
           WHEN r1.state = r2.state THEN 1
           ELSE 0 
       END,
       edit_distance(r1.address, r2.address),
       CASE 
           WHEN r1.sex IS NULL THEN -1
           WHEN r1.sex = 'M' THEN 0
           WHEN r1.sex = 'F' THEN 1
           ELSE -1
       END,
       CASE 
           WHEN r2.sex IS NULL THEN -1
           WHEN r2.sex = 'M' THEN 0
           WHEN r2.sex = 'F' THEN 1
           ELSE -1
       END,
       CASE 
           WHEN r1.age_years IS NULL THEN -1
           WHEN r1.age_years < 18 THEN 0
           WHEN r1.age_years BETWEEN 18 AND 50 THEN 1
           ELSE 2
       END,
       CASE 
           WHEN r2.age_years IS NULL THEN -1
           WHEN r2.age_years < 18 THEN 0
           WHEN r2.age_years BETWEEN 18 AND 50 THEN 1
           ELSE 2
       END
   FROM labels l
   JOIN age_data r1 ON l.from_id = r1.id
   JOIN age_data r2 ON l.to_id = r2.id
   ON CONFLICT (from_id, to_id) DO NOTHING;
   
   RAISE NOTICE 'Processing complete in %', clock_timestamp() - start_time;
END;
$$ LANGUAGE plpgsql;