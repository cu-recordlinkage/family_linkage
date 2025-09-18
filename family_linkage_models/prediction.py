import pandas as pd
import joblib
import os
from sqlalchemy import create_engine, text
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

def compare(database_url, job_schema, records_table, logger, size_threshold=10000, 
           max_block_size=500, window_size=100, overlap=50, blocking_batch_size=100000, 
           num_workers=4, similarity_threshold=2.0, progress_callback=None, tqdm_flag=False):
    
    def update_progress(message, percentage):
        if progress_callback:
            progress_callback(message, percentage)
        logger.info(f"Progress {percentage}%: {message}")
    
    try:
        update_progress("Creating database connection", 10)
        engine = create_engine(database_url)

        # Get dataset size
        with engine.begin() as connection:
            result = connection.execute(
                text(f"SELECT COUNT(*) FROM {job_schema}.{records_table}")
            )
            dataset_size = result.scalar()
        
        update_progress(f"Dataset loaded: {dataset_size} records", 20)
        logger.info(f"Dataset size: {dataset_size} records")
        
        update_progress("Cleaning up existing tables", 25)
        # Clean up existing tables before starting
        with engine.begin() as connection:
            tables_to_drop = ['record_blocks', 'block_sizes', 'processed_records']
            for table in tables_to_drop:
                connection.execute(text(f"DROP TABLE IF EXISTS {job_schema}.{table} CASCADE"))
            
            # Drop worker tables
            worker_tables = connection.execute(text(f"""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = '{job_schema}' 
                AND (tablename LIKE 'record_blocks_%' 
                     OR tablename LIKE 'processed_records_%')
            """)).fetchall()
            
            for table in worker_tables:
                connection.execute(text(f"DROP TABLE IF EXISTS {job_schema}.{table[0]} CASCADE"))
            
            logger.info("Cleaned up existing tables")

        # Choose strategy based on dataset size
        if dataset_size > size_threshold:
            update_progress(f"Using optimized blocking strategy", 30)
            logger.info(f"Using optimized blocking strategy (dataset > {size_threshold})")
            _run_parallel_optimized_comparison_with_progress(
                database_url, job_schema, records_table, logger, dataset_size,
                max_block_size, window_size, overlap, blocking_batch_size, 
                num_workers, similarity_threshold, tqdm_flag
            )
        else:
            update_progress(f"Using exhaustive comparison", 30)
            logger.info(f"Using parallel exhaustive comparison (dataset <= {size_threshold})")
            _run_parallel_exhaustive_comparison_with_progress(
                database_url, job_schema, records_table, logger, dataset_size, num_workers, tqdm_flag
            )
        
        update_progress("Comparison completed", 90)
        
        # Get final count
        with engine.begin() as connection:
            result = connection.execute(
                text(f"SELECT COUNT(*) FROM {job_schema}.processed_records")
            )
            total_pairs = result.scalar()
        
        update_progress(f"Generated {total_pairs} record pairs", 100)
        logger.info(f"Comparison complete: {total_pairs} record pairs generated")
    except Exception as e:
        logger.error(f"Error in compare function: {e}")
        raise
    finally:
        engine.dispose()

def _run_parallel_optimized_comparison_with_progress(database_url, job_schema, records_table, logger,
                                                   dataset_size, max_block_size, window_size, 
                                                   overlap, blocking_batch_size, num_workers, 
                                                   similarity_threshold, tqdm_flag):
    try:
        # Step 1: Parallel blocking keys creation
        logger.info("Creating blocking keys in parallel...")
        engine = create_engine(database_url)
        with engine.begin() as connection:
            result = connection.execute(text(
                f"SELECT ARRAY_AGG(DISTINCT id ORDER BY id) FROM {job_schema}.{records_table}"
            ))
            gid_array = result.scalar()

            if not gid_array or len(gid_array) == 0:
                raise ValueError("No records found in database")

            total_gids = len(gid_array)
            chunk_size = total_gids // num_workers
            gid_ranges = []

            for i in range(num_workers):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size - 1 if i < num_workers - 1 else total_gids - 1
                gid_ranges.append((gid_array[start_idx], gid_array[end_idx]))

        # Execute parallel blocking
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = []
            for i, (gid_start, gid_end) in enumerate(gid_ranges):
                future = executor.submit(
                    _worker_create_blocking_keys,
                    engine.url, job_schema, records_table,
                    blocking_batch_size, gid_start, gid_end, i
                )
                futures.append(future)

            # Track progress with tqdm if enabled
            if tqdm_flag:
                with tqdm(total=len(futures), desc="Creating blocking keys", unit="worker") as pbar:
                    for future in as_completed(futures):
                        try:
                            future.result()  # This will raise an exception if the worker failed
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Worker failed: {e}")
                            raise
            else:
                for future in as_completed(futures):
                    try:
                        future.result()  # This will raise an exception if the worker failed
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")
                        raise

        # Merge blocking tables
        logger.info("Merging blocking tables...")
        with engine.begin() as connection:
            connection.execute(text(f"SELECT merge_blocking_tables('{job_schema}')"))

            block_count = connection.execute(
                text(f"SELECT COUNT(*) FROM {job_schema}.record_blocks")
            ).scalar()
            unique_blocks = connection.execute(
                text(f"SELECT COUNT(*) FROM {job_schema}.block_sizes")
            ).scalar()
            logger.info(f"Merged to {block_count} blocking entries with {unique_blocks} unique blocks")

        # Step 2: Parallel optimized comparison
        logger.info("Performing parallel optimized record comparison...")

        with engine.begin() as connection:
            result = connection.execute(text(f"""
                SELECT ARRAY_AGG(block_key ORDER BY block_key) 
                FROM {job_schema}.block_sizes 
                WHERE block_size <= :max_block_size AND block_size > 3
            """), {"max_block_size": max_block_size})
            block_array = result.scalar()

            if not block_array or len(block_array) == 0:
                logger.warning("No blocks found for comparison")
                return

        # Partition blocks for workers
        total_blocks = len(block_array)
        chunk_size = total_blocks // num_workers
        block_ranges = []

        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size - 1 if i < num_workers - 1 else total_blocks - 1
            block_ranges.append((block_array[start_idx], block_array[end_idx]))

        # Execute parallel comparison
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, (block_start, block_end) in enumerate(block_ranges):
                future = executor.submit(
                    _worker_compare_records_optimized,
                    database_url, job_schema, records_table,
                    block_start, block_end, max_block_size,
                    window_size, overlap, similarity_threshold, i
                )
                futures.append(future)

            # Track comparison progress
            if tqdm_flag:
                with tqdm(total=len(futures), desc="Comparing records", unit="worker") as pbar:
                    for future in as_completed(futures):
                        try:
                            future.result()
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Comparison worker failed: {e}")
                            raise
            else:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Comparison worker failed: {e}")
                        raise

        # Merge results
        logger.info("Merging comparison results...")
        with engine.begin() as connection:
            connection.execute(text(f"SELECT merge_processed_records('{job_schema}')"))
            final_count = connection.execute(
                text(f"SELECT COUNT(*) FROM {job_schema}.processed_records")
            ).scalar()
            logger.info(f"Merged {final_count} unique record pairs")
    finally:
        engine.dispose()

def _run_parallel_exhaustive_comparison_with_progress(database_url, job_schema, records_table, logger,
                                                    dataset_size, num_workers, tqdm_flag):
    try:
        logger.info("Running parallel exhaustive comparison...")
        engine = create_engine(database_url)
        with engine.begin() as connection:
            result = connection.execute(text(
                f"SELECT ARRAY_AGG(DISTINCT id ORDER BY id) FROM {job_schema}.{records_table}"
            ))
            gid_array = result.scalar()

            if not gid_array:
                raise ValueError("No records found")

        total_gids = len(gid_array)
        chunk_size = total_gids // num_workers
        gid_ranges = []

        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size - 1 if i < num_workers - 1 else total_gids - 1
            gid_ranges.append((gid_array[start_idx], gid_array[end_idx]))

        # Execute parallel exhaustive comparison
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, (gid_start, gid_end) in enumerate(gid_ranges):
                future = executor.submit(
                    _worker_compare_records_exhaustive,
                    database_url, job_schema, records_table,
                    gid_start, gid_end, i
                )
                futures.append(future)

            # Track progress
            if tqdm_flag:
                with tqdm(total=len(futures), desc="Exhaustive comparison", unit="worker") as pbar:
                    for future in as_completed(futures):
                        try:
                            future.result()
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Exhaustive worker failed: {e}")
                            raise
            else:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Exhaustive worker failed: {e}")
                        raise

        # Merge results
        logger.info("Merging exhaustive comparison results...")
        with engine.begin() as connection:
            connection.execute(text(f"SELECT merge_processed_records('{job_schema}')"))
            final_count = connection.execute(
                text(f"SELECT COUNT(*) FROM {job_schema}.processed_records")
            ).scalar()
            logger.info(f"Merged {final_count} unique record pairs")
    finally:
        engine.dispose()

def extract_processed_records_chunked(database_url, job_schema, chunk_size=50000, logger=None, tqdm_flag=False):
    try:
        # Get total count first
        engine = create_engine(database_url)
        with engine.begin() as connection:
            result = connection.execute(
                text(f"SELECT COUNT(*) FROM {job_schema}.processed_records")
            )
            total_records = result.scalar()
        
        if logger:
            logger.info(f"Extracting {total_records} processed records in chunks of {chunk_size}")
        
        # Calculate number of chunks
        num_chunks = (total_records + chunk_size - 1) // chunk_size
        
        # Extract in chunks with progress tracking
        if tqdm_flag:
            with tqdm(total=num_chunks, desc="Extracting processed records", unit="chunk") as pbar:
                for i in range(num_chunks):
                    offset = i * chunk_size
                    
                    query = f"""
                    SELECT * FROM {job_schema}.processed_records 
                    ORDER BY from_id, to_id 
                    LIMIT {chunk_size} OFFSET {offset}
                    """
                    
                    chunk_df = pd.read_sql_query(query, engine)
                    pbar.update(1)
                    
                    if len(chunk_df) > 0:
                        yield chunk_df
                    else:
                        break
        else:
            for i in range(num_chunks):
                offset = i * chunk_size
                
                query = f"""
                SELECT * FROM {job_schema}.processed_records 
                ORDER BY from_id, to_id 
                LIMIT {chunk_size} OFFSET {offset}
                """
                
                chunk_df = pd.read_sql_query(query, engine)
                
                if len(chunk_df) > 0:
                    yield chunk_df
                else:
                    break

    except Exception as e:
        if logger:
            logger.error(f"Error extracting processed records: {e}")
        raise
    finally:
        engine.dispose()

def _worker_create_blocking_keys(db_url, job_schema, records_table, batch_size, 
                               gid_start, gid_end, worker_id):
    """Worker function for parallel blocking key creation"""
    try:
        engine = create_engine(db_url)
        with engine.begin() as connection:
            table_suffix = f"worker_{worker_id}"
            connection.execute(text(
                "SELECT create_blocking_keys_test(:batch_size, :gid_start, :gid_end, :table_suffix, :job_schema, :records_table)"
            ), {
                "batch_size": batch_size,
                "gid_start": gid_start,
                "gid_end": gid_end,
                "table_suffix": table_suffix,
                "job_schema": job_schema, 
                "records_table": records_table
            })
    except Exception as e:
        logging.error(f"Worker {worker_id} error: {e}")
        raise
    finally:
        engine.dispose()

def _worker_compare_records_optimized(db_url, job_schema, records_table,
                                    block_start, block_end, max_block_size,
                                    window_size, overlap, similarity_threshold,
                                    worker_id):
    try:
        engine = create_engine(db_url)
        with engine.begin() as connection:
            connection.execute(text(
                "SELECT compare_records_optimized_parallel(:block_start, :block_end, "
                ":max_block_size, :window_size, :overlap, :similarity_threshold, "
                ":table_suffix, :job_schema, :records_table)"
            ), {
                "block_start": block_start,
                "block_end": block_end,
                "max_block_size": max_block_size,
                "window_size": window_size,
                "overlap": overlap,
                "similarity_threshold": similarity_threshold,
                "table_suffix": str(worker_id),
                "job_schema": job_schema, 
                "records_table": records_table
            })
    except Exception as e:
        logging.error(f"Worker {worker_id} error: {e}")
        raise
    finally:
        engine.dispose()

def _worker_compare_records_exhaustive(db_url, job_schema, records_table,
                                     gid_start, gid_end, worker_id):
    try:
        engine = create_engine(db_url)
        with engine.begin() as connection:
            connection.execute(text(
                "SELECT compare_records_exhaustive_parallel(:gid_start, :gid_end, "
                ":table_suffix, :job_schema, :records_table)"
            ), {
                "gid_start": gid_start,
                "gid_end": gid_end,
                "table_suffix": str(worker_id),
                "job_schema": job_schema,  # Ensure job_schema is passed
                "records_table": records_table
            })
    except Exception as e:
        logging.error(f"Worker {worker_id} error: {e}")
        raise
    finally:
        engine.dispose()

def predict_chunked(database_url, job_schema, relationship, model_directory, output_directory,
                   logger, chunk_size=50000, probable_match_threshold=None, match_threshold=None, tqdm_flag=False):
    try:
        model_path = os.path.join(model_directory, f'rf_{relationship}_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Prepare output directory and file
        os.makedirs(output_directory, exist_ok=True)
        results_path = os.path.join(output_directory, f'predictions_{relationship}.csv')
        
        # Initialize results file with headers
        first_chunk = True
        total_processed = 0
        
        # Process chunks
        for chunk_df in extract_processed_records_chunked(database_url, job_schema, chunk_size, logger, tqdm_flag):
            if len(chunk_df) == 0:
                continue
                
            # Prepare features
            feature_columns = [col for col in chunk_df.columns 
                             if col not in ['from_id', 'to_id', 'relationship', 'similarity_score']]
            
            X = chunk_df[feature_columns]
            
            # Make predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]
            
            # Create results for this chunk
            chunk_results = chunk_df[['from_id', 'to_id']].copy()
            chunk_results[f'relationship_{relationship}'] = predictions
            chunk_results[f'predicted_probability_{relationship}'] = probabilities
            
            # Save results (append mode after first chunk)
            chunk_results.to_csv(
                results_path, 
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False
            )
            
            total_processed += len(chunk_results)
            first_chunk = False
            
        logger.info(f"Generated predictions for {total_processed} record pairs")
        logger.info(f"Saved predictions to {results_path}")
        
        # Return summary results
        summary_results = pd.read_csv(results_path)
        return summary_results
        
    except Exception as e:
        logger.error(f"Error in predict_chunked function: {e}")
        raise
