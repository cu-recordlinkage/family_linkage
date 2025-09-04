
CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    last_name TEXT,
    middle_name TEXT,
    ssn TEXT,
    sex TEXT,
    dob DATE,
    phone TEXT,
    zip TEXT,
    city TEXT,
    state TEXT,
    address TEXT
);

CREATE TABLE IF NOT EXISTS labels (
    from_id TEXT,
    to_id TEXT,
    relationship TEXT
);