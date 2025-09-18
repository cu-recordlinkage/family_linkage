import argparse
import logging
import os
import time
import yaml
import pandas as pd
from sqlalchemy import create_engine, text
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from family_linkage_models.common import normalize
from family_linkage_models.prediction import compare, predict_chunked
from family_linkage_models.evaluation import cleanup_old_files

class ProgressTracker:
    def __init__(self, total_steps=8):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        
    def update(self, step_name):
        self.current_step += 1
        progress = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        print(f"\n[Progress {progress:3.0f}%] {step_name} ✓")
        print(f"[Elapsed: {elapsed:.1f}s]")
        return progress

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/test_app.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def cleanup_previous_predictions(output_dir, logger):
    try:
        print("Cleaning up previous prediction results...")
        cleanup_old_files(output_dir, 'predictions_*.csv', logger)
        logger.info("Previous predictions cleaned up")
    except Exception as e:
        logger.warning(f"Error during prediction cleanup: {e}")

def validate_model_exists(relationship_type, model_dir):
    model_path = os.path.join(model_dir, f'rf_{relationship_type}_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path

def main(relationship_type, size_threshold, max_block_size, window_size, 
         overlap, blocking_batch_size, output_dir, num_workers, chunk_size):
    try:
        logger = setup_logging()
        progress = ProgressTracker(total_steps=8)
        
        print(f"\n{'='*60}")
        print(f"ENHANCED FAMILY LINKAGE TEST PIPELINE")
        print(f"{'='*60}")
        print(f"Relationship: {relationship_type}")
        print(f"Workers: {num_workers}")
        print(f"Threshold: {size_threshold}")
        print(f"{'='*60}")
        print(f"\n[Progress   0%] Starting enhanced pipeline...")
        
        # Step 1: Cleanup previous results
        cleanup_previous_predictions(output_dir, logger)
        progress.update("Previous results cleaned")
        
        # Step 2: Load configuration and validate
        config = load_config()
        test_dataset_path = config['data']['test_dataset']
        
        if not os.path.exists(test_dataset_path):
            raise FileNotFoundError(f"Test dataset not found: {test_dataset_path}")
        
        # Validate model exists
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, 'test_pipeline', 'data', 'models')
        model_path = validate_model_exists(relationship_type, model_dir)
        print(f"Using model: {model_path}")
        
        progress.update("Configuration loaded and validated")
        
        # Step 3: Read test dataset with progress
        logger.info(f"Reading test dataset from {test_dataset_path}")
        print("Reading test dataset...")
        
        # Read dataset with progress indication
        dataset = pd.read_csv(
            test_dataset_path,
            usecols=['id', 'last_name', 'phone', 'middle_name', 'zip', 'city', 'dob', 'state', 'address', 'sex', 'ssn'],
            dtype='string'
        )
        logger.info(f"Read {len(dataset)} records")
        progress.update(f"Dataset loaded ({len(dataset)} records)")
        
        # Step 4: Normalize data
        print("Normalizing data with advanced cleaning...")
        cleaned_dataset = normalize(dataset, logger)
        dataset_size = len(cleaned_dataset)
        progress.update(f"Data normalized ({dataset_size} records)")
        
        # Step 5: Setup database
        print("Setting up database connection...")
        db_config = config['database']
        db_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

        try:
            engine = create_engine(db_string)

            # Step 6: Initialize database
            print("Initializing database schema...")
            tables_sql_path = os.path.join(project_root, 'scripts', 'postgres_tables.sql')
            functions_sql_path = os.path.join(project_root, 'scripts', 'postgres_functions.sql')
            
            with open(tables_sql_path, 'r') as file:
                sql_commands = file.read()
            with engine.connect() as connection:
                connection.execute(text(sql_commands))
            
            with open(functions_sql_path, 'r') as file:
                sql_commands = file.read()
            with engine.connect().execution_options(autocommit=True) as connection:
                connection.execute(text(sql_commands))
            
            # Load data to database with progress
            print("Loading normalized data to database...")
            if 'ssn_formatted' in cleaned_dataset.columns:
                cleaned_dataset = cleaned_dataset.rename(columns={'ssn_formatted': 'ssn'})
            
            from sqlalchemy.types import String, Date
            column_types = {
                'id': String, 'last_name': String, 'middle_name': String,
                'ssn': String, 'sex': String, 'dob': Date, 'phone': String,
                'zip': String, 'city': String, 'state': String, 'address': String
            }
            
            # Use tqdm for upload progress (if dataset is large)
            if dataset_size > 10000:
                tqdm.pandas(desc="Uploading to database")
                cleaned_dataset.to_sql('records', engine, if_exists='replace', index=False, dtype=column_types)
            else:
                cleaned_dataset.to_sql('records', engine, if_exists='replace', index=False, dtype=column_types)
            
            logger.info(f"Loaded {dataset_size} records to database")
            progress.update("Database initialized and data loaded")
            
            # Step 7: Run comparison
            print("Starting record comparison phase...")
            processing_start = time.time()
            
            compare(
                database_url=db_string,
                job_schema='public', 
                records_table='records',
                logger=logger,
                size_threshold=size_threshold,
                max_block_size=max_block_size,
                window_size=window_size,
                overlap=overlap,
                blocking_batch_size=blocking_batch_size,
                num_workers=num_workers
            )
            
            processing_time = time.time() - processing_start
            progress.update(f"Record comparison completed ({processing_time:.1f}s)")
            
            # Step 8: Run chunked predictions to save memory
            print(f"Starting chunked prediction phase (chunk size: {chunk_size})...")
            prediction_start = time.time()
            
            results_summary = predict_chunked(
                database_url=db_string,
                job_schema='public',
                relationship=relationship_type,
                model_directory=model_dir,
                output_directory=output_dir,
                logger=logger,
                chunk_size=chunk_size
            )
            
            prediction_time = time.time() - prediction_start
            progress.update(f"Chunked predictions completed ({prediction_time:.1f}s)")
            
        finally:
            engine.dispose()
        
        # Final summary
        total_time = time.time() - progress.start_time
        total_pairs = len(results_summary) if results_summary is not None and not results_summary.empty else 0
        
        print(f"\n{'='*60}")
        print(f"TEST PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Relationship: {relationship_type}")
        print(f"Dataset size: {dataset_size} records")
        print(f"Workers: {num_workers}")
        print(f"Chunk size: {chunk_size} records")
        print(f"Processing strategy: {'Optimized' if dataset_size > size_threshold else 'Exhaustive'}")
        print(f"Comparison time: {processing_time:.2f}s")
        print(f"Prediction time: {prediction_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")
        print(f"Record pairs processed: {total_pairs}")
        print(f"Results: {output_dir}/predictions_{relationship_type}.csv")
        print(f"Log: logs/test_app.log")  
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"\n[ERROR] Enhanced pipeline failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Family Linkage Test Pipeline")
    parser.add_argument('--relationship', type=str, required=True, help="Relationship type")
    parser.add_argument('--size-threshold', type=int, default=10000, help="Size threshold for processing strategy")
    parser.add_argument('--max-block-size', type=int, default=500, help="Max block size for optimized comparison")
    parser.add_argument('--window-size', type=int, default=100, help="Window size for sliding window")
    parser.add_argument('--overlap', type=int, default=50, help="Overlap size for sliding window")
    parser.add_argument('--blocking-batch-size', type=int, default=100000, help="Batch size for blocking operations")
    parser.add_argument('--output-dir', type=str, default='data/predictions', help="Output directory for results")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of parallel workers")
    parser.add_argument('--chunk-size', type=int, default=50000, help="Chunk size for memory-efficient prediction")
    
    args = parser.parse_args()
    
    main(args.relationship, args.size_threshold, args.max_block_size,
         args.window_size, args.overlap, args.blocking_batch_size, 
         args.output_dir, args.num_workers, args.chunk_size)
