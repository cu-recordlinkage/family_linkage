import argparse
import logging
import os
import time
import yaml
import pandas as pd
from sqlalchemy import create_engine, text
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from family_linkage_models.common import normalize
from family_linkage_models.prediction import compare

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
        filename='logs/training_app.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def read_and_clean_data(config, relationship_type, logger):
    try:
        # Read dataset
        dataset = pd.read_csv(
            config['data']['raw_dataset'],
            usecols=['id', 'last_name', 'phone', 'middle_name', 'zip', 'city', 'dob', 'state', 'address', 'sex', 'ssn'],
            dtype='string'
        )
        
        # Read and filter labels
        labels = pd.read_csv(config['data']['raw_labels'], dtype='string')
        filtered_labels = labels[labels['relationship'] == relationship_type]
        
        logger.info(f"Read {len(dataset)} records and {len(filtered_labels)} labels")
        
        # Normalize data using common module
        cleaned_dataset = normalize(dataset, logger)
        
        return cleaned_dataset, filtered_labels
        
    except Exception as e:
        logger.error(f"Error reading/cleaning data: {e}")
        raise

def setup_database(config, cleaned_dataset, labels, logger, project_root):
    try:
        # Create engine
        db_config = config['database']
        db_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(db_string)
        
        # Initialize tables
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
        
        # Load data
        if 'ssn_formatted' in cleaned_dataset.columns:
            cleaned_dataset = cleaned_dataset.rename(columns={'ssn_formatted': 'ssn'})
        
        from sqlalchemy.types import String, Date
        column_types = {
            'id': String, 'last_name': String, 'middle_name': String,
            'ssn': String, 'sex': String, 'dob': Date, 'phone': String,
            'zip': String, 'city': String, 'state': String, 'address': String
        }
        
        cleaned_dataset.to_sql('records', engine, if_exists='replace', index=False, dtype=column_types)
        labels.to_sql('labels', engine, if_exists='replace', index=False)
        
        logger.info("Database setup completed")
        return engine, db_string
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        raise

def main(relationship_type, size_threshold, max_block_size, window_size, 
         overlap, blocking_batch_size, num_workers):
    try:
        logger = setup_logging()
        progress = ProgressTracker(total_steps=8)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        print(f"\n{'='*60}")
        print(f"FAMILY LINKAGE TRAINING PIPELINE")
        print(f"{'='*60}")
        print(f"Relationship: {relationship_type}")
        print(f"Workers: {num_workers}")
        print(f"{'='*60}")
        print(f"\n[Progress   0%] Starting pipeline...")
        
        # Step 1: Load configuration
        config = load_config()
        progress.update("Configuration loaded")
        
        # Step 2: Read and clean data
        cleaned_dataset, labels = read_and_clean_data(config, relationship_type, logger)
        dataset_size = len(cleaned_dataset)
        progress.update(f"Data loaded and normalized ({dataset_size} records)")
        
        # Step 3: Setup database
        engine, db_string = setup_database(config, cleaned_dataset, labels, logger, project_root)
        progress.update("Database initialized")
        
        try:
            # Step 4: Run comparison
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
            progress.update("Record comparison completed")
            
            # Step 5: Process positive pairs
            with engine.connect().execution_options(autocommit=True) as connection:
                connection.execute(text("SELECT process_positive_record_pairs()"))
            progress.update("Positive pairs processed")
            
            # Step 6: Extract data
            processed_records_df = pd.read_sql_query("SELECT * FROM processed_records", engine)
            processed_records_df['relationship'] = 0
            
            processed_positive_df = pd.read_sql_query("SELECT * FROM processed_positive_records", engine)  
            processed_positive_df['relationship'] = 1
            progress.update("Data extracted from database")
            
            # Step 7: Combine and prepare datasets
            combined_df = pd.concat([processed_records_df, processed_positive_df], ignore_index=True)
            
            # Handle ID mapping if needed
            if combined_df['from_id'].astype(str).str.contains('-').any():
                unique_ids = pd.concat([combined_df['from_id'], combined_df['to_id']]).unique()
                id_mapping = {uid: idx + 1 for idx, uid in enumerate(unique_ids)}
                combined_df['from_id'] = combined_df['from_id'].map(id_mapping)
                combined_df['to_id'] = combined_df['to_id'].map(id_mapping)
            else:
                combined_df['from_id'] = combined_df['from_id'].astype(int)
                combined_df['to_id'] = combined_df['to_id'].astype(int)
            
            # Remove duplicates
            combined_df = combined_df.sort_values(by='relationship', ascending=False)
            combined_df = combined_df.drop_duplicates(subset=['from_id', 'to_id'], keep='first')
            progress.update(f"Training dataset prepared ({len(combined_df)} pairs)")
            
            # Step 8: Train model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
            import matplotlib.pyplot as plt
            import joblib
            
            X = combined_df.drop(['from_id', 'to_id', 'relationship', 'similarity_score'], axis=1, errors='ignore')
            y = combined_df['relationship']
            
            # Check class distribution
            logger.info(f"Class distribution:\n{y.value_counts()}")
            print(f"Class distribution:\n{y.value_counts()}")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            logger.info(f"Training set: {len(X_train)} rows, Test set: {len(X_test)} rows")
            print(f"Training set: {len(X_train)} rows, Test set: {len(X_test)} rows")
            
            # Use config parameters if available
            model_config = config.get('models', {}).get(relationship_type)
            
            rf_model = RandomForestClassifier(**model_config)
            logger.info(f"RandomForestClassifier initialized with hyperparameters: {model_config}")
            
            # Train the model
            start_time = time.time()
            rf_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            print(f"--- Running Time for the Model: {training_time:.2f} seconds ---")
            
            # Feature importance
            feature_scores = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            logger.info(f"Feature importance:\n{feature_scores}")
            
            # Create plots directory
            os.makedirs('data/plots', exist_ok=True)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            feature_scores.plot.bar()
            plt.title(f"Feature Importance for {relationship_type} Model")
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.savefig(f"data/plots/feature_importance_{relationship_type}.png")
            plt.close()
            logger.info(f"Feature importance plot saved to data/plots/feature_importance_{relationship_type}.png")
            
            # Predict and evaluate
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Accuracy: {accuracy:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            
            # Classification report
            report = classification_report(y_test, y_pred)
            logger.info(f"Classification Report:\n{report}")
            print("Classification Report:")
            print(report)
            
            # Confusion matrix
            from sklearn.metrics import ConfusionMatrixDisplay
            cm = confusion_matrix(y_test, y_pred)
            ConfusionMatrixDisplay(confusion_matrix=cm).plot()
            plt.title(f"Confusion Matrix for {relationship_type} Model")
            plt.savefig(f"data/plots/confusion_matrix_{relationship_type}.png")
            plt.close()
            logger.info(f"Confusion matrix plot saved to data/plots/confusion_matrix_{relationship_type}.png")
            
            # ROC-AUC score and ROC curve
            y_pred_probs = rf_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_probs)
            logger.info(f"ROC-AUC Score: {auc:.4f}")
            print(f"ROC-AUC Score: {auc:.4f}")
            
            # Calculate and plot ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs, pos_label=1)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {relationship_type} Model')
            plt.legend()
            plt.savefig(f'data/plots/roc_curve_{relationship_type}.png')
            plt.close()
            logger.info(f"ROC curve plot saved to data/plots/roc_curve_{relationship_type}.png")
            
            # Save model
            os.makedirs('data/models', exist_ok=True)
            model_path = f"data/models/rf_{relationship_type}_model.pkl"
            joblib.dump(rf_model, model_path)
            logger.info(f"Model saved to {model_path}")
            progress.update("Model trained and saved")
            
            total_time = time.time() - progress.start_time
            
            print(f"\n{'='*60}")
            print(f"TRAINING PIPELINE COMPLETE")
            print(f"{'='*60}")
            print(f"Relationship: {relationship_type}")
            print(f"Dataset size: {dataset_size} records")
            print(f"Workers: {num_workers}")
            print(f"Processing strategy: {'Optimized' if dataset_size > size_threshold else 'Exhaustive'}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Training dataset: {len(combined_df)} record pairs")
            print(f"Model saved: {model_path}")
            print(f"Log: logs/training_app.log")
            print(f"{'='*60}\n")
            
        finally:
            engine.dispose()
            
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"\n[ERROR] Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Family Linkage Training Pipeline")
    parser.add_argument('--relationship', type=str, required=True, help="Relationship type")
    parser.add_argument('--size-threshold', type=int, default=10000, help="Size threshold")
    parser.add_argument('--max-block-size', type=int, default=500, help="Max block size")
    parser.add_argument('--window-size', type=int, default=100, help="Window size")
    parser.add_argument('--overlap', type=int, default=50, help="Overlap")
    parser.add_argument('--blocking-batch-size', type=int, default=100000, help="Blocking batch size")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of workers")
    
    args = parser.parse_args()
    
    main(args.relationship, args.size_threshold, args.max_block_size, 
         args.window_size, args.overlap, args.blocking_batch_size, args.num_workers)
