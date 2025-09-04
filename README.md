# Family Linkage

A Python package for record linkage and relationship detection using machine learning and advanced blocking techniques. This project provides parallel processing capabilities for efficient comparison of large datasets and includes both training and testing pipelines.

## Overview

Family Linkage Package is designed to identify potential relationships between records in large datasets. The system uses:

- Advanced text normalization and cleaning
- Multi-strategy blocking techniques (name-based, address-based, demographic-based)
- Parallel processing with worker-based architecture
- Machine learning models (Random Forest) for relationship prediction
- PostgreSQL functions for optimized database operations
- Sliding window comparisons for memory efficiency

## Project Structure

```
family_linkage_project/
├── family_linkage_models/              # Core package (for pip install)
│   ├── common.py                       # normalize() function
│   ├── prediction.py                   # compare() and predict() functions
│   ├── training.py                     # training functions
└── scripts/
│       ├── postgres_functions.sql      # Parameterized SQL functions
│       └── postgres_tables.sql         # Table definitions
│
├── training_pipeline/                  # Training pipeline
│   ├── main.py                         # Training script
│   ├── config.yaml                     # Training configuration
│   ├── data/
│   │   ├── raw/                        # Raw training data
│   │   ├── processed/                  # Processed data
│   │   ├── models/                     # Saved ML models
│   │   └── plots/                      # Training visualizations
│   └── logs/                           # Training logs
│
├── test_pipeline/                      # Testing/prediction pipeline
│   ├── main.py                         # Testing script
│   ├── config.yaml                     # Testing configuration
│   ├── data/
│   │   ├── test/                       # Test datasets
│   │   ├── models/                     # RF Models
│   │   └── predictions/                # Prediction results
│   └── logs/                           # Testing logs
│
└── setup.py                           # Package installation


## Installation

### Prerequisites

- Python 3.10
- PostgreSQL database

### Install from Source

1. Clone the repository:
```bash
git clone <repository-url>
cd family_linkage_project
```

2. Install the package in development mode:
```bash
pip install -e .
```

This will install the package with all dependencies including:
- pandas, numpy, scikit-learn
- PostgreSQL drivers (psycopg2-binary)
- SQLAlchemy for database operations
- Visualization libraries (matplotlib, seaborn)

### Database Setup

1. Create a PostgreSQL database
2. Update the database configuration in `config.yaml` files
3. The application will automatically create required tables and functions

## Quick Start

### 1. Training a Model

```bash
cd training_pipeline
python main.py --relationship sibling --num-workers 4
```

### 2. Testing/Prediction

```bash
cd test_pipeline  
python main.py --relationship sibling --num-workers 4 --chunk-size 50000
```

## Usage

### Training Pipeline

The training pipeline processes labeled data to create machine learning models:

**Key Features:**
- Reads raw dataset and relationship labels
- Performs advanced data normalization
- Creates blocking keys for efficient comparisons
- Generates positive and negative training examples
- Trains Random Forest classifier
- Saves model and evaluation plots

**Required Data Files:**
- Raw dataset: CSV with columns `id`, `last_name`, `middle_name`, `ssn`, `sex`, `dob`, `phone`, `zip`, `city`, `state`, `address`
- Labels file: CSV with columns `from_id`, `to_id`, `relationship`

### Test Pipeline

The test pipeline applies trained models to new data:

**Key Features:**
- Loads and normalizes test dataset
- Performs record comparison using optimized or exhaustive strategy
- Applies trained ML model in memory-efficient chunks
- Outputs prediction results with probabilities

## Command Line Arguments

### Training Pipeline Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--relationship` | str | **Required** | Relationship type to train |
| `--size-threshold` | int | 10000 | Dataset size threshold for processing strategy |
| `--max-block-size` | int | 500 | Maximum block size for optimized comparison |
| `--window-size` | int | 100 | Window size for sliding window comparison |
| `--overlap` | int | 50 | Overlap size for sliding window |
| `--blocking-batch-size` | int | 100000 | Batch size for blocking operations |
| `--num-workers` | int | 4 | Number of parallel workers |

### Test Pipeline Arguments

All training arguments plus:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | str | data/predictions | Output directory for results |
| `--chunk-size` | int | 50000 | Chunk size for memory-efficient prediction |

### Example Commands
### 1. Training Pipeline

cd training_pipeline
python main.py --relationship sibling --num-workers 4


### 2. Test Pipeline

cd test_pipeline
python main.py --relationship sibling --size-threshold 10000 --max-block-size 500 --num-workers 4 --output-dir data/predictions

### Processing Strategies

The system automatically selects processing strategy based on dataset size:

- **Exhaustive Comparison** (≤ size_threshold): All-vs-all comparison
- **Optimized Comparison** (> size_threshold): Uses blocking and sliding windows

### Memory Management

- **Chunked Processing**: Large datasets processed in configurable chunks
- **Parallel Workers**: Distributes workload across multiple processes

### Schema Management

- Tables and functions are created automatically
- Uses parameterized schemas for multi-tenant support
- Temporary tables are cleaned up after processing

## Output Files

### Training Pipeline
- `data/models/rf_{relationship}_model.pkl`: Trained model
- `data/plots/feature_importance_{relationship}.png`: Feature importance plot
- `data/plots/confusion_matrix_{relationship}.png`: Confusion matrix
- `data/plots/roc_curve_{relationship}.png`: ROC curve
- `logs/training_app.log`: Training log

### Test Pipeline
- `data/predictions/predictions_{relationship}.csv`: Prediction results with probabilities
- `logs/test_app.log`: Testing log

## Advanced Features

### Error Handling
- Comprehensive logging at all stages
- Handling of missing data
- Database connection pooling and retry logic
- Progress tracking with elapsed time

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce chunk size and increase workers
2. **Database Connection**: Check PostgreSQL service and credentials
3. **Slow Performance**: Increase worker count and optimize PostgreSQL settings
4. **Missing Models**: Ensure training completed successfully

### Performance Monitoring
- Check logs for processing times and memory usage
- Monitor PostgreSQL performance during processing
- Use system tools to track CPU and memory utilization
