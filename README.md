# OSYN: Optimized SYNthetic data for Evaluating Generalization ability

## Overview

OSYN is a comprehensive framework for generating optimal synthetic data using Generative Adversarial Networks (GANs) with Lower bounds. This is the main implementation of the paper **"Using Synthetic Data to estimate the True Error is theoretically and practically doable"**, which was conditionally accepted at the Journal track in ACML 2025. 
<!-- The project implements a novel approach that combines CTGAN (Conditional Tabular GAN) with clustering-based optimization to generate high-quality synthetic data that minimizes prediction loss for machine learning classifiers. -->

> **📄 Paper Status**: Conditionally Accepted at ACML 2025 (Journal Track)  
> **🚀 Full Release**: After conference publication

## Key Features

- **CTGAN Integration**: Uses Conditional Tabular GAN for generating realistic synthetic data
- **Clustering-based Optimization**: Implements FAISS-based clustering for optimal data distribution
- **Lower Bounds**: Provides mathematical guarantees for synthetic data quality
- **Multiple Classifier Support**: Compatible with various ML classifiers (Random Forest, SVM, Decision Tree, etc.)
- **Comprehensive Evaluation**: Includes bootstrap, theoretical, and empirical evaluation methods
- **Configurable Parameters**: Extensive configuration system for fine-tuning experiments

## Project Structure

```
OSYN/
├── main.py                    # Main execution script (original version)
├── main_b.py                  # Enhanced main script with parameter sweeps
├── config.py                  # Main configuration file
├── classifier_config.py       # Classifier-specific hyperparameters
├── get_data.py               # Data loading and preprocessing utilities
├── losses.py                 # Loss function implementations
├── finetune_classifiers.py   # Classifier training and evaluation
├── finetune_gan.py           # CTGAN training and management
├── clustering_data.py        # Clustering and data distribution algorithms
├── calculate_lower_bound.py  # Lower bound calculation and uncertainty estimation
├── build_models.py           # Model factory for various classifiers
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── Datasets/                 # Data storage directory
│   ├── Bank_Marketing/       # Bank Marketing dataset (UCI Repository #222)
│   ├── Distribution_data/    # Precomputed distribution files
│   ├── Loss_data/           # Loss calculation results
│   ├── Optim_data/          # Optimization results
│   └── Temp_cluster/        # Temporary clustering files
└── GAN_models/              # Pre-trained GAN models
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Dependencies

#### Option 1: Install from requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

#### Option 2: Manual installation
```bash
pip install torch torchvision
pip install ctgan
pip install scikit-learn
pip install pandas numpy
pip install faiss-cpu  # Note: Use faiss-cpu instead of faiss-gpu
pip install ucimlrepo
pip install yacs
pip install tqdm
pip install scipy
pip install joblib
```

#### Requirements Overview
The project requires the following key dependencies:
- **Core**: numpy, pandas, torch, scipy
- **ML**: scikit-learn, ctgan
- **Data**: ucimlrepo, yacs, tqdm, joblib
- **Clustering**: faiss-cpu (or faiss-gpu for GPU acceleration)

#### Version Compatibility
The `requirements.txt` file specifies minimum versions for compatibility:
- Python 3.7+ required
- PyTorch 1.9.0+ for deep learning operations
- Scikit-learn 1.0.0+ for ML algorithms
- FAISS 1.7.0+ for efficient clustering

## Quick Start

### Basic Usage

1. **Run the original version:**
```bash
python main.py
```

2. **Run the enhanced version with parameter sweeps:**
```bash
python main_b.py
```

### Configuration

The project uses YACS configuration system. Key parameters can be modified in `config.py`:

```python
# Data configuration
_C.DATA.DATASET_ID = 222                    # UCI Repository dataset ID
_C.DATA.ORACLE_RATIO = 0.7                 # Ratio of oracle points
_C.DATA.TOTAL_PARTITONS = 500              # Total partitions for clustering

# GAN configuration
_C.GAN.EPOCHS = 3                          # Training epochs
_C.GAN.BATCH_SIZE = 500                    # Batch size for training

# Clustering configuration
_C.CLUSTER.NUM_CENTROIDS = 500             # Number of cluster centroids
_C.CLUSTER.ADJUST_FACTOR = 0.7             # Adjustment factor for distribution

# Lower bound configuration
_C.LOWER_BOUND.DELTA_1 = 0.01              # Delta 1 parameter
_C.LOWER_BOUND.DELTA_2 = 0.0005            # Delta 2 parameter
_C.LOWER_BOUND.NUM_ITERATIONS = 15         # Number of iterations
```

## Methodology

### 1. Data Preparation
- Loads Bank Marketing dataset from UCI Repository
- Splits data into training, small test, and oracle sets
- Preprocesses categorical and numerical features
- Handles missing values and feature encoding

### 2. GAN Training
- Trains CTGAN on oracle dataset
- Generates synthetic data samples
- Evaluates synthetic data quality using downstream classifiers

### 3. Clustering and Distribution
- Creates cluster centroids using FAISS
- Computes radius for each cluster
- Calculates polynomial distribution of synthetic data
- Optimizes data distribution across clusters

### 4. Lower Bound Calculation
- Implements theoretical lower bound for synthetic data quality
- Calculates uncertainty measures (uncertainty1, uncertainty2)
- Provides mathematical guarantees for data quality

### 5. Baselines Methods
- **Bootstrap Method**: Statistical evaluation using resampling
- **Syn-wo-Opt Method**: Direct evaluation on synthetic data

## Key Algorithms

### Clustering Algorithm
```python
def cluster_syn_data(df, eta, preprocessor_cluster, index, batch_size=10000):
    # Clusters synthetic data points into optimal partitions
    # Uses FAISS for efficient nearest neighbor search
    # Returns clustered data and partition assignments
```

### Lower Bound Calculation
```python
def calculate_lower_bound(loss_df, delta1, delta2, Pg, check_flag):
    # Calculates theoretical lower bound for synthetic data quality
    # Implements uncertainty estimation
    # Returns comprehensive quality metrics
```

### Distribution Optimization
```python
def choose_optim_data(result_df, folder_csv, no_syn, no_syn_adj, cls_models):
    # Selects optimal synthetic data points for each partition
    # Minimizes prediction loss while maintaining distribution
    # Returns optimized data selection
```

## Output Files

### Results
- `final_result_{model}_{delta2}_{b_value}_iter_{iteration}.csv`: Main results
- `runtime_result_{model}_{delta2}_{b_value}_iter_{iteration}.csv`: Runtime metrics
- `result_{b_value}_{classifier}.txt`: Detailed per-classifier results

### Intermediate Files
- `clustered_data_{delta2}_{b_value}_iter_{iteration}.csv`: Clustered synthetic data
- `cluster_assignments_{delta2}_{b_value}_iter_{iteration}.txt`: Cluster assignments
- `syn_dist_500.npy`: Precomputed distribution data
- `opt_syn_total_50000_partion_500.npy`: Optimal point distribution

## Performance Metrics

The framework tracks several key metrics:

- **F(G,h)**: Average loss on synthetic data
- **Epsilon**: Average difference between small set and synthetic data loss
- **Lower Bound**: Theoretical quality guarantee
- **Bootstrap Loss**: Statistical evaluation (baseline)

## Advanced Usage

### Parameter Sweeps
The enhanced version (`main_b.py`) supports parameter sweeps:

```python
params_ls = [[0.01, 1], [0.01, 1.5],
             [0.001, 1], [0.001, 2],
             [0.0001, 1.5], [0.0001, 2]]

# Each parameter set: [delta_2, b_value]
# delta_1 = 0.2 - delta_2
```

### Custom Classifiers
Add new classifiers by modifying `classifiers` dictionary in `finetune_classifiers.py`:

```python
classifiers = {
    "Random Forest": get_rf_classifier(max_depth=5, n_estimators=10),
    "SVM": get_svm_classifier(),
    "Decision Tree": get_dt_classifier(),
    # Add your custom classifier here
}
```

### Custom Datasets
To use a different dataset:

1. Update `DATASET_ID` in `config.py`
2. Modify data preprocessing in `get_data.py` if needed
3. Adjust clustering parameters based on data characteristics

## Troubleshooting

### Common Issues

1. **FAISS Installation**: Use `faiss-cpu` instead of `faiss-gpu` for compatibility
2. **Memory Issues**: Reduce `BATCH_SIZE` and `NUM_SYN_POINTS_PER_EPOCH`
3. **CUDA Issues**: Set `DEVICE.NAME = "cpu"` in config for CPU-only execution
4. **Convergence Issues**: Increase `NUM_ITERATIONS` or adjust learning rates

### Performance Optimization

- Use GPU acceleration when available
- Adjust batch sizes based on available memory
- Precompute distributions for faster execution
- Use parallel processing for classifier training

## Citation

This code implements the paper **"Using Synthetic Data to estimate the True Error is theoretically and practically doable"** (conditionally accepted at ACML 2025 Journal track).

**Note**: The full code will be released after the conference. If you use this code in your research, please cite our paper once it's officially published.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions and support, please contact [tungn1201@gmail.com] or open an issue on GitHub.

## Acknowledgments

- **ACML 2025** for accepting our paper "Using Synthetic Data to estimate the True Error is theoretically and practically doable"
- UCI Machine Learning Repository for the Bank Marketing dataset
- CTGAN authors for the synthetic data generation framework
- FAISS team for efficient similarity search
- Scikit-learn contributors for machine learning utilities