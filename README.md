# OSYN
This repo is of ACML 2025 paper "**Using Synthetic Data to estimate the True Error is theoretically and practically doable**".

**OSYN is a theoretically grounded framework** that selects (and optimizes) synthetic test data from a generator to evaluate a trained model, yielding **reliable estimates** of its **true test error** when real test sets are scarce.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Detailed Usage](#detailed-usage)
- [Advanced Usage](#advanced-usage)
- [Output Files](#output-files)
- [Citation](#citation)

## Project Structure

```
OSYN/
├── main.py                    # Main execution script
├── main_b.py                  # Parameter sweep script (recommended)
├── config.py                  # Configuration parameters
├── classifier_config.py       # Classifier hyperparameters
├── get_data.py               # Data loading and preprocessing
├── losses.py                 # Loss functions (0-1 loss)
├── finetune_classifiers.py   # Classifier training
├── finetune_gan.py           # CTGAN training and loading
├── clustering_data.py        # FAISS clustering algorithms
├── calculate_lower_bound.py  # True model loss estimation
├── build_models.py           # Classifier factory functions
├── requirements.txt          # Python dependencies
├── Datasets/                 # Data storage
│   ├── Bank_Marketing/       # UCI Bank Marketing dataset (ID: 222)
│   │   └── 300_200/         # Train/small/oracle splits
│   ├── Distribution_data/    # Precomputed distributions (Pg, optimal counts)
│   ├── Loss_data/           # Loss CSVs (oracle_loss_0_1.csv, small_loss_0_1.csv, syn_loss_0_1.csv)
│   ├── Optim_data/          # Optimization results
│   └── Temp_cluster/        # Temporary cluster assignments
└── GAN_models/              # CTGAN models (ctgan_bank.pt)
```

## Installation

### Prerequisites
- Python 3.7+
- CUDA 9.0+ (optional, for GPU acceleration)
- 8GB+ RAM

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd OSYN
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv osyn_env
source osyn_env/bin/activate  # On Windows: osyn_env\Scripts\activate

# Or using conda
conda create -n osyn_env python=3.8
conda activate osyn_env
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation
- `torch>=1.9.0` - Deep learning framework
- `scipy>=1.7.0` - Scientific computing
- `scikit-learn>=1.0.0` - Machine learning algorithms
- `ctgan>=0.5.0` - Conditional Tabular GAN
- `faiss-cpu>=1.7.0` - Efficient clustering (use `faiss-gpu` for GPU)
- `ucimlrepo>=0.0.3` - UCI dataset access
- `yacs>=0.1.8` - Configuration management
- `tqdm>=4.62.0` - Progress bars
- `joblib>=1.1.0` - Parallel processing

## Quick Start

### Option 1: Run Main Script (Single Configuration)
```bash
python main.py
```

This runs the full pipeline with default configuration (DELTA_1=0.01, DELTA_2=0.0005, ADJUST_FACTOR=0.7) for 15 iterations.

### Option 2: Run Parameter Sweep (Recommended)
```bash
python main_b.py
```

This runs multiple experiments with parameter combinations:
- `delta_2 ∈ {0.01, 0.001, 0.0001}`
- `b_value ∈ {1, 1.5, 2}`
- `delta_1 = 0.2 - delta_2`

Each combination runs for 5 iterations.

## Configuration

All configuration parameters are managed through `config.py` using YACS.

### Data Configuration (`_C.DATA`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_ID` | 222 | UCI Repository ID for Bank Marketing dataset |
| `ORACLE_RATIO` | 0.7 | Ratio of test data to use as oracle set |
| `TOTAL_PARTITONS` | 500 | Number of partitions/clusters |
| `NUMBER_DOMINATE` | 300 | Number of points in dominant class for small set |
| `SAVE_DIR` | 'Datasets' | Directory for saving datasets |
| `NUM_DEL_MISSING_COLS` | 2 | Number of columns with most missing values to remove |
| `SEED` | 42 | Random seed for reproducibility |

### GAN Configuration (`_C.GAN`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 3 | Training epochs for CTGAN |
| `BATCH_SIZE` | 500 | Batch size for GAN training |
| `GENERATOR_DIM` | (256, 256) | Generator network dimensions |
| `DISCRIMINATOR_DIM` | (256, 256) | Discriminator network dimensions |
| `GENERATOR_LR` | 2e-4 | Generator learning rate |
| `DISCRIMINATOR_LR` | 2e-4 | Discriminator learning rate |
| `DISCRIMINATOR_STEPS` | 1 | Discriminator update steps per generator step |
| `PAC` | 10 | Number of samples to group in PAC (Packing) |
| `SAVE_DIR` | 'GAN_models/ctgan_bank.pt' | Path to save/load GAN model |

### Clustering Configuration (`_C.CLUSTER`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_CENTROIDS` | 500 | Number of cluster centroids (should match TOTAL_PARTITONS) |
| `PERCENT_LIMIT` | 10 | Percentage of nearest clusters for radius calculation |
| `NUM_ITERS_CAL_DISTRIBUTION` | 30 | Iterations for computing Pg (distribution) |
| `NUM_POINTS_CAL_DISTRIBUTION` | 50000 | Points per iteration for Pg estimation |
| `OPT_NUM_POINTS` | 50000 | Total synthetic points for optimal distribution |
| `ADJUST_FACTOR` | 0.7 | Distribution adjustment factor (b in paper) |
| `BATCH_SIZE` | 10000 | Batch size for clustering operations |
| `SAVE_DIR_DISTRIBUTION` | 'Datasets/Distribution_data/syn_dist_500.npy' | Path for Pg |
| `SAVE_DIR_OPT_NUM_POINTS` | 'Datasets/Distribution_data/opt_syn_total_50000_partion_500.npy' | Path for optimal counts |

### Lower Bound Configuration (`_C.LOWER_BOUND`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_SYN_POINTS_PER_EPOCH` | 50000 | Synthetic points generated per iteration |
| `DELTA_1` | 0.01 | First confidence parameter (δ₁) |
| `DELTA_2` | 0.0005 | Second confidence parameter (δ₂) |
| `NUM_ITERATIONS` | 15 | Number of optimization iterations |
| `NUM_WO_ITERATIONS` | 5 | Iterations for synthetic-without-optimization baseline |
| `NUM_BOOTSTRAP_ITERATIONS` | 10000 | Bootstrap samples for baseline |

### Device Configuration (`_C.DEVICE`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NAME` | "cuda" | Device: "cuda" for GPU, "cpu" for CPU |
| `NUM_GPUS` | 1 | Number of GPUs to use |

## Detailed Usage

### Pipeline Overview

An example dataset (Bank Marketing dataset) executes the following steps:

1. **Data Preparation** (`prepare_data`)
   - Loads or downloads Bank Marketing dataset from UCI (ID: 222)
   - Removes columns with excessive missing values
   - Splits into: train set, small test set (500 points: 300 dominant class + 200 minority), oracle set
   - Saves splits to `Datasets/Bank_Marketing/300_200/`

2. **Classifier Training** (`prepare_classifiers`)
   - Preprocesses data (StandardScaler for numerical, OneHotEncoder for categorical)
   - Trains classifiers (default: Random Forest with max_depth=5, n_estimators=10)
   - Evaluates on oracle set, small set. The results are saved in `Datasets/Loss_data`.

3. **GAN Training/Loading** (`prepare_gan`)
   - If `GAN_models/ctgan_bank.pt` doesn't exist: trains CTGAN on oracle set for 3 epochs
   - If exists: loads pre-trained model
   - Evaluates synthetic data quality using Random Forest

4. **Clustering Setup** (`prepare_clustering`)
   - Creates FAISS index with `NUM_CENTROIDS` centroids from small test set.
   - Computes radius for each partition using 10% nearest neighbors.
   - Calculates or loads polynomial distribution Pg.
   - Calculates or loads optimal synthetic counts per partition.
   - Adjusts optimal counts using ADJUST_FACTOR (b).

5. **Iterative True Model Loss Estimation** (`run_one_epoch_lower_bound`)
   
   Per iteration (repeated 15 times by default):
   
   a. **Generate Synthetic Data**: Sample 50K points from CTGAN
   
   b. **Cluster Data**: Assign each synthetic point to nearest centroid
   
   c. **Calculate Losses**: Compute 0-1 loss for each synthetic point using trained classifiers
   
   d. **Optimize Selection**: For each partition, select synthetic points that minimize loss difference with small test set
   
   e. **Estimate True loss**: Calculate uncertainty terms and OSYN-based true loss using paper's formula.
   
   f. **Baseline Comparisons**:
      - **Syn-wo-Opt**: Generate g synthetic points uniformly, compute average 0-1 loss
      - **Bootstrap**: Resample small test set with replacement, compute average loss
     
6. **Results Aggregation**
   - Saves per-model, per-iteration results to CSV
   - Outputs F(G,h), ε, uncertainties, true model loss, and baseline comparisons

### Running with Custom Parameters

#### Modify config.py for Single Run
```python
# Edit config.py
_C.LOWER_BOUND.DELTA_1 = 0.05  # Change δ₁
_C.LOWER_BOUND.DELTA_2 = 0.001 # Change δ₂
_C.CLUSTER.ADJUST_FACTOR = 0.8  # Change b

# Then run
python main.py
```

#### Modify main_b.py for Parameter Sweeps
```python
# Edit main_b.py
params_ls = [
    [0.01, 1],      # [delta_2, b_value]
    [0.005, 1.5],
    [0.001, 2],
]
delta = 0.2  # δ₁ = delta - delta_2

# Then run
python main_b.py
```

### Understanding the Loss Computation

The framework uses **0-1 loss**:

```python
# For a single sample
def zero_one_loss_element(x, y, clf):
    y_pred = clf.predict(x)
    loss = np.abs(y_pred - y).item()  # 0 if correct, 1 if wrong
    return loss

# For a dataset
def zero_one_loss_set(X, y, clf):
    y_pred = clf.predict(X)
    loss = np.mean(np.abs(y_pred - y))  # Average 0-1 loss
    return loss
```

Loss files store per-sample losses:
- `oracle_loss_0_1.csv`: Loss for each oracle point
- `small_loss_0_1.csv`: Loss for each small test point  
- `syn_loss_0_1.csv`: Loss for each synthetic point (generated per iteration)

## Advanced Usage

### Using a Custom Dataset

#### From UCI Repository

1. Find dataset ID from [UCI ML Repository](https://archive.ics.uci.edu/)

2. **Modify `config.py`:**
```python
_C.DATA.DATASET_ID = YOUR_DATASET_ID  # e.g., 45 for Adult
_C.DATA.NUMBER_DOMINATE = 300  # Adjust based on class balance
```

3. **Check `get_data.py`** - May need custom preprocessing:
```python
# If your dataset has different target column name
def split_data(df, ...):
    # Change this line
    df_no = df_test[df_test['target_col_name'] == 'class_0']
    df_yes = df_test[df_test['target_col_name'] == 'class_1']
    ...
```

4. **Delete existing data directory** to force re-download:
```bash
rm -rf Datasets/Bank_Marketing/
python main.py
```

#### From Custom CSV File

1. **Prepare CSV** with features and target column

2. **Modify `get_data.py`:**
```python
def get_dataset(...):
    # Replace fetch_dataset with custom loader
    df = pd.read_csv('path/to/your/data.csv')
    df = filter_null(df, num_dels)
    df_train, df_test, df_small, df_oracle = split_data(df, ratio, seed, total_points, number_dominate)
    ...
```

3. **Update `split_data` function** for your target column:
```python
def split_data(df, ...):
    df_train, df_test = train_test_split(df, test_size=ratio, random_state=seed)
    
    # Modify these based on your binary target
    df_no = df_test[df_test['your_target_col'] == 0]
    df_yes = df_test[df_test['your_target_col'] == 1]
    ...
```

### Finetuning the GAN

#### Train from Scratch

1. **Delete pre-trained model:**
```bash
rm GAN_models/ctgan_bank.pt
```

2. **Adjust GAN parameters in `config.py`:**
```python
_C.GAN.EPOCHS = 10           # Increase epochs for better quality
_C.GAN.BATCH_SIZE = 500
_C.GAN.GENERATOR_DIM = (512, 512)  # Larger network
_C.GAN.DISCRIMINATOR_DIM = (512, 512)
_C.GAN.GENERATOR_LR = 1e-4   # Lower learning rate
_C.GAN.PAC = 10              # Packing parameter
```

3. **Run training:**
```bash
python main.py
```

The GAN is trained by `finetune_gan()` in `finetune_gan.py`:
```python
def finetune_gan(df_gan, save_dir='GAN_models/ctgan_bank.pt'):
    discrete_columns, num_columns = find_columns(df_gan)
    
    gan_wrapper = CTGAN(
        epochs=cfg.GAN.EPOCHS,
        cuda=True,
        batch_size=cfg.GAN.BATCH_SIZE,
        generator_dim=cfg.GAN.GENERATOR_DIM,
        discriminator_dim=cfg.GAN.DISCRIMINATOR_DIM,
        ...
    )
    gan_wrapper.fit(df_gan, discrete_columns=discrete_columns)
    save_gan(gan_wrapper, save_dir)
    return gan_wrapper
```

#### Evaluate Synthetic Data Quality

After training, evaluation happens automatically:
```python
# In main.py -> prepare_gan()
n = len(df_gan)
syn_wrap = model.sample(n)
scores_wrap = eval_synthetic(df_gan, syn_wrap)
print("CTGAN wrapper:", scores_wrap)  # {'accuracy': ..., 'roc_auc': ...}
```

To manually evaluate:
```python
from finetune_gan import eval_synthetic

real_data = pd.read_csv('Datasets/Bank_Marketing/300_200/df_oracle.csv')
synthetic_data = model.sample(len(real_data))
scores = eval_synthetic(real_data, synthetic_data)
print(f"Accuracy: {scores['accuracy']}, ROC-AUC: {scores['roc_auc']}")
```

### Adding/Modifying Classifiers

The default configuration uses **only Random Forest**. To enable other classifiers:

1. **Edit `finetune_classifiers.py`:**
```python
classifiers = {
    "Random Forest": get_rf_classifier(max_depth=5, n_estimators=10, max_features=1),
    "Linear SVM": get_svm_classifier(),  # Uncomment
    "Decision Tree": get_dt_classifier(),  # Uncomment
    "Neural Net": get_mlp_classifier(),  # Uncomment
    "Logistic Regression": get_lr_classifier(),  # Uncomment
    "Gradient Boosting": get_gb_classifier(learning_rate=0.1, max_depth=3),  # Uncomment
}
```

2. **Customize hyperparameters in `classifier_config.py`:**
```python
# Example: Tune Random Forest
_C.CLS.RF.N_ESTIMATORS = 50  # Increase trees
_C.CLS.RF.MAX_DEPTH = 10     # Deeper trees
_C.CLS.RF.MAX_FEATURES = 'sqrt'  # sqrt of features

# Example: Tune Neural Net
_C.CLS.MLP.HIDDEN_LAYER_SIZES = (100, 50, 25)
_C.CLS.MLP.MAX_ITER = 1000
_C.CLS.MLP.EARLY_STOPPING = True
```

3. **Add custom classifier in `build_models.py`:**
```python
from sklearn.ensemble import AdaBoostClassifier

def get_adaboost_classifier(n_estimators=50, learning_rate=1.0):
    clf = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=hp.CLS.RANDOM_STATE
    )
    return clf
```

Then register in `finetune_classifiers.py`:
```python
classifiers = {
    "Random Forest": get_rf_classifier(...),
    "AdaBoost": get_adaboost_classifier(n_estimators=50),
}
```

<!-- ### Adjusting Clustering Parameters

#### Finding Optimal Number of Clusters

```python
# Create test script
import numpy as np
from clustering_data import *
from get_data import *

# Load data
df_oracle = load_dataset('Datasets/Bank_Marketing/300_200/df_oracle.csv')
df_small = load_dataset('Datasets/Bank_Marketing/300_200/df_small.csv')

preprocessor_cluster, dim = get_transform(df_oracle)

# Test different numbers
for num_centroids in [100, 200, 500, 1000]:
    index, centroids = get_centroids_partion(
        df_small, preprocessor_cluster, dim, num_centroids=num_centroids
    )
    eta = get_radius_centroids(df_small, index, preprocessor_cluster, num_centroids=num_centroids)
    print(f"\nCentroids: {num_centroids}")
    print(f"  Mean radius: {np.mean(eta):.4f}")
    print(f"  Std radius: {np.std(eta):.4f}")
```

#### Modify Clustering Parameters

```python
# In config.py
_C.CLUSTER.NUM_CENTROIDS = 1000  # More granular partitions
_C.CLUSTER.PERCENT_LIMIT = 5     # Tighter radius (use 5% nearest neighbors)
_C.CLUSTER.ADJUST_FACTOR = 0.9   # More aggressive optimization

# Delete precomputed distributions
rm Datasets/Distribution_data/syn_dist_500.npy
rm Datasets/Distribution_data/opt_syn_total_50000_partion_500.npy

# Rerun
python main.py
``` -->

### Customizing Lower Bound Calculation

The lower bound is computed in `calculate_lower_bound.py`:

```python
def calculate_lower_bound(loss_df, delta1, delta2, Pg, check_flag):
    """
    Computes lower bound on true test error
    
    Args:
        loss_df: DataFrame with optimized synthetic losses per partition
        delta1: First confidence parameter (δ₁)
        delta2: Second confidence parameter (δ₂)
        Pg: Probability distribution over partitions
        check_flag: Whether all partitions have sufficient synthetic data
    
    Returns:
        dict with F_g_h, epsilon, uncertainty1, uncertainty2, lower_bound, etc.
    """
    # Implementation calculates:
    # - F(G,h): Average loss on optimized synthetic data
    # - ε: Average difference between small test and synthetic losses
    # - uncertainty1: First uncertainty term
    # - uncertainty2: Second uncertainty term
    # - lower_bound = F(G,h) - ε - uncertainty1 + uncertainty2
    ...
```

<!-- To experiment with different confidence parameters:

```python
# In main_b.py, modify params_ls
params_ls = [
    [0.02, 1],      # Higher δ₂ = wider confidence intervals
    [0.005, 1.5],   # Lower δ₂ = tighter bounds
    [0.0001, 2],    # Very low δ₂, high b
]

# δ₁ = 0.2 - δ₂
``` -->

### Running on CPU Only

```python
# In config.py
_C.DEVICE.NAME = "cpu"

# Or set environment variable
export CUDA_VISIBLE_DEVICES=""
python main.py
```

### Parallel Processing

The framework already uses parallel processing in several places:

1. **Classifier loss computation** (batch processing in `finetune_classifiers.py`)
2. **Optimization** (joblib parallelization in `calculate_lower_bound.py`)

To adjust parallelization:
```python
# In calculate_lower_bound.py
n_cores = multiprocessing.cpu_count()
# Parallel(..., n_jobs=n_cores)  # Use all cores
# Parallel(..., n_jobs=4)  # Use 4 cores
```

## Output Files

### Main Results

| File | Description |
|------|-------------|
| `Datasets/final_result_{model}_{delta2}_{b_value}_iter_{i}.csv` | Per-classifier, per-iteration results with F(G,h), ε, uncertainties, lower bounds, and baselines |
| `Datasets/runtime_result_{model}_{delta2}_{b_value}_iter_{i}.csv` | Runtime metrics for each component |
| `Datasets/Temp_cluster/result_{b_value}_{classifier}.txt` | Detailed text log per classifier |

**Columns in `final_result_*.csv`:**
- `Model`: Classifier name
- `Iteration`: Iteration number (0-14 by default)
- `b value`: Adjustment factor (ADJUST_FACTOR)
- `Delta_1`, `Delta_2`: Confidence parameters
- `F(G,h)`: Average 0-1 loss on optimized synthetic data
- `Epsilon`: Average absolute difference between small test and synthetic losses
- `Uncertainty_1`: First uncertainty term
- `Uncertainty_2`: Second uncertainty term
- `Lower bound`: Estimated true model error 
- `Loss without Opt Lower bound`: Baseline using uniformly sampled synthetic data (mean ± std)
- `Loss with Boostrap`: Bootstrap baseline
- `Loss Small`: Average 0-1 loss on small test set
- `Loss Oracle`: Average 0-1 loss on oracle set (ground truth)

### Intermediate Files

| File | Location | Description |
|------|----------|-------------|
| `df_train.csv` | `Datasets/Bank_Marketing/300_200/` | Training data |
| `df_small.csv` | `Datasets/Bank_Marketing/300_200/` | Small test set (500 points) |
| `df_oracle.csv` | `Datasets/Bank_Marketing/300_200/` | Oracle test set |
| `oracle_loss_0_1.csv` | `Datasets/Loss_data/` | Per-sample 0-1 losses on oracle set |
| `small_loss_0_1.csv` | `Datasets/Loss_data/` | Per-sample 0-1 losses on small test set |
| `syn_loss_0_1.csv` | `Datasets/Loss_data/` | Per-sample 0-1 losses on synthetic data (updated per iteration) |
| `syn_dist_500.npy` | `Datasets/Distribution_data/` | Pg: Empirical distribution over 500 partitions |
| `opt_syn_total_50000_partion_500.npy` | `Datasets/Distribution_data/` | Optimal synthetic counts per partition (before adjustment) |
| `clustered_data_{delta2}_{b_value}_iter_{i}.csv` | `Datasets/Temp_cluster/` | Clustered synthetic data per iteration |
| `cluster_assignments_{delta2}_{b_value}_iter_{i}.txt` | `Datasets/Temp_cluster/` | Cluster assignments per iteration |

### GAN Model

| File | Location | Description |
|------|----------|-------------|
| `ctgan_bank.pt` | `GAN_models/` | Trained CTGAN model (torch saved object) |

<!-- ## Troubleshooting

### Common Issues

#### 1. FAISS Installation Errors

**Error:** `ImportError: cannot import name 'faiss'`

**Solution:**
```bash
# For CPU
pip uninstall faiss-cpu faiss-gpu
pip install faiss-cpu

# For GPU (requires CUDA)
pip uninstall faiss-cpu faiss-gpu
pip install faiss-gpu
```

#### 2. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# Option 1: Reduce batch sizes in config.py
_C.GAN.BATCH_SIZE = 250  # Reduce from 500
_C.CLUSTER.BATCH_SIZE = 5000  # Reduce from 10000

# Option 2: Use CPU
_C.DEVICE.NAME = "cpu"

# Option 3: Reduce synthetic points per iteration
_C.LOWER_BOUND.NUM_SYN_POINTS_PER_EPOCH = 25000  # Reduce from 50000
```

#### 3. UCI Dataset Download Fails

**Error:** `ConnectionError` or `TimeoutError`

**Solutions:**
```bash
# Option 1: Retry with longer timeout
# Option 2: Download manually and place in Datasets/

# Option 3: Use cached data if available
python main.py  # Will use existing data in Datasets/Bank_Marketing/300_200/
```

#### 4. Memory Error During Clustering

**Error:** `MemoryError` in `calculate_polynomial_distribution`

**Solutions:**
```python
# In config.py
_C.CLUSTER.NUM_POINTS_CAL_DISTRIBUTION = 25000  # Reduce from 50000
_C.CLUSTER.NUM_ITERS_CAL_DISTRIBUTION = 20  # Reduce from 30
_C.CLUSTER.BATCH_SIZE = 5000  # Reduce from 10000
```

#### 5. Precomputed Files Mismatch

**Error:** Shapes don't match or unexpected errors loading `.npy` files

**Solution:**
```bash
# Delete precomputed files to force recalculation
rm Datasets/Distribution_data/*.npy
rm Datasets/Loss_data/*.csv
python main.py
```

#### 6. Slow Execution

**Issue:** Training/clustering takes too long

**Solutions:**
```python
# 1. Use GPU acceleration
_C.DEVICE.NAME = "cuda"

# 2. Reduce iterations
_C.LOWER_BOUND.NUM_ITERATIONS = 5  # Reduce from 15

# 3. Reduce GAN training time
_C.GAN.EPOCHS = 1  # Use pre-trained model after first run

# 4. Reduce distribution computation
_C.CLUSTER.NUM_ITERS_CAL_DISTRIBUTION = 10  # Reduce from 30

# 5. Enable precomputed distributions (run once, reuse)
# After first run, distributions are cached automatically
```

#### 7. Classifier Convergence Warnings

**Warning:** `ConvergenceWarning: lbfgs failed to converge`

**Solutions:**
```python
# In classifier_config.py
_C.CLS.LR.MAX_ITER = 2000  # Increase from 1000
_C.CLS.MLP.MAX_ITER = 2000  # Increase from 1000
_C.CLS.TOL = 1e-3  # Increase tolerance
```

#### 8. Index Out of Bounds Errors

**Error:** Index errors in `calculate_diff_loss`

**Cause:** Mismatch between cluster assignments and dataframe indices

**Solution:**
```bash
# Clear temporary files
rm -rf Datasets/Temp_cluster/*
rm -rf Datasets/Optim_data/*
python main.py
```

### Performance Optimization Tips

1. **Use pre-trained GAN:** After first run, GAN is cached in `GAN_models/ctgan_bank.pt`
2. **Cache distributions:** Pg and optimal counts are cached in `Datasets/Distribution_data/`
3. **Use GPU:** Set `_C.DEVICE.NAME = "cuda"` for 5-10x speedup on large batches
4. **Reduce classifiers:** By default, only Random Forest is enabled. Adding more classifiers increases runtime proportionally.
5. **Batch processing:** The code already uses batch processing for loss computation -->

### Getting Help

If you encounter issues not covered here:

1. Check the paper for theoretical details
2. Review error messages and stack traces
3. Verify configuration parameters match your dataset
4. Check file paths and permissions
5. Open an issue on GitHub with:
   - Full error message
   - Configuration used (from `config.py`)
   - Dataset information
   - Steps to reproduce

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{osyn2025,
  title={Using Synthetic Data to estimate the True Error is theoretically and practically doable},
  author={Hai, Hoang Thanh and Duy-Tung, Nguyen and Hung, The Tran and Khoat, Than},
  journal={Machine Learning},
  year={2025},
  publisher={Springer}
}
```

<!-- ## License

[Add your license here]

## Acknowledgments

- UCI Machine Learning Repository for the Bank Marketing dataset
- CTGAN authors for the synthetic data generation framework
- FAISS team for efficient similarity search and clustering
- Scikit-learn contributors for machine learning utilities -->

## Contact

For questions and support:
- Email: tungn1201@gmail.com
- Open an issue on GitHub
