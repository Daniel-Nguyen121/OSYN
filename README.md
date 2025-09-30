# OSYN
This repo is of ACML 2025 paper "**Using Synthetic Data to estimate the True Error is theoretically and practically doable**" (conditionally accepted at Journal track).

**OSYN is a theoretically grounded framework** that selects (and optimizes) synthetic test data from a generator to evaluate a trained model, yielding **reliable estimates** of its **true test error** when real test sets are scarce.
<!-- 
## Key Features

- **CTGAN Integration**: Uses Conditional Tabular GAN for generating realistic synthetic data
- **Clustering-based Optimization**: Implements FAISS-based clustering for optimal data distribution
- **Lower Bounds**: Provides mathematical guarantees for synthetic data quality
- **Multiple Classifier Support**: Compatible with various ML classifiers (Random Forest, SVM, Decision Tree, etc.)
- **Comprehensive Evaluation**: Includes bootstrap, theoretical, and empirical evaluation methods
- **Configurable Parameters**: Extensive configuration system for fine-tuning experiments -->

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
├── calculate_lower_bound.py  # Estimating True Test Error of the model
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

## Requirements
Python3 support only. Tested on CUDA9.0, cudnn7.

* torch==1.9.0
* torchvision
* ctgan
* scikit-learn==1.0.0
* pandas
* numpy
* faiss-cpu
* ucimlrepo
* yacs
* tqdm
* scipy
* joblib

### Configuration
| configure                       | description                                                               |
|---------------------------------|---------------------------------------------------------------------------|
| DATASET_ID                      | UCI Repository dataset ID, eg: 222                                       |
<!-- | ORACLE_RATIO                    | Ratio of oracle points, eg: 0.7                                          | -->
| TOTAL_PARTITONS                 | Total partitions for clustering, eg: 500                                  |
<!-- | EPOCHS                          | Training epochs, eg: 3                                                    |
| BATCH_SIZE                      | Batch size for training, eg: 500                                          | -->
<!-- | NUM_CENTROIDS                   | Number of cluster centroids, eg: 500                                      | -->
| ADJUST_FACTOR                   | Adjustment factor for distribution, eg: 0.7                               |
| DELTA_1                         | Delta 1 parameter, eg: 0.01                                               |
| DELTA_2                         | Delta 2 parameter, eg: 0.0005                                             |
| NUM_ITERATIONS                  | Number of iterations, eg: 15                                              |

*Note: You can see more configuration parameters in config.py and classifier_config.py*

### Training
1. Run the original version:
```bash
python main.py
```

2. Run the enhanced version with parameter sweeps:
```bash
python main_b.py
```


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{osyn2025,
  title={Using Synthetic Data to estimate the True Error is theoretically and practically doable},
  author={Hai, Hoang Thanh and Duy-Tung, Nguyen and Hung, The Tran and Khoat, Than},
  journal={Machine Learning},
  year={2025},
  publisher={Springer},
  note={Conditionally Accepted at ACML 2025}
}
```