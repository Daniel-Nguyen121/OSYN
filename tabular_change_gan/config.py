from yacs.config import CfgNode as CN
import os

_C = CN()

# Device
_C.DEVICE = CN()
_C.DEVICE.NAME = "cuda"  # Use GPU 1
_C.DEVICE.NUM_GPUS = 1

### ------------------------------------- ###

# Data
_C.DATA = CN()
_C.DATA.DATASET_ID = 222            # ID of dataset in UCI repository
_C.DATA.ORACLE_RATIO = 0.7          # Ratio of oracle points
_C.DATA.TOTAL_PARTITONS = 500       # Total partitions of dataset
_C.DATA.NUMBER_DOMINATE = 300       # Number of points belong to the dominate class (default, 300 points)
_C.DATA.SAVE_DIR = 'Datasets'       # Save directory for dataset
_C.DATA.NUM_DEL_MISSING_COLS = 2    # Number of columns to delete if they have too many missing values
_C.DATA.SEED = 42                   # Seed for random number generator

### ------------------------------------- ###
# Cluster
_C.CLUSTER = CN()
_C.CLUSTER.NUM_CENTROIDS = 500
_C.CLUSTER.PERCENT_LIMIT = 10
_C.CLUSTER.NUM_ITERS_CAL_DISTRIBUTION = 30
_C.CLUSTER.NUM_POINTS_CAL_DISTRIBUTION = 50000
_C.CLUSTER.SAVE_DIR_DISTRIBUTION = 'Datasets/Distribution_data/syn_dist_500.npy'
_C.CLUSTER.OPT_NUM_POINTS = 50000
_C.CLUSTER.SAVE_DIR_OPT_NUM_POINTS = 'Datasets/Distribution_data/opt_syn_total_50000_partion_500.npy'
_C.CLUSTER.ADJUST_FACTOR = 1.1
_C.CLUSTER.SAVE_DIR_SYN_DATA = 'Datasets/Optim_data'
_C.CLUSTER.BATCH_SIZE = 10000
_C.CLUSTER.TEMP_DIR = 'Datasets/Temp_cluster'

### ------------------------------------- ###
# GAN
_C.GAN = CN()
_C.GAN.EPOCHS = 3
_C.GAN.BATCH_SIZE = 500
_C.GAN.GENERATOR_DIM = (256, 256)
_C.GAN.DISCRIMINATOR_DIM = (256, 256)
_C.GAN.GENERATOR_LR = 2e-4
_C.GAN.DISCRIMINATOR_LR = 2e-4
_C.GAN.DISCRIMINATOR_STEPS = 1
_C.GAN.PAC = 10
_C.GAN.LOG_FREQUENCY = True
_C.GAN.VERBOSE = True
_C.GAN.SAVE_DIR = 'GAN_models/ctgan_bank.pt'

### ------------------------------------- ###
# Losses
_C.LOSS = CN()
_C.LOSS.SAVE_DIR = 'Datasets/Loss_data'
_C.LOSS.SAVE_CLUSTER = 'Datasets/Loss_data/small_loss_0_1.csv'
_C.LOSS.SAVE_ORACLE = 'Datasets/Loss_data/oracle_loss_0_1.csv'

### ------------------------------------- ###
# Lower bound
_C.LOWER_BOUND = CN()
_C.LOWER_BOUND.NUM_SYN_POINTS_PER_EPOCH = 50000
_C.LOWER_BOUND.DELTA_1 = 0.01
_C.LOWER_BOUND.DELTA_2 = 0.0005              # Ori: 0.2
_C.LOWER_BOUND.NUM_ITERATIONS = 15
_C.LOWER_BOUND.NUM_WO_ITERATIONS = 10
_C.LOWER_BOUND.NUM_BOOTSTRAP_ITERATIONS = 10000
_C.LOWER_BOUND.SAVE_DIR = 'Datasets/final_result.csv'

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    return _C.clone() 