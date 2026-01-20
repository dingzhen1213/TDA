"""
Configuration file for TDN-UCI-HAR
"""

import torch

# Data paths
UCI_DATA_PATH = ""
SAVE_PATH = "save/UCIHAR/checkpoints"
LOG_PATH = "logs"

# Data parameters
WINDOW_SIZE = 128  # UCI-HAR default window size
TEST_SIZE = 0.2    # Validation split from training data

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LR = 5e-4
SEED = 45

# Model parameters
EMBED_DIM = 64
FEATURE_DIM = 32
NUM_HEADS = 4
USE_MULTI_SCALE = True
USE_FREQ_ENHANCE = True
DROPOUT = 0.1
KERNEL_SIZES = [13, 25, 37]  # Multi-scale kernel sizes

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Activity labels for UCI-HAR
ACTIVITY_LABELS = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}

# UCI-HAR sensor channels (9 channels total)
SENSOR_CHANNELS = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z'
]