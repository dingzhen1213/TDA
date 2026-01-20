import torch

# Data paths
WISDM_DATA_PATH = ""
SAVE_PATH = "save/WISDM/checkpoints"

# Data parameters
WINDOW_SIZE = 200
OVERLAP = 0.5
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # Validation split from training data

# Training parameters
BATCH_SIZE = 64
EPOCHS = 50
LR = 5e-4
SEED = 42

# Model parameters
EMBED_DIM = 64
FEATURE_DIM = 32
NUM_HEADS = 4
USE_MULTI_SCALE = True
USE_FREQ_ENHANCE = True
DROPOUT = 0.1
KERNEL_SIZES = [13, 25, 37]

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Activity labels mapping
ACTIVITY_NAMES = {
    'Walking': 'Walking',
    'Jogging': 'Jogging',
    'Sitting': 'Sitting',
    'Standing': 'Standing',
    'Upstairs': 'Upstairs',
    'Downstairs': 'Downstairs'
}