
import torch

# Data paths
DATA_PATH = ""
SAVE_PATH = "save/pamap2/checkpoints"
LOG_PATH = "logs"

# Data parameters
WINDOW_SIZE = 128
OVERLAP = 0.5
USE_18_CHANNELS = True
SPLIT_BY_USER = False
TEST_SIZE = 0.2
RANDOM_STATE = 40

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

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Valid activity IDs for PAMAP2 dataset
VALID_ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

# Column names for PAMAP2 dataset
COLUMN_NAMES = [
    'timestamp', 'activity_id', 'heart_rate',
    'hand_temp', 'hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z',
    'hand_acc6_x', 'hand_acc6_y', 'hand_acc6_z',
    'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
    'hand_magn_x', 'hand_magn_y', 'hand_magn_z',
    'hand_orient1', 'hand_orient2', 'hand_orient3', 'hand_orient4',
    'chest_temp', 'chest_acc16_x', 'chest_acc16_y', 'chest_acc16_z',
    'chest_acc6_x', 'chest_acc6_y', 'chest_acc6_z',
    'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
    'chest_magn_x', 'chest_magn_y', 'chest_magn_z',
    'chest_orient1', 'chest_orient2', 'chest_orient3', 'chest_orient4',
    'ankle_temp', 'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z',
    'ankle_acc6_x', 'ankle_acc6_y', 'ankle_acc6_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_magn_x', 'ankle_magn_y', 'ankle_magn_z',
    'ankle_orient1', 'ankle_orient2', 'ankle_orient3', 'ankle_orient4'
]

# Sensor columns for 18-channel configuration
SENSOR_COLUMNS_18CH = [
    'hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z',
    'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
    'chest_acc16_x', 'chest_acc16_y', 'chest_acc16_z',
    'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
    'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z'
]