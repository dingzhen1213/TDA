import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import config_UCI


def set_seed(seed=config_UCI.SEED):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_uci_raw_inertial_data(data_path=config_UCI.UCI_DATA_PATH):
    """
    Load raw inertial data from UCI-HAR dataset

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Loading UCI-HAR raw inertial data...")

    # Sensor types and axes
    sensors = ['body_acc', 'body_gyro', 'total_acc']
    axes = ['x', 'y', 'z']

    def load_inertial_file(dataset_type='train', sensor='body_acc', axis='x'):
        """Load a single inertial data file"""
        path = os.path.join(data_path, dataset_type, 'Inertial Signals',
                            f'{sensor}_{axis}_{dataset_type}.txt')
        return np.loadtxt(path)

    # Collect all data
    X_data = {}
    y_data = {}

    for dataset_type in ['train', 'test']:
        # Load labels
        y_path = os.path.join(data_path, dataset_type, f'y_{dataset_type}.txt')
        y = np.loadtxt(y_path).astype(int) - 1  # Convert to 0-5

        # Load all sensor data
        sensor_data = []
        for sensor in sensors:
            for axis in axes:
                data = load_inertial_file(dataset_type, sensor, axis)  # (samples, 128)
                sensor_data.append(data)

        # Combine all sensor channels (samples, 128, 9)
        X = np.stack(sensor_data, axis=-1)
        X_data[dataset_type] = X
        y_data[dataset_type] = y

        print(f"{dataset_type} set: {X.shape}, labels: {y.shape}")

    return X_data['train'], X_data['test'], y_data['train'], y_data['test']


def preprocess_raw_data(X_train, X_test):
    """Preprocess raw inertial data"""
    # Reshape for standardization
    n_train, T, C = X_train.shape
    n_test = X_test.shape[0]

    X_train_2d = X_train.reshape(-1, C)
    X_test_2d = X_test.reshape(-1, C)

    # Standardization
    scaler = StandardScaler()
    X_train_2d_scaled = scaler.fit_transform(X_train_2d)
    X_test_2d_scaled = scaler.transform(X_test_2d)

    # Reshape back
    X_train = X_train_2d_scaled.reshape(n_train, T, C)
    X_test = X_test_2d_scaled.reshape(n_test, T, C)

    return X_train, X_test


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=config_UCI.BATCH_SIZE, val_size=0.2, seed=config_UCI.SEED):
    """
    Create train, validation, and test data loaders
    """
    # Split training data into train and validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=seed,
        stratify=y_train
    )

    def to_tensor_dataset(X, y):
        X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, C, T)
        y_t = torch.tensor(y, dtype=torch.long)
        return TensorDataset(X_t, y_t)

    # Create datasets
    train_dataset = to_tensor_dataset(X_tr, y_tr)
    val_dataset = to_tensor_dataset(X_val, y_val)
    test_dataset = to_tensor_dataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data loaders created:")
    print(f"  Training: {len(train_loader.dataset)} samples")
    print(f"  Validation: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")

    return train_loader, val_loader, test_loader