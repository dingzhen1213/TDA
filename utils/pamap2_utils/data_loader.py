import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from config import config_pamap2
from config.config_pamap2 import DATA_PATH


def set_seed(seed=config_pamap2.SEED):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pamap2_dataset(data_path=DATA_PATH,
                        window_size=config_pamap2.WINDOW_SIZE,
                        overlap=config_pamap2.OVERLAP,
                        use_18_channels=config_pamap2.USE_18_CHANNELS,
                        split_by_user=config_pamap2.SPLIT_BY_USER,
                        test_size=config_pamap2.TEST_SIZE,
                        random_state=config_pamap2.RANDOM_STATE):
    """
    Load and preprocess PAMAP2 dataset

    Returns:
        X_train, y_train, X_test, y_test, label_encoder, users_train, users_test
    """
    print("Loading PAMAP2 dataset...")

    # Set seed for reproducibility
    set_seed(random_state)

    # Find data files
    files = sorted(glob.glob(os.path.join(data_path, "subject*.dat")))
    if len(files) == 0:
        raise FileNotFoundError(f"No subject*.dat files found in {data_path}")

    windows = []
    labels = []
    users = []

    # Calculate step size
    step = int(window_size * (1 - overlap))
    if step < 1:
        step = 1

    print(f"Window configuration: size={window_size}, step={step}, overlap={overlap}")

    # Process each subject file
    for file_path in files:
        fname = os.path.basename(file_path)
        sid = int(fname.replace("subject", "").replace(".dat", ""))

        try:
            df = pd.read_csv(file_path, sep='\s+', header=None,
                             names=config_pamap2.COLUMN_NAMES, na_values='NaN')
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

        # Handle missing values
        df_interpolated = df.interpolate(method='linear', axis=0)
        df_filled = df_interpolated.bfill()

        # Select sensor columns
        if use_18_channels:
            sensor_columns = config_pamap2.SENSOR_COLUMNS_18CH
        else:
            sensor_columns = [col for col in df_filled.columns
                              if col not in ['timestamp', 'activity_id', 'heart_rate']]

        activity = df_filled['activity_id'].values.astype(int)
        sensors = df_filled[sensor_columns].values

        print(f"File {fname}: {len(sensors)} samples (100Hz)")

        # Create sliding windows
        for i in range(0, sensors.shape[0] - window_size + 1, step):
            win = sensors[i:i + window_size]

            # Skip windows with NaN values
            if np.any(np.isnan(win)):
                continue

            act_win = activity[i:i + window_size]
            # Use mode of window as label
            lab = int(Counter(act_win).most_common(1)[0][0])
            windows.append(win)
            labels.append(lab)
            users.append(sid)

    if len(windows) == 0:
        raise ValueError("No valid windows generated. Check data files and processing.")

    X = np.array(windows)
    y_raw = np.array(labels)
    users = np.array(users)

    print(f"Generated windows: {X.shape[0]}, time steps: {X.shape[1]}, channels: {X.shape[2]}")

    # Filter valid activity IDs
    mask_valid = np.isin(y_raw, config_pamap2.VALID_ACTIVITY_IDS)
    X = X[mask_valid]
    y_raw = y_raw[mask_valid]
    users = users[mask_valid]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("Activity ID mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    # Split data
    if split_by_user:
        unique_users = np.unique(users)
        train_users, test_users = train_test_split(unique_users, test_size=test_size,
                                                   random_state=random_state)
        train_mask = np.isin(users, train_users)
        test_mask = np.isin(users, test_users)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        users_train = users[train_mask]
        users_test = users[test_mask]
        print(f"User split: train users {len(np.unique(users_train))}, "
              f"test users {len(np.unique(users_test))}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        users_train = users_test = None
        print("Random window split")

    # Standardize data
    n_train, T, C = X_train.shape
    n_test = X_test.shape[0]

    X_train_2d = X_train.reshape(-1, C)
    X_test_2d = X_test.reshape(-1, C)

    scaler = StandardScaler()
    X_train_2d_scaled = scaler.fit_transform(X_train_2d)
    X_test_2d_scaled = scaler.transform(X_test_2d)

    X_train = X_train_2d_scaled.reshape(n_train, T, C)
    X_test = X_test_2d_scaled.reshape(n_test, T, C)

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    return X_train, y_train, X_test, y_test, le, users_train, users_test


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=config_pamap2.BATCH_SIZE):
    """Create PyTorch DataLoaders from numpy arrays"""

    def to_tensor_dataset(X, y):
        X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, C, T)
        y_t = torch.tensor(y, dtype=torch.long)
        return TensorDataset(X_t, y_t)

    train_dataset = to_tensor_dataset(X_train, y_train)
    test_dataset = to_tensor_dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader