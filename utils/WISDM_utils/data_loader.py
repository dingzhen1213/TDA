import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from config import config_WISDM


def set_seed(seed=config_WISDM.SEED):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_wisdm_data_simple(filepath, window_size=config_WISDM.WINDOW_SIZE, overlap=config_WISDM.OVERLAP):
    """Load WISDM data from raw text file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().replace(';', '')
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                user, activity, timestamp = parts[0].strip(), parts[1].strip(), parts[2].strip()
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                data.append([user, activity, timestamp, x, y, z])
            except:
                continue

    df = pd.DataFrame(data, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
    df = df.dropna()

    le = LabelEncoder()
    df['activity_encoded'] = le.fit_transform(df['activity'])

    # Sliding window segmentation
    windows, labels = [], []
    for activity in df['activity_encoded'].unique():
        act_data = df[df['activity_encoded'] == activity][['x', 'y', 'z']].values
        if len(act_data) < window_size:
            continue
        step = int(window_size * (1 - overlap))
        for i in range(0, len(act_data) - window_size + 1, step):
            windows.append(act_data[i:i + window_size])
            labels.append(activity)

    return np.array(windows), np.array(labels), le


def load_wisdm_dataset(data_path=config_WISDM.WISDM_DATA_PATH,
                       test_size=config_WISDM.TEST_SIZE, random_state=config_WISDM.SEED,
                       split_by_user=True, window_size=config_WISDM.WINDOW_SIZE, overlap=config_WISDM.OVERLAP):
    """Load and preprocess WISDM dataset"""
    X, y, le = load_wisdm_data_simple(data_path, window_size, overlap)

    if split_by_user:
        # Split by user
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip().replace(';', '')
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                try:
                    user, activity, timestamp = parts[0], parts[1], parts[2]
                    x, y_, z = float(parts[3]), float(parts[4]), float(parts[5])
                    data.append([user, activity, timestamp, x, y_, z])
                except:
                    continue

        df = pd.DataFrame(data, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
        df['activity_encoded'] = le.transform(df['activity'])
        users = df['user'].unique()
        train_users, test_users = train_test_split(users, test_size=test_size, random_state=random_state)

        def make_windows(sub_df):
            windows, labels = [], []
            for activity in sub_df['activity_encoded'].unique():
                act_data = sub_df[sub_df['activity_encoded'] == activity][['x', 'y', 'z']].values
                if len(act_data) < window_size:
                    continue
                step = int(window_size * (1 - overlap))
                for i in range(0, len(act_data) - window_size + 1, step):
                    windows.append(act_data[i:i + window_size])
                    labels.append(activity)
            return np.array(windows), np.array(labels)

        train_df = df[df['user'].isin(train_users)]
        test_df = df[df['user'].isin(test_users)]
        X_train, y_train = make_windows(train_df)
        X_test, y_test = make_windows(test_df)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    # Standardization
    n_train, T, C = X_train.shape
    n_test = X_test.shape[0]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, C)).reshape(n_train, T, C)
    X_test = scaler.transform(X_test.reshape(-1, C)).reshape(n_test, T, C)

    return X_train, y_train, X_test, y_test, le