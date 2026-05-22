"""
download_datasets.py
Download and preprocess NSL-KDD dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import urllib.request

DATA_DIR = os.path.dirname(__file__)

NSL_KDD_TRAIN_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
NSL_KDD_TEST_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"

NSL_KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]


def download_nsl_kdd():
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")
    if not os.path.exists(train_path):
        print("Downloading NSL-KDD training set...")
        urllib.request.urlretrieve(NSL_KDD_TRAIN_URL, train_path)
    if not os.path.exists(test_path):
        print("Downloading NSL-KDD test set...")
        urllib.request.urlretrieve(NSL_KDD_TEST_URL, test_path)
    return train_path, test_path


def preprocess_nsl_kdd(train_path, test_path, n_clients=10):
    print("Preprocessing NSL-KDD...")
    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
    train_df.drop("difficulty", axis=1, inplace=True)
    test_df.drop("difficulty", axis=1, inplace=True)
    train_df["label"] = (train_df["label"] != "normal").astype(int)
    test_df["label"] = (test_df["label"] != "normal").astype(int)
    cat_cols = ["protocol_type", "service", "flag"]
    le = LabelEncoder()
    for col in cat_cols:
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int64)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int64)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    sort_idx = np.argsort(y_train)
    X_train_sorted = X_train[sort_idx]
    y_train_sorted = y_train[sort_idx]
    split_size = len(X_train_sorted) // n_clients
    for i in range(n_clients):
        start = i * split_size
        end = start + split_size if i < n_clients - 1 else len(X_train_sorted)
        np.save(os.path.join(DATA_DIR, f"client_{i}_X.npy"), X_train_sorted[start:end])
        np.save(os.path.join(DATA_DIR, f"client_{i}_y.npy"), y_train_sorted[start:end])
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
    print(f"Done. {n_clients} client splits saved to /data/")


if __name__ == "__main__":
    train_path, test_path = download_nsl_kdd()
    preprocess_nsl_kdd(train_path, test_path, n_clients=10)
