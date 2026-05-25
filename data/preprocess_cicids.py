"""
preprocess_cicids.py — Preprocess the CICIDS2017 dataset.

Produces:
    data/processed/cicids2017/train.csv
    data/processed/cicids2017/test.csv

Usage:
    python data/preprocess_cicids.py \
        --input  data/raw/cicids2017/ \
        --output data/processed/cicids2017/ \
        --sample 100000

Download raw CSV files from:
    https://www.unb.ca/cic/datasets/ids-2017.html
    (MachineLearningCSV.zip — extract all day CSVs into input dir)
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

LABEL_MAP = {
    "BENIGN": 0,
    "FTP-Patator": 1, "SSH-Patator": 1,                          # Brute Force
    "DoS slowloris": 2, "DoS Slowhttptest": 2,
    "DoS Hulk": 2, "DoS GoldenEye": 2, "Heartbleed": 2,          # DoS
    "Web Attack – Brute Force": 3, "Web Attack – XSS": 3,
    "Web Attack – Sql Injection": 3,                              # Web Attacks
    "Infiltration": 4,                                            # Infiltration
    "Bot": 5,                                                     # Botnet
    "DDoS": 6,                                                    # DDoS
}

DROP_COLS = [
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port", "Timestamp", "Label",
]

FILL_VALUE = 0.0


def preprocess(input_dir: str, output_dir: str, sample: int = 100_000,
               test_size: float = 0.2, seed: int = 42) -> None:
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {input_dir}.\n"
            f"Download from https://www.unb.ca/cic/datasets/ids-2017.html"
        )

    print(f"Found {len(csv_files)} CSV files. Loading ...")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8", low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df):,}")

    # Normalise label column name
    label_col = None
    for col in df.columns:
        if col.strip().lower() == "label":
            label_col = col
            break
    if label_col is None:
        raise ValueError("Could not find 'Label' column.")

    # Map labels
    df[label_col] = df[label_col].str.strip()
    df["label"] = df[label_col].map(LABEL_MAP).fillna(0).astype(int)

    # Drop non-feature columns
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop + [label_col])

    # Convert to numeric, fill inf/nan
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(FILL_VALUE)

    y = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32)

    # Sample if requested
    if sample and sample < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=sample, replace=False)
        X, y = X[idx], y[idx]
        print(f"Sampled {sample:,} rows.")

    # z-score normalisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.clip(X, -10, 10)    # clip extreme values

    # Train / test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    for split, Xs, ys in [("train", X_tr, y_tr), ("test", X_te, y_te)]:
        out = pd.DataFrame(Xs)
        out["label"] = ys
        path = os.path.join(output_dir, f"{split}.csv")
        out.to_csv(path, index=False)
        print(f"Saved {split}: {Xs.shape[0]} samples, {Xs.shape[1]} features → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/cicids2017/")
    parser.add_argument("--output", default="data/processed/cicids2017/")
    parser.add_argument("--sample", type=int, default=100_000)
    args = parser.parse_args()
    preprocess(args.input, args.output, args.sample)
