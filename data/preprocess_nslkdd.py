"""
preprocess_nslkdd.py — Download and preprocess the NSL-KDD dataset.

Produces:
    data/processed/nsl-kdd/train.csv   (122 features + label)
    data/processed/nsl-kdd/test.csv

Usage:
    python data/preprocess_nslkdd.py \
        --input  data/raw/nsl-kdd/ \
        --output data/processed/nsl-kdd/

Download the raw files manually from:
    https://www.unb.ca/cic/datasets/nsl.html
    Files needed: KDDTrain+.txt  KDDTest+.txt
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ── Column names (41 features + label + difficulty) ──────────────────────────
COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

CATEGORICAL = ["protocol_type", "service", "flag"]

ATTACK_MAP = {
    "normal": 0,
    # DoS
    "back":1,"land":1,"neptune":1,"pod":1,"smurf":1,"teardrop":1,
    "apache2":1,"udpstorm":1,"processtable":1,"worm":1,
    # Probe
    "ipsweep":2,"nmap":2,"portsweep":2,"satan":2,"mscan":2,"saint":2,
    # R2L
    "ftp_write":3,"guess_passwd":3,"imap":3,"multihop":3,"phf":3,
    "spy":3,"warezclient":3,"warezmaster":3,"sendmail":3,"named":3,
    "snmpgetattack":3,"snmpguess":3,"xlock":3,"xsnoop":3,"httptunnel":3,
    # U2R
    "buffer_overflow":4,"loadmodule":4,"perl":4,"rootkit":4,
    "mailbomb":4,"ps":4,"sqlattack":4,"xterm":4,
}


def preprocess(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for split, fname in [("train", "KDDTrain+.txt"), ("test", "KDDTest+.txt")]:
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}.\n"
                f"Download from https://www.unb.ca/cic/datasets/nsl.html"
            )

        df = pd.read_csv(path, names=COLUMNS)
        df = df.drop(columns=["difficulty"])

        # Map attack labels to 5-class integer
        df["label"] = df["label"].str.strip(".").str.lower().map(ATTACK_MAP).fillna(0).astype(int)

        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=CATEGORICAL)

        # Separate features and labels
        y = df["label"].values
        X = df.drop(columns=["label"]).values.astype(np.float32)

        # Normalize continuous features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Save
        out = pd.DataFrame(X)
        out["label"] = y
        out_path = os.path.join(output_dir, f"{split}.csv")
        out.to_csv(out_path, index=False)
        print(f"Saved {split} set: {X.shape[0]} samples, {X.shape[1]} features → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/nsl-kdd/")
    parser.add_argument("--output", default="data/processed/nsl-kdd/")
    args = parser.parse_args()
    preprocess(args.input, args.output)
