"""
run_all_experiments.py — Reproduce all results from the QI-FedDetect paper.

Produces the main manuscript tables and saves raw per-run metrics to
results/nslkdd_results.csv and results/cicids2017_results.csv.

Usage:
    # Full reproduction (all datasets, all methods, 5 seeds each)
    python run_all_experiments.py

    # Single dataset
    python run_all_experiments.py --dataset nslkdd

    # Single method
    python run_all_experiments.py --dataset cicids2017 --method qi_feddetect

    # Quick smoke-test (1 seed, 5 rounds)
    python run_all_experiments.py --dataset nslkdd --method fedavg --quick

Expected runtime:
    Full run on CPU:  ~6-10 hours
    Full run on GPU:  ~1-2 hours
    Quick test:       ~5 minutes
"""

import argparse
import os
import time
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from src.utils import set_seed, dirichlet_partition, save_results_csv, get_logger
from src.federated import run_federated

logger = get_logger("run_all_experiments")

METHODS  = ["qi_feddetect", "fltrust", "trimmed_mean", "krum", "fedprox", "fedavg"]
DATASETS = ["nslkdd", "cicids2017"]
CONFIG_PATHS = {
    "nslkdd":    "experiments/config_nslkdd.yaml",
    "cicids2017":"experiments/config_cicids2017.yaml",
}
RESULTS_PATHS = {
    "nslkdd":    "results/nslkdd_results.csv",
    "cicids2017":"results/cicids2017_results.csv",
}


def load_data(data_path: str):
    """Load preprocessed train/test CSVs into numpy arrays."""
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    test  = pd.read_csv(os.path.join(data_path, "test.csv"))
    y_tr  = train["label"].values
    X_tr  = train.drop(columns=["label"]).values.astype(np.float32)
    y_te  = test["label"].values
    X_te  = test.drop(columns=["label"]).values.astype(np.float32)
    return X_tr, y_tr, X_te, y_te


def print_summary(csv_path: str, dataset: str) -> None:
    """Print mean ± std table for all methods on a dataset."""
    df = pd.read_csv(csv_path)
    print(f"\n{'='*65}")
    print(f"  Results — {dataset.upper()}")
    print(f"{'='*65}")
    print(f"{'Method':<22} {'Accuracy':>12} {'F1-Score':>12} {'Byz Det Rate':>14}")
    print(f"{'-'*65}")
    for method in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        acc = f"{sub['accuracy'].mean():.4f} ± {sub['accuracy'].std():.4f}"
        f1  = f"{sub['f1_macro'].mean():.4f} ± {sub['f1_macro'].std():.4f}"
        bdr = f"{sub['byzantine_detection_rate'].mean()*100:.2f}% ± {sub['byzantine_detection_rate'].std()*100:.2f}%"
        print(f"{method:<22} {acc:>12} {f1:>12} {bdr:>14}")
    print(f"{'='*65}\n")


def run_dataset(dataset: str, methods: List[str], config: Dict[str, Any],
                quick: bool = False) -> None:
    logger.info(f"\n{'#'*60}\n  Dataset: {dataset.upper()}\n{'#'*60}")

    X_tr, y_tr, X_te, y_te = load_data(config["data_path"])
    logger.info(f"Loaded data: train={X_tr.shape}, test={X_te.shape}")

    n_clients  = config["n_clients"]
    n_byz      = config["n_byzantine"]
    byzantine_ids = list(range(n_clients - n_byz, n_clients))   # last n_byz clients

    seeds = [42] if quick else config["seeds"]
    if quick:
        config = dict(config)
        config["n_rounds"] = 5

    results_path = RESULTS_PATHS[dataset]
    os.makedirs("results", exist_ok=True)

    # Clear old results file for this run
    if os.path.exists(results_path):
        os.remove(results_path)

    for method in methods:
        logger.info(f"\n--- Method: {method.upper()} ---")
        for seed in seeds:
            set_seed(seed)
            client_indices = dirichlet_partition(
                y_tr, n_clients=n_clients,
                alpha=config["dirichlet_alpha"], seed=seed
            )
            result = run_federated(
                method=method,
                X_train=X_tr, y_train=y_tr,
                X_test=X_te,  y_test=y_te,
                client_indices=client_indices,
                byzantine_ids=byzantine_ids,
                config=config,
                seed=seed,
            )
            # Save this run immediately
            row = {
                "method":                   method,
                "seed":                     seed,
                "accuracy":                 round(result["accuracy"], 4),
                "f1_macro":                 round(result["f1_macro"], 4),
                "byzantine_detection_rate": round(result["byzantine_detection_rate"], 4),
                "total_time_seconds":       round(result["total_time_seconds"], 1),
            }
            save_results_csv([row], results_path)
            logger.info(
                f"  Saved: method={method} seed={seed} "
                f"acc={row['accuracy']:.4f} bdr={row['byzantine_detection_rate']:.3f}"
            )

    print_summary(results_path, dataset)


def main():
    parser = argparse.ArgumentParser(description="Reproduce QI-FedDetect results")
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--method",  choices=METHODS  + ["all"], default="all")
    parser.add_argument("--quick",   action="store_true",
                        help="Smoke-test: 1 seed, 5 rounds only")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    methods  = METHODS  if args.method  == "all" else [args.method]

    for dataset in datasets:
        with open(CONFIG_PATHS[dataset]) as f:
            config = yaml.safe_load(f)
        run_dataset(dataset, methods, config, quick=args.quick)

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
