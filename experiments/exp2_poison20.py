"""
exp2_poison20.py
Experiment 2: FL under 20% gradient poisoning attack.
Compares QI-FedDetect vs FedAvg vs FedProx.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import LSTMAnomalyDetector, train_one_epoch, evaluate
from src.aggregation import QIFedDetect, FedAvgBaseline, FedProxBaseline
from src.attack import inject_poisoned_clients

N_CLIENTS = 10
POISON_FRACTION = 0.2
N_ROUNDS = 30
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LR = 1e-3
N_RUNS = 5
ATTACK_TYPE = "sign_flip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALGORITHMS = {
    "QI-FedDetect": QIFedDetect(threshold=0.5),
    "FedAvg": FedAvgBaseline(),
    "FedProx": FedProxBaseline(mu=0.01),
}


def load_client_data(client_id):
    X = np.load(os.path.join(DATA_DIR, f"client_{client_id}_X.npy"))
    y = np.load(os.path.join(DATA_DIR, f"client_{client_id}_y.npy"))
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)


def load_test_data():
    X = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=False)


def run_experiment(algorithm_name, aggregator, run_id):
    print(f"\n  [{algorithm_name}] Run {run_id + 1}/{N_RUNS}")
    global_model = LSTMAnomalyDetector(input_dim=41).to(DEVICE)
    test_loader = load_test_data()
    client_loaders = [load_client_data(i) for i in range(N_CLIENTS)]
    round_metrics = []
    detection_rates = []

    for rnd in range(N_ROUNDS):
        client_updates = []
        for i in range(N_CLIENTS):
            local_model = LSTMAnomalyDetector(input_dim=41).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            opt = torch.optim.Adam(local_model.parameters(), lr=LR)
            for _ in range(LOCAL_EPOCHS):
                train_one_epoch(local_model, client_loaders[i], opt, DEVICE)
            n_samples = len(client_loaders[i].dataset)
            client_updates.append((local_model.state_dict(), n_samples))

        poisoned_updates, true_poisoned = inject_poisoned_clients(
            client_updates, poison_fraction=POISON_FRACTION, attack_type=ATTACK_TYPE
        )
        ref_state = global_model.state_dict()
        agg_state, round_info = aggregator.aggregate(poisoned_updates, ref_state)
        global_model.load_state_dict(agg_state)
        metrics = evaluate(global_model, test_loader, DEVICE)
        round_metrics.append(metrics)

        if "flagged_indices" in round_info and round_info["n_flagged"] > 0:
            flagged = set(round_info["flagged_indices"])
            true_set = set(true_poisoned)
            tp = len(flagged & true_set)
            detection_rate = tp / len(true_set) if true_set else 0.0
        else:
            detection_rate = 0.0
        detection_rates.append(detection_rate)

        if (rnd + 1) % 10 == 0:
            print(f"    Round {rnd+1:2d} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | Det: {detection_rate:.2f}")

    return round_metrics, detection_rates


def main():
    print("=" * 60)
    print("Experiment 2: 20% Gradient Poisoning Attack")
    print(f"Device: {DEVICE} | Runs: {N_RUNS} | Rounds: {N_ROUNDS}")
    print("=" * 60)
    all_results = {}

    for alg_name, aggregator in ALGORITHMS.items():
        run_accuracies, run_f1s, run_detection = [], [], []
        for run_id in range(N_RUNS):
            round_metrics, detection_rates = run_experiment(alg_name, aggregator, run_id)
            run_accuracies.append(round_metrics[-1]["accuracy"])
            run_f1s.append(round_metrics[-1]["f1"])
            run_detection.append(np.mean(detection_rates))

        all_results[alg_name] = {
            "accuracy_mean": round(np.mean(run_accuracies), 4),
            "accuracy_std": round(np.std(run_accuracies), 4),
            "f1_mean": round(np.mean(run_f1s), 4),
            "f1_std": round(np.std(run_f1s), 4),
            "detection_rate_mean": round(np.mean(run_detection), 4),
            "detection_rate_std": round(np.std(run_detection), 4),
        }

    print("\n" + "=" * 60)
    print("RESULTS (mean +/- std over 5 runs)")
    print("=" * 60)
    for alg, res in all_results.items():
        print(f"{alg:<18} Acc: {res['accuracy_mean']:.4f}+/-{res['accuracy_std']:.4f}  F1: {res['f1_mean']:.4f}+/-{res['f1_std']:.4f}  Det: {res['detection_rate_mean']:.4f}+/-{res['detection_rate_std']:.4f}")

    out_path = os.path.join(RESULTS_DIR, "exp2_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Experiment 2: 20% Gradient Poisoning Attack", fontsize=13)
    metrics_keys = ["accuracy_mean", "f1_mean", "detection_rate_mean"]
    err_keys = ["accuracy_std", "f1_std", "detection_rate_std"]
    titles = ["Accuracy", "F1 Score", "Detection Rate"]
    colors = ["#534AB7", "#1D9E75", "#BA7517"]
    algs = list(all_results.keys())
    for ax, mk, ek, title in zip(axes, metrics_keys, err_keys, titles):
        vals = [all_results[a][mk] for a in algs]
        errs = [all_results[a][ek] for a in algs]
        bars = ax.bar(algs, vals, yerr=errs, capsize=5, color=colors, edgecolor="white")
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, f"{val:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "exp2_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
