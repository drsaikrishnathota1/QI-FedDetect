"""
exp4_communication.py
Experiment 4: Communication cost comparison.
Measures bits transmitted per round across all algorithms.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import LSTMAnomalyDetector, train_one_epoch
from src.aggregation import QIFedDetect, FedAvgBaseline, FedProxBaseline
from src.encoding import flatten_gradients

N_CLIENTS = 10
N_ROUNDS = 30
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LR = 1e-3
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


def compute_bits(state_dict):
    flat = flatten_gradients(state_dict)
    return flat.nbytes * 8


def main():
    print("=" * 60)
    print("Experiment 4: Communication Cost Per Round")
    print(f"Device: {DEVICE} | Rounds: {N_ROUNDS}")
    print("=" * 60)

    client_loaders = [load_client_data(i) for i in range(N_CLIENTS)]
    all_results = {}

    for alg_name, aggregator in ALGORITHMS.items():
        print(f"\n  [{alg_name}]")
        global_model = LSTMAnomalyDetector(input_dim=41).to(DEVICE)
        round_bits = []

        for rnd in range(N_ROUNDS):
            client_updates = []
            total_bits = 0

            for i in range(N_CLIENTS):
                local_model = LSTMAnomalyDetector(input_dim=41).to(DEVICE)
                local_model.load_state_dict(global_model.state_dict())
                opt = torch.optim.Adam(local_model.parameters(), lr=LR)
                for _ in range(LOCAL_EPOCHS):
                    train_one_epoch(local_model, client_loaders[i], opt, DEVICE)
                n_samples = len(client_loaders[i].dataset)
                client_updates.append((local_model.state_dict(), n_samples))
                total_bits += compute_bits(local_model.state_dict())

            ref_state = global_model.state_dict()
            agg_state, _ = aggregator.aggregate(client_updates, ref_state)
            global_model.load_state_dict(agg_state)
            total_bits += compute_bits(agg_state)
            round_bits.append(total_bits)

            if (rnd + 1) % 10 == 0:
                print(f"    Round {rnd+1:2d} | Bits: {total_bits:,}")

        all_results[alg_name] = {
            "mean_bits_per_round": int(np.mean(round_bits)),
            "total_bits": int(np.sum(round_bits)),
            "round_bits": round_bits,
        }

    print("\n" + "=" * 60)
    print("RESULTS: Communication Cost")
    print("=" * 60)
    for alg, res in all_results.items():
        mb = res["mean_bits_per_round"] / (1024 * 1024)
        print(f"{alg:<18} Mean per round: {mb:.2f} MB | Total: {res['total_bits']/(1024*1024*1024):.3f} GB")

    out_path = os.path.join(RESULTS_DIR, "exp4_results.json")
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != "round_bits"}
                    for k, v in all_results.items()}
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    plt.figure(figsize=(10, 5))
    colors = ["#534AB7", "#1D9E75", "#BA7517"]
    for (alg_name, res), color in zip(all_results.items(), colors):
        plt.plot(range(1, N_ROUNDS + 1), [b/(1024*1024) for b in res["round_bits"]],
                 label=alg_name, color=color, linewidth=2)
    plt.xlabel("Communication Round")
    plt.ylabel("Bits Transmitted (MB)")
    plt.title("Experiment 4: Communication Cost Per Round")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "exp4_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
