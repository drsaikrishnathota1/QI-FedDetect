"""
exp5_convergence.py
Experiment 5: Convergence speed under attack.
Measures rounds needed to reach 95% accuracy under poisoning.
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
TARGET_ACCURACY = 0.90
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
    accuracy_curve = []
    rounds_to_target = None

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

        poisoned_updates, _ = inject_poisoned_clients(
            client_updates, poison_fraction=POISON_FRACTION, attack_type=ATTACK_TYPE
        )
        ref_state = global_model.state_dict()
        agg_state, _ = aggregator.aggregate(poisoned_updates, ref_state)
        global_model.load_state_dict(agg_state)
        metrics = evaluate(global_model, test_loader, DEVICE)
        accuracy_curve.append(metrics["accuracy"])

        if rounds_to_target is None and metrics["accuracy"] >= TARGET_ACCURACY:
            rounds_to_target = rnd + 1
            print(f"    Reached {TARGET_ACCURACY} accuracy at round {rounds_to_target}")

        if (rnd + 1) % 10 == 0:
            print(f"    Round {rnd+1:2d} | Acc: {metrics['accuracy']:.4f}")

    if rounds_to_target is None:
        rounds_to_target = N_ROUNDS + 1
        print(f"    Did not reach {TARGET_ACCURACY} accuracy within {N_ROUNDS} rounds")

    return accuracy_curve, rounds_to_target


def main():
    print("=" * 60)
    print(f"Experiment 5: Convergence Speed (target={TARGET_ACCURACY})")
    print(f"Device: {DEVICE} | Runs: {N_RUNS} | Rounds: {N_ROUNDS}")
    print("=" * 60)
    all_results = {}
    all_curves = {}

    for alg_name, aggregator in ALGORITHMS.items():
        run_rounds = []
        run_curves = []
        for run_id in range(N_RUNS):
            curve, rounds = run_experiment(alg_name, aggregator, run_id)
            run_rounds.append(rounds)
            run_curves.append(curve)

        all_results[alg_name] = {
            "rounds_to_target_mean": round(np.mean(run_rounds), 2),
            "rounds_to_target_std": round(np.std(run_rounds), 2),
            "final_accuracy_mean": round(np.mean([c[-1] for c in run_curves]), 4),
        }
        all_curves[alg_name] = np.mean(run_curves, axis=0).tolist()

    print("\n" + "=" * 60)
    print(f"RESULTS: Rounds to reach {TARGET_ACCURACY} accuracy")
    print("=" * 60)
    for alg, res in all_results.items():
        print(f"{alg:<18} Rounds: {res['rounds_to_target_mean']:.1f}+/-{res['rounds_to_target_std']:.1f}  Final Acc: {res['final_accuracy_mean']:.4f}")

    out_path = os.path.join(RESULTS_DIR, "exp5_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    plt.figure(figsize=(10, 6))
    colors = ["#534AB7", "#1D9E75", "#BA7517"]
    for (alg_name, curve), color in zip(all_curves.items(), colors):
        plt.plot(range(1, N_ROUNDS + 1), curve, label=alg_name, color=color, linewidth=2)
    plt.axhline(y=TARGET_ACCURACY, color="red", linestyle="--", linewidth=1, label=f"Target ({TARGET_ACCURACY})")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.title("Experiment 5: Convergence Speed Under 20% Poisoning Attack")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "exp5_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
