"""
run_all_experiments.py
Run all experiments in sequence.
Usage: python run_all_experiments.py
"""

import os
import sys
import subprocess

EXPERIMENTS = [
    ("Experiment 2: 20% poisoned clients", "experiments/exp2_poison20.py"),
]


def main():
    print("=" * 65)
    print("QI-FedDetect: Full Experiment Suite")
    print("=" * 65)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(os.path.join(data_dir, "X_test.npy")):
        print("\n[Setup] Downloading and preprocessing datasets...")
        subprocess.run([sys.executable, "data/download_datasets.py"], check=True)
    else:
        print("\n[Setup] Datasets already present.")

    for i, (label, script) in enumerate(EXPERIMENTS, 1):
        print(f"\n{'=' * 65}")
        print(f"Running {label}")
        print(f"{'=' * 65}")
        subprocess.run([sys.executable, script])

    print("\n" + "=" * 65)
    print("All experiments complete. Results saved to /results/")
    print("=" * 65)


if __name__ == "__main__":
    main()
