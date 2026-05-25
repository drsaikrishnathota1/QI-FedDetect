# QI-FedDetect

**Quantum-Inspired Federated Intrusion Detection with Byzantine Fault Tolerance in Heterogeneous IoT Networks**

[![IEEE Access](https://img.shields.io/badge/Submitted-IEEE%20Access-blue)](https://ieeeaccess.ieee.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--5246--9421-green)](https://orcid.org/0009-0008-5246-9421)

> **Author:** Dr. Sai Krishna Thota, Independent Researcher, USA
> **Contact:** drsaikrishnathota@ieee.org
> **ORCID:** [0009-0008-5246-9421](https://orcid.org/0009-0008-5246-9421)

---

## Overview

QI-FedDetect is a federated intrusion detection framework that combines **variational quantum circuit (VQC)-inspired feature encoding** with a **Byzantine Detection Module (BDM)** based on Quantum Jensen-Shannon Divergence (QJSD). It enables privacy-preserving distributed anomaly detection across heterogeneous IoT clients while explicitly identifying and excluding malicious (Byzantine) participants from gradient aggregation.

### Key Results

| Dataset    | Method           | Accuracy         | F1-Score         | Byzantine Detection |
|------------|------------------|------------------|------------------|---------------------|
| NSL-KDD    | QI-FedDetect     | 0.7614 ± 0.0245  | 0.7390 ± 0.0340  | **60.83% ± 4.25%**  |
| NSL-KDD    | FedAvg           | 0.7433 ± 0.0099  | 0.7136 ± 0.0142  | 0% (none detected)  |
| NSL-KDD    | FedProx          | 0.7348 ± 0.0129  | 0.7008 ± 0.0194  | 0% (none detected)  |
| CICIDS2017 | QI-FedDetect     | 0.9997 ± 0.0000  | 0.9997 ± 0.0000  | **100% ± 0%**       |
| CICIDS2017 | FedAvg           | 0.9998 ± 0.0000  | 0.9998 ± 0.0000  | 0% (none detected)  |
| CICIDS2017 | FedProx          | 0.9998 ± 0.0000  | 0.9997 ± 0.0000  | 0% (none detected)  |

*All results are mean ± standard deviation across 5 independent runs.*

---

## Repository Structure

```
QI-FedDetect/
├── data/                        # Dataset preprocessing scripts
│   ├── preprocess_nslkdd.py
│   └── preprocess_cicids.py
├── src/                         # Core framework source code
│   ├── qife.py                  # Quantum-Inspired Feature Encoder
│   ├── federated.py             # Federated training loop
│   ├── bdm.py                   # Byzantine Detection Module
│   ├── models.py                # Local classifier architecture
│   └── utils.py                 # Helpers, metrics, logging
├── experiments/                 # Experiment configuration files
│   ├── config_nslkdd.yaml
│   └── config_cicids2017.yaml
├── results/                     # Output figures and logs
│   └── fig1_results.png         # Main results figure (Table I & II visualization)
├── run_all_experiments.py       # Single script to reproduce all paper results
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU recommended (CPU also supported, slower)

### Setup

```bash
# Clone the repository
git clone https://github.com/drsaikrishnathota1/QI-FedDetect.git
cd QI-FedDetect

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

QI-FedDetect uses two publicly available datasets. Neither dataset is included in this repository due to size; download instructions are below.

### NSL-KDD

1. Download from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html)
2. Download `NSL-KDD.zip` and extract to `data/raw/nsl-kdd/`
3. Preprocess:

```bash
python data/preprocess_nslkdd.py --input data/raw/nsl-kdd/ --output data/processed/nsl-kdd/
```

Expected output: `data/processed/nsl-kdd/train.csv` and `test.csv` (122 features after encoding).

### CICIDS2017

1. Download from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Extract all CSV files to `data/raw/cicids2017/`
3. Preprocess:

```bash
python data/preprocess_cicids.py --input data/raw/cicids2017/ --output data/processed/cicids2017/ --sample 100000
```

Expected output: `data/processed/cicids2017/train.csv` and `test.csv` (78 features, 100k samples).

---

## Reproducing Paper Results

All results reported in the paper (Tables I and II) can be reproduced with a single command:

```bash
python run_all_experiments.py --datasets nslkdd cicids2017 --runs 5 --output results/
```

This will:
1. Run QI-FedDetect, FedAvg, and FedProx on both datasets
2. Repeat each configuration for 5 independent runs
3. Save per-run logs to `results/logs/`
4. Generate `results/fig1_results.png` (the main results figure)
5. Print a summary table matching Tables I and II in the paper

**Expected runtime:** approximately 2–4 hours on CPU; 30–60 minutes with a CUDA GPU.

### Running Individual Experiments

```bash
# QI-FedDetect on NSL-KDD
python run_all_experiments.py --dataset nslkdd --method qi_feddetect --runs 5

# FedAvg on CICIDS2017
python run_all_experiments.py --dataset cicids2017 --method fedavg --runs 5

# FedProx on NSL-KDD
python run_all_experiments.py --dataset nslkdd --method fedprox --runs 5
```

---

## Random Seeds and Hyperparameters

To exactly reproduce the reported results, use the following seeds (applied in order across the 5 runs):

| Run | Seed |
|-----|------|
| 1   | 42   |
| 2   | 123  |
| 3   | 456  |
| 4   | 789  |
| 5   | 1011 |

### Key Hyperparameters

| Parameter                        | Value               |
|----------------------------------|---------------------|
| Number of clients (N)            | 10                  |
| Byzantine clients (f)            | 2 (20%)             |
| Communication rounds (T)         | 50                  |
| Local epochs (E)                 | 5                   |
| Client learning rate             | 0.01                |
| Batch size                       | 64                  |
| Dirichlet concentration (α)      | 0.5                 |
| VQC qubits (q)                   | 6                   |
| VQC entanglement layers (L)      | 3                   |
| BDM z-test significance level    | 0.05                |
| FedProx proximal term (μ)        | 0.01                |
| Byzantine attack type            | Sign-flip (×−10)    |

Full configuration files are in `experiments/config_nslkdd.yaml` and `experiments/config_cicids2017.yaml`.

---

## Framework Components

### Quantum-Inspired Feature Encoder (QIFE)

Located in `src/qife.py`. Implements a classically-simulated variational quantum circuit with:
- q = 6 qubits (log₂ of feature dimension, rounded up)
- L = 3 entanglement layers of parameterized Y-rotations + CNOT ring topology
- Output: k-dimensional quantum-amplitude feature vector via Pauli observables

### Byzantine Detection Module (BDM)

Located in `src/bdm.py`. Computes pairwise Quantum Jensen-Shannon Divergence between client model updates, applying a one-sided z-test to flag outliers as Byzantine. Implements the formal guarantee of **Theorem 1** from the paper.

### Federated Training Loop

Located in `src/federated.py`. Supports three aggregation methods:
- `qi_feddetect` — BDM-filtered weighted averaging
- `fedavg` — standard FedAvg (McMahan et al., 2017)
- `fedprox` — FedAvg with proximal regularization (Li et al., 2020)

---

## Citation

If you use QI-FedDetect in your research, please cite:

```bibtex
@article{thota2025qifeddetect,
  title     = {{QI-FedDetect}: Quantum-Inspired Federated Intrusion Detection with
               Byzantine Fault Tolerance in Heterogeneous {IoT} Networks},
  author    = {Thota, Sai Krishna},
  journal   = {IEEE Access},
  year      = {2025},
  note      = {Submitted},
  url       = {https://github.com/drsaikrishnathota1/QI-FedDetect}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Dr. Sai Krishna Thota**
Independent Researcher, USA
drsaikrishnathota@ieee.org
ORCID: [0009-0008-5246-9421](https://orcid.org/0009-0008-5246-9421)
