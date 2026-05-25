"""
federated.py — Federated training loop for QI-FedDetect, FedAvg, and FedProx.

Supports three aggregation methods:
    'qi_feddetect' — BDM-filtered weighted averaging (proposed method)
    'fedavg'       — Standard weighted averaging (McMahan et al., 2017)
    'fedprox'      — FedAvg + proximal regularisation (Li et al., 2020)

Usage:
    results = run_federated(
        method='qi_feddetect',
        X_train=..., y_train=..., X_test=..., y_test=...,
        client_indices=[...],
        byzantine_ids=[8, 9],
        config=config_dict,
    )
"""

import copy
import time
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models import get_model
from src.qife import QIFE
from src.bdm import detect_byzantine
from src.utils import (
    compute_metrics, byzantine_detection_rate,
    flatten_params, unflatten_params, get_logger
)

logger = get_logger(__name__)


# ── Local client training ─────────────────────────────────────────────────────

def local_train(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    mu: float = 0.0,          # FedProx proximal term (0 = FedAvg)
    global_params: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, float]:
    """
    Train model locally for `epochs` epochs and return updated model + avg loss.
    """
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X.to(device), y.to(device))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    total_loss = 0.0
    steps = 0

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)

            # FedProx proximal term
            if mu > 0.0 and global_params is not None:
                prox = 0.0
                for p, gp in zip(model.parameters(),
                                 _numpy_to_params(global_params, model)):
                    prox += (p - gp.to(device)).norm(2) ** 2
                loss += (mu / 2) * prox

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

    return model, total_loss / max(steps, 1)


def _numpy_to_params(flat: np.ndarray, model: nn.Module):
    """Yield parameter tensors reconstructed from a flat numpy array."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        yield torch.tensor(flat[offset: offset + numel], dtype=p.dtype).view(p.shape)
        offset += numel


# ── Byzantine attack ──────────────────────────────────────────────────────────

def sign_flip_attack(delta: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """Sign-flipping poisoning attack: negate and scale the honest gradient."""
    return -scale * delta


# ── Aggregation ───────────────────────────────────────────────────────────────

def federated_average(
    updates: Dict[int, np.ndarray],
    weights: Dict[int, float],
    honest_ids: List[int],
) -> np.ndarray:
    """Weighted average of updates from honest_ids only."""
    total_w = sum(weights[i] for i in honest_ids)
    agg = np.zeros_like(next(iter(updates.values())))
    for i in honest_ids:
        agg += (weights[i] / total_w) * updates[i]
    return agg


# ── Main federated loop ───────────────────────────────────────────────────────

def run_federated(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    client_indices: List[np.ndarray],
    byzantine_ids: List[int],
    config: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a full federated experiment.

    Args:
        method         : 'qi_feddetect', 'fedavg', or 'fedprox'
        X_train/y_train: full training set (numpy arrays)
        X_test/y_test  : test set
        client_indices : list of index arrays (one per client)
        byzantine_ids  : which client IDs are Byzantine
        config         : hyperparameter dict (see experiments/config_*.yaml)
        seed           : random seed for this run

    Returns:
        dict with keys: accuracy, f1_macro, byzantine_detection_rate,
                        per_round_accuracy, total_time_seconds
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{method.upper()}] seed={seed}  device={device}  "
                f"byzantine={byzantine_ids}")

    n_clients   = len(client_indices)
    n_rounds    = config["n_rounds"]
    local_epochs= config["local_epochs"]
    lr          = config["lr"]
    batch_size  = config["batch_size"]
    mu          = config.get("fedprox_mu", 0.0) if method == "fedprox" else 0.0
    bdm_alpha   = config.get("bdm_alpha", 0.05)
    n_qubits    = config.get("n_qubits", 6)
    n_layers    = config.get("n_layers", 3)
    num_classes = config.get("num_classes", 5)

    # ── Build QIFE encoder ────────────────────────────────────────────────
    raw_input_dim = X_train.shape[1]
    qife = QIFE(input_dim=raw_input_dim, n_qubits=n_qubits, n_layers=n_layers)
    qife.eval()

    def encode(X_np: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            X_t = torch.tensor(X_np, dtype=torch.float32)
            return qife(X_t)

    logger.info("Encoding training features through QIFE ...")
    t0 = time.time()
    X_enc_train = encode(X_train)
    X_enc_test  = encode(X_test)
    encode_time = time.time() - t0
    logger.info(f"QIFE encoding done in {encode_time:.1f}s  "
                f"output_dim={X_enc_train.shape[1]}")

    input_dim = X_enc_train.shape[1]

    # Per-client encoded tensors
    client_data = []
    for idx in client_indices:
        Xc = X_enc_train[idx]
        yc = torch.tensor(y_train[idx], dtype=torch.long)
        client_data.append((Xc, yc))

    # ── Initialise global model ───────────────────────────────────────────
    global_model = get_model(input_dim=input_dim, num_classes=num_classes)
    global_model.to(device)
    global_params = flatten_params(global_model)

    # Dataset sizes for weighted aggregation
    weights = {i: len(client_indices[i]) for i in range(n_clients)}

    per_round_acc   = []
    per_round_bdr   = []
    start_time = time.time()

    for rnd in range(1, n_rounds + 1):
        updates: Dict[int, np.ndarray] = {}

        # ── Local training ────────────────────────────────────────────────
        for client_id in range(n_clients):
            local_model = copy.deepcopy(global_model)
            Xc, yc = client_data[client_id]

            if client_id in byzantine_ids:
                # Byzantine: submit sign-flipped global gradient
                updates[client_id] = sign_flip_attack(global_params)
            else:
                trained, _ = local_train(
                    model=local_model,
                    X=Xc, y=yc,
                    epochs=local_epochs,
                    lr=lr,
                    batch_size=batch_size,
                    mu=mu,
                    global_params=global_params,
                    device=device,
                )
                updates[client_id] = flatten_params(trained) - global_params

        # ── Aggregation ───────────────────────────────────────────────────
        if method == "qi_feddetect":
            honest_ids, detected_byz, _ = detect_byzantine(
                updates, significance=bdm_alpha
            )
            bdr = byzantine_detection_rate(byzantine_ids, detected_byz)
        else:
            # FedAvg / FedProx: no Byzantine filtering
            honest_ids = list(range(n_clients))
            bdr = 0.0

        agg_delta  = federated_average(updates, weights, honest_ids)
        global_params = global_params + config.get("server_lr", 1.0) * agg_delta
        unflatten_params(global_model, global_params)

        # ── Evaluation ────────────────────────────────────────────────────
        global_model.eval()
        with torch.no_grad():
            logits = global_model(X_enc_test.to(device))
            preds  = logits.argmax(dim=1).cpu().numpy()
        metrics = compute_metrics(y_test, preds)
        per_round_acc.append(metrics["accuracy"])
        per_round_bdr.append(bdr)

        if rnd % 10 == 0 or rnd == n_rounds:
            logger.info(
                f"  Round {rnd:3d}/{n_rounds} | "
                f"Acc={metrics['accuracy']:.4f}  "
                f"F1={metrics['f1_macro']:.4f}  "
                f"ByzDet={bdr:.3f}"
            )

    total_time = time.time() - start_time

    # Final evaluation
    global_model.eval()
    with torch.no_grad():
        logits = global_model(X_enc_test.to(device))
        preds  = logits.argmax(dim=1).cpu().numpy()
    final_metrics = compute_metrics(y_test, preds)
    final_bdr = float(np.mean(per_round_bdr[-10:]))   # avg over last 10 rounds

    return {
        "method":                   method,
        "seed":                     seed,
        "accuracy":                 final_metrics["accuracy"],
        "f1_macro":                 final_metrics["f1_macro"],
        "byzantine_detection_rate": final_bdr,
        "per_round_accuracy":       per_round_acc,
        "per_round_bdr":            per_round_bdr,
        "total_time_seconds":       total_time,
    }
