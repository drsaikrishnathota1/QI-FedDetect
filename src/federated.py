"""
federated.py — Federated training loop for QI-FedDetect and baselines.

Supports six aggregation methods:
    'qi_feddetect' — BDM-filtered weighted averaging (proposed method)
    'fltrust'      — FLTrust-style trust-weighted robust aggregation
    'trimmed_mean' — Coordinate-wise trimmed mean
    'krum'         — Krum robust aggregation
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
    if not honest_ids:
        honest_ids = list(updates.keys())
    total_w = sum(weights[i] for i in honest_ids)
    agg = np.zeros_like(next(iter(updates.values())))
    for i in honest_ids:
        agg += (weights[i] / total_w) * updates[i]
    return agg


def _flag_top_scores(scores: Dict[int, float], n_flag: int, reverse: bool = True) -> List[int]:
    """Return client IDs with the largest or smallest scores."""
    if n_flag <= 0:
        return []
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=reverse)
    return [client_id for client_id, _ in ordered[:n_flag]]


def krum_aggregate(
    updates: Dict[int, np.ndarray],
    n_byzantine: int,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Krum aggregation with score-based Byzantine ranking for reporting."""
    client_ids = sorted(updates.keys())
    n_clients = len(client_ids)
    neighbor_count = max(1, n_clients - n_byzantine - 2)
    scores: Dict[int, float] = {}

    for client_id in client_ids:
        distances = []
        for other_id in client_ids:
            if other_id == client_id:
                continue
            diff = updates[client_id] - updates[other_id]
            distances.append(float(np.dot(diff, diff)))
        scores[client_id] = float(sum(sorted(distances)[:neighbor_count]))

    selected_id = min(scores, key=scores.get)
    detected_byz = _flag_top_scores(scores, n_byzantine, reverse=True)
    return updates[selected_id].copy(), [selected_id], detected_byz


def trimmed_mean_aggregate(
    updates: Dict[int, np.ndarray],
    n_byzantine: int,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Coordinate-wise trimmed mean with trim-frequency detection reporting."""
    client_ids = sorted(updates.keys())
    matrix = np.stack([updates[i] for i in client_ids], axis=0)
    n_clients = matrix.shape[0]
    trim = min(n_byzantine, max(0, (n_clients - 1) // 2))

    if trim == 0:
        return matrix.mean(axis=0), client_ids, []

    order = np.argsort(matrix, axis=0)
    keep_mask = np.ones_like(matrix, dtype=bool)
    low = order[:trim, :]
    high = order[n_clients - trim:, :]
    cols = np.arange(matrix.shape[1])
    keep_mask[low, cols] = False
    keep_mask[high, cols] = False

    trimmed = np.where(keep_mask, matrix, np.nan)
    agg = np.nanmean(trimmed, axis=0)
    trim_counts = {
        client_ids[i]: int((~keep_mask[i]).sum())
        for i in range(n_clients)
    }
    detected_byz = _flag_top_scores(trim_counts, n_byzantine, reverse=True)
    honest_ids = [i for i in client_ids if i not in detected_byz]
    return agg, honest_ids, detected_byz


def fltrust_aggregate(
    updates: Dict[int, np.ndarray],
    n_byzantine: int,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    FLTrust-style trust-weighted aggregation.

    The original FLTrust uses a trusted server dataset to derive a reference
    update. This reproducible benchmark does not ship private server data, so
    it uses the coordinate-wise median update as a deterministic pseudo-root.
    """
    client_ids = sorted(updates.keys())
    matrix = np.stack([updates[i] for i in client_ids], axis=0)
    root = np.median(matrix, axis=0)
    root_norm = np.linalg.norm(root) + 1e-12
    scores: Dict[int, float] = {}
    scaled_updates = []

    for client_id in client_ids:
        update = updates[client_id]
        norm = np.linalg.norm(update) + 1e-12
        trust = max(0.0, float(np.dot(update, root) / (norm * root_norm)))
        scores[client_id] = trust
        scaled_updates.append(trust * root_norm * update / norm)

    total_trust = sum(scores.values())
    detected_byz = _flag_top_scores(scores, n_byzantine, reverse=False)
    honest_ids = [i for i in client_ids if i not in detected_byz]

    if total_trust < 1e-12:
        return matrix.mean(axis=0), honest_ids, detected_byz

    agg = np.zeros_like(root)
    for client_id, scaled in zip(client_ids, scaled_updates):
        agg += (scores[client_id] / total_trust) * scaled
    return agg, honest_ids, detected_byz


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
        method         : one of qi_feddetect, fltrust, trimmed_mean, krum,
                         fedavg, or fedprox
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
            agg_delta = federated_average(updates, weights, honest_ids)
        elif method == "krum":
            agg_delta, honest_ids, detected_byz = krum_aggregate(
                updates, n_byzantine=len(byzantine_ids)
            )
            bdr = byzantine_detection_rate(byzantine_ids, detected_byz)
        elif method == "trimmed_mean":
            agg_delta, honest_ids, detected_byz = trimmed_mean_aggregate(
                updates, n_byzantine=len(byzantine_ids)
            )
            bdr = byzantine_detection_rate(byzantine_ids, detected_byz)
        elif method == "fltrust":
            agg_delta, honest_ids, detected_byz = fltrust_aggregate(
                updates, n_byzantine=len(byzantine_ids)
            )
            bdr = byzantine_detection_rate(byzantine_ids, detected_byz)
        else:
            # FedAvg / FedProx: no Byzantine filtering
            honest_ids = list(range(n_clients))
            bdr = 0.0
            agg_delta = federated_average(updates, weights, honest_ids)

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
