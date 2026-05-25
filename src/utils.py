"""
utils.py — Utility functions for QI-FedDetect.

Covers: seeding, metric computation, Dirichlet data partitioning,
result logging, and CSV export.
"""

import os
import csv
import random
import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute accuracy and macro F1-score."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def byzantine_detection_rate(
    true_byzantine_ids: List[int],
    detected_byzantine_ids: List[int],
) -> float:
    """
    Fraction of true Byzantine clients correctly detected.
    Returns 0.0 if there are no Byzantine clients.
    """
    if not true_byzantine_ids:
        return 0.0
    detected = set(detected_byzantine_ids)
    true_set = set(true_byzantine_ids)
    return len(detected & true_set) / len(true_set)


# ── Non-IID Data Partitioning ─────────────────────────────────────────────────

def dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Partition dataset indices across clients using Dirichlet distribution.

    Args:
        labels    : array of class labels for the full dataset
        n_clients : number of federated clients
        alpha     : Dirichlet concentration parameter (smaller = more heterogeneous)
        seed      : random seed

    Returns:
        List of index arrays, one per client.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    client_indices = [[] for _ in range(n_clients)]

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        # Convert proportions to split points
        splits = (proportions * len(cls_idx)).astype(int)
        splits[-1] = len(cls_idx) - splits[:-1].sum()   # ensure full coverage
        splits = np.maximum(splits, 0)
        idx = 0
        for client_id, count in enumerate(splits):
            client_indices[client_id].extend(cls_idx[idx: idx + count].tolist())
            idx += count

    return [np.array(idxs) for idxs in client_indices]


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── CSV result export ─────────────────────────────────────────────────────────

def save_results_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Append result rows to a CSV file, creating headers on first write."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


# ── Gradient helpers ──────────────────────────────────────────────────────────

def flatten_params(model: torch.nn.Module) -> np.ndarray:
    """Flatten all model parameters into a 1-D numpy array."""
    return np.concatenate([
        p.detach().cpu().numpy().flatten()
        for p in model.parameters()
    ])


def unflatten_params(model: torch.nn.Module, flat: np.ndarray) -> None:
    """Load a flat numpy array back into model parameters (in-place)."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(
            torch.tensor(flat[offset: offset + numel], dtype=p.dtype).view(p.shape)
        )
        offset += numel
