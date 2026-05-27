"""
bdm.py — Byzantine Detection Module (BDM).

Identifies Byzantine (malicious) clients by computing pairwise
Quantum Jensen-Shannon Divergence (QJSD) between client model updates,
then applying a one-sided z-test to flag outliers.

Implements Theorem 1 from the paper:
    If ||Delta_j - mu_G|| >= kappa > sqrt(4f/n) * delta for all Byzantine j,
    the BDM correctly identifies all Byzantine clients w.p. >= 1 - exp(-C*n).

Reference: QI-FedDetect (Thota, 2026)
"""

import numpy as np
from statistics import NormalDist
from typing import Dict, List, Tuple


def _von_neumann_entropy(rho: np.ndarray) -> float:
    """
    Compute von Neumann entropy S(rho) = -Tr(rho log rho).
    rho : square density matrix (n x n), must be PSD with Tr=1.
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]   # numerical stability
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def _outer_density_matrix(delta: np.ndarray) -> np.ndarray:
    """
    Compute outer-product density matrix of a flattened gradient update.
    rho_i = (delta @ delta.T) / ||delta||^2
    """
    flat = delta.flatten()
    norm_sq = np.dot(flat, flat)
    if norm_sq < 1e-12:
        n = len(flat)
        return np.eye(n) / n
    return np.outer(flat, flat) / norm_sq


def _pairwise_topk_projection(
    delta_i: np.ndarray,
    delta_j: np.ndarray,
    k: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project two updates onto their shared top-k high-energy coordinates."""
    flat_i = delta_i.flatten()
    flat_j = delta_j.flatten()
    k = min(k, len(flat_i), len(flat_j))
    if k <= 0:
        return np.zeros(1), np.zeros(1)

    energy = np.abs(flat_i) + np.abs(flat_j)
    if len(energy) > k:
        idx = np.argpartition(energy, -k)[-k:]
        idx = idx[np.argsort(idx)]
    else:
        idx = np.arange(len(energy))

    return flat_i[idx], flat_j[idx]


def qjsd(delta_i: np.ndarray, delta_j: np.ndarray) -> float:
    """
    Quantum Jensen-Shannon Divergence between two gradient updates.

    QJSD(rho_i || rho_j) = S((rho_i + rho_j)/2) - (S(rho_i) + S(rho_j)) / 2

    For efficiency, operates on a compressed projection of the updates
    rather than the full outer-product matrix. A cosine direction term is
    added because pure outer-product density matrices are invariant to a
    global sign flip, while sign-flip poisoning is explicitly adversarial for
    gradient aggregation.

    Args:
        delta_i, delta_j : 1-D numpy arrays (flattened gradient updates)
    Returns:
        Direction-aware QJSD score, where larger values indicate more
        divergent update geometry or direction.
    """
    # Project onto a shared low-dimensional subspace for tractability. The
    # previous SVD sketch multiplied arrays with incompatible shapes for real
    # flattened model updates; a shared top-k coordinate projection keeps the
    # comparison deterministic and dimensionally valid.
    sketch_i, sketch_j = _pairwise_topk_projection(delta_i, delta_j, k=16)

    rho_i = _outer_density_matrix(sketch_i)
    rho_j = _outer_density_matrix(sketch_j)
    rho_m = (rho_i + rho_j) / 2

    s_m  = _von_neumann_entropy(rho_m)
    s_i  = _von_neumann_entropy(rho_i)
    s_j  = _von_neumann_entropy(rho_j)

    density_divergence = max(0.0, s_m - (s_i + s_j) / 2)

    denom = (np.linalg.norm(sketch_i) * np.linalg.norm(sketch_j)) + 1e-12
    cosine = float(np.dot(sketch_i, sketch_j) / denom)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    direction_divergence = 0.5 * (1.0 - cosine)

    return density_divergence + direction_divergence


def detect_byzantine(
    updates: Dict[int, np.ndarray],
    significance: float = 0.05,
) -> Tuple[List[int], List[int], np.ndarray]:
    """
    Run BDM on a round's client updates.

    Args:
        updates      : {client_id: flattened_gradient_update}
        significance : z-test significance level alpha (default 0.05)

    Returns:
        honest_ids   : list of client IDs classified as honest
        byzantine_ids: list of client IDs classified as Byzantine
        score_matrix : (N x N) pairwise QJSD matrix
    """
    client_ids = sorted(updates.keys())
    n = len(client_ids)

    # Build pairwise QJSD matrix
    score_matrix = np.zeros((n, n))
    for i, ci in enumerate(client_ids):
        for j, cj in enumerate(client_ids):
            if i < j:
                d = qjsd(updates[ci], updates[cj])
                score_matrix[i, j] = d
                score_matrix[j, i] = d

    # Mean QJSD for each client (average divergence to all others)
    mean_scores = score_matrix.mean(axis=1)   # shape (n,)

    # One-sided z-test: flag clients whose score is a significant outlier
    mu    = mean_scores.mean()
    sigma = mean_scores.std(ddof=1) if n > 1 else 1.0
    if sigma < 1e-10:
        # All updates identical — no Byzantine clients
        return client_ids, [], score_matrix

    z_scores = (mean_scores - mu) / sigma
    threshold_z = NormalDist().inv_cdf(1 - significance)

    honest_ids    = [client_ids[i] for i in range(n) if z_scores[i] <= threshold_z]
    byzantine_ids = [client_ids[i] for i in range(n) if z_scores[i] >  threshold_z]

    return honest_ids, byzantine_ids, score_matrix
