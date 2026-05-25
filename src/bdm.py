"""
bdm.py — Byzantine Detection Module (BDM).

Identifies Byzantine (malicious) clients by computing pairwise
Quantum Jensen-Shannon Divergence (QJSD) between client model updates,
then applying a one-sided z-test to flag outliers.

Implements Theorem 1 from the paper:
    If ||Delta_j - mu_G|| >= kappa > sqrt(4f/n) * delta for all Byzantine j,
    the BDM correctly identifies all Byzantine clients w.p. >= 1 - exp(-C*n).

Reference: QI-FedDetect (Thota, 2025)
"""

import numpy as np
from scipy import stats
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


def qjsd(delta_i: np.ndarray, delta_j: np.ndarray) -> float:
    """
    Quantum Jensen-Shannon Divergence between two gradient updates.

    QJSD(rho_i || rho_j) = S((rho_i + rho_j)/2) - (S(rho_i) + S(rho_j)) / 2

    For efficiency, operates on a compressed projection of the updates
    (top-k PCA directions) rather than the full outer-product matrix.

    Args:
        delta_i, delta_j : 1-D numpy arrays (flattened gradient updates)
    Returns:
        QJSD value in [0, log(2)]
    """
    # Project onto a low-dimensional subspace for tractability
    k = min(16, len(delta_i))
    mat = np.stack([delta_i.flatten(), delta_j.flatten()])   # (2, d)
    # Use SVD-based sketch: keep top-k outer products in the joint subspace
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    sketch_i = (Vt[:k] * delta_i.flatten()[:k]).flatten()[:k]
    sketch_j = (Vt[:k] * delta_j.flatten()[:k]).flatten()[:k]

    rho_i = _outer_density_matrix(sketch_i)
    rho_j = _outer_density_matrix(sketch_j)
    rho_m = (rho_i + rho_j) / 2

    s_m  = _von_neumann_entropy(rho_m)
    s_i  = _von_neumann_entropy(rho_i)
    s_j  = _von_neumann_entropy(rho_j)

    return max(0.0, s_m - (s_i + s_j) / 2)


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
    threshold_z = stats.norm.ppf(1 - significance)

    honest_ids    = [client_ids[i] for i in range(n) if z_scores[i] <= threshold_z]
    byzantine_ids = [client_ids[i] for i in range(n) if z_scores[i] >  threshold_z]

    return honest_ids, byzantine_ids, score_matrix
