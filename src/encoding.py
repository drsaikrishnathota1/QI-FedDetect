"""
encoding.py
Quantum-inspired amplitude encoding for gradient vectors.
"""

import numpy as np
import torch


def amplitude_encode(gradient_vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(gradient_vector)
    if norm < 1e-10:
        d = len(gradient_vector)
        return np.ones(d) / np.sqrt(d)
    return gradient_vector / norm


def batch_amplitude_encode(gradients: list) -> np.ndarray:
    return np.array([amplitude_encode(g) for g in gradients])


def quantum_inner_product(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    inner = np.dot(psi_a, psi_b)
    return float(inner ** 2)


def compute_similarity_matrix(encoded_gradients: np.ndarray) -> np.ndarray:
    n = len(encoded_gradients)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = quantum_inner_product(
                encoded_gradients[i], encoded_gradients[j]
            )
    return sim_matrix


def detect_outliers(similarity_matrix: np.ndarray, threshold: float = 0.5) -> list:
    n = similarity_matrix.shape[0]
    mean_similarities = []
    for i in range(n):
        others = [similarity_matrix[i, j] for j in range(n) if j != i]
        mean_similarities.append(np.mean(others))
    flagged = [i for i, s in enumerate(mean_similarities) if s < threshold]
    return flagged


def flatten_gradients(model_state_dict: dict) -> np.ndarray:
    tensors = [v.cpu().numpy().flatten() for v in model_state_dict.values()]
    return np.concatenate(tensors)


def unflatten_gradients(flat_vector: np.ndarray, reference_state_dict: dict) -> dict:
    import torch
    new_state = {}
    idx = 0
    for key, tensor in reference_state_dict.items():
        size = tensor.numel()
        new_state[key] = torch.tensor(
            flat_vector[idx: idx + size].reshape(tensor.shape),
            dtype=tensor.dtype
        )
        idx += size
    return new_state
