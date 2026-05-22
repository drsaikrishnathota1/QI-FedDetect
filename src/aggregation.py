"""
aggregation.py
QI-FedDetect: Quantum-Inspired Federated Learning Aggregation Algorithm
Core novel contribution of the paper.
"""

import numpy as np
import time
from src.encoding import flatten_gradients, batch_amplitude_encode, compute_similarity_matrix, detect_outliers, unflatten_gradients


class QIFedDetect:
    def __init__(self, threshold=0.5, min_clients=2):
        self.threshold = threshold
        self.min_clients = min_clients
        self.round_logs = []

    def aggregate(self, client_updates, reference_state_dict):
        t_start = time.time()
        n_clients = len(client_updates)
        state_dicts = [upd[0] for upd in client_updates]
        sample_counts = [upd[1] for upd in client_updates]
        flat_gradients = [flatten_gradients(sd) for sd in state_dicts]
        encoded = batch_amplitude_encode(flat_gradients)
        sim_matrix = compute_similarity_matrix(encoded)
        flagged = detect_outliers(sim_matrix, threshold=self.threshold)
        clean_indices = [i for i in range(n_clients) if i not in flagged]
        if len(clean_indices) < self.min_clients:
            clean_indices = list(range(n_clients))
            flagged = []
        total_samples = sum(sample_counts[i] for i in clean_indices)
        aggregated_flat = np.zeros_like(flat_gradients[0], dtype=np.float64)
        for i in clean_indices:
            weight = sample_counts[i] / total_samples
            aggregated_flat += weight * flat_gradients[i]
        aggregated_state = unflatten_gradients(aggregated_flat, reference_state_dict)
        t_end = time.time()
        round_info = {
            "n_clients": n_clients,
            "n_flagged": len(flagged),
            "flagged_indices": flagged,
            "n_clean": len(clean_indices),
            "threshold": self.threshold,
            "aggregation_time_s": round(t_end - t_start, 4),
        }
        self.round_logs.append(round_info)
        return aggregated_state, round_info


class FedAvgBaseline:
    def aggregate(self, client_updates, reference_state_dict):
        state_dicts = [upd[0] for upd in client_updates]
        sample_counts = [upd[1] for upd in client_updates]
        flat_gradients = [flatten_gradients(sd) for sd in state_dicts]
        total_samples = sum(sample_counts)
        aggregated_flat = np.zeros_like(flat_gradients[0], dtype=np.float64)
        for i, fg in enumerate(flat_gradients):
            weight = sample_counts[i] / total_samples
            aggregated_flat += weight * fg
        aggregated_state = unflatten_gradients(aggregated_flat, reference_state_dict)
        return aggregated_state, {"n_clients": len(client_updates), "n_flagged": 0}


class FedProxBaseline:
    def __init__(self, mu=0.01):
        self.mu = mu

    def aggregate(self, client_updates, reference_state_dict):
        state_dicts = [upd[0] for upd in client_updates]
        sample_counts = [upd[1] for upd in client_updates]
        flat_gradients = [flatten_gradients(sd) for sd in state_dicts]
        total_samples = sum(sample_counts)
        aggregated_flat = np.zeros_like(flat_gradients[0], dtype=np.float64)
        for i, fg in enumerate(flat_gradients):
            weight = sample_counts[i] / total_samples
            aggregated_flat += weight * fg
        aggregated_state = unflatten_gradients(aggregated_flat, reference_state_dict)
        return aggregated_state, {"n_clients": len(client_updates), "n_flagged": 0}
