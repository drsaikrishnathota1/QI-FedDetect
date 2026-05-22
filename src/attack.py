"""
attack.py
Gradient poisoning attack simulation for federated learning.
"""

import numpy as np
import torch
import copy


def random_noise_attack(state_dict, noise_scale=1.0):
    poisoned = copy.deepcopy(state_dict)
    for key in poisoned:
        noise = torch.randn_like(poisoned[key]) * noise_scale
        poisoned[key] = noise
    return poisoned


def sign_flip_attack(state_dict, flip_factor=-5.0):
    poisoned = copy.deepcopy(state_dict)
    for key in poisoned:
        poisoned[key] = poisoned[key] * flip_factor
    return poisoned


def scaling_attack(state_dict, scale_factor=10.0):
    poisoned = copy.deepcopy(state_dict)
    for key in poisoned:
        poisoned[key] = poisoned[key] * scale_factor
    return poisoned


def inject_poisoned_clients(client_updates, poison_fraction=0.2, attack_type="sign_flip", noise_scale=1.0, scale_factor=10.0):
    n_clients = len(client_updates)
    n_poison = max(1, int(n_clients * poison_fraction))
    poisoned_indices = list(range(n_clients - n_poison, n_clients))
    poisoned_updates = []
    for i, (state_dict, n_samples) in enumerate(client_updates):
        if i in poisoned_indices:
            if attack_type == "random_noise":
                p_state = random_noise_attack(state_dict, noise_scale)
            elif attack_type == "sign_flip":
                p_state = sign_flip_attack(state_dict)
            elif attack_type == "scaling":
                p_state = scaling_attack(state_dict, scale_factor)
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")
            poisoned_updates.append((p_state, n_samples))
        else:
            poisoned_updates.append((state_dict, n_samples))
    return poisoned_updates, poisoned_indices
