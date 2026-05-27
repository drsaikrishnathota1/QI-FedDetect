# QI-FedDetect: Formal Algorithm Specification

## Algorithm 1: QI-FedDetect Aggregation
G = {g1, g2, ..., gn}  — local gradients from n clients
w = {w1, w2, ..., wn}  — client sample weights
t in (0,1)             — fidelity threshold (default: 0.5)
k_min                  — minimum honest clients required

The current BDM implementation uses a direction-aware QJSD score so sign-flip
poisoning remains distinguishable from honest updates.
## Complexity Analysis

| Phase | Complexity |
|-------|------------|
| Amplitude encoding | O(n x d) |
| Fidelity matrix | O(n^2 x d) |
| Outlier detection | O(n^2) |
| Aggregation | O(n x d) |
| Total | O(n^2 x d) |

Communication complexity: O(n x d) — identical to FedAvg, zero overhead.

## Baseline Coverage

The reproducibility runner evaluates QI-FedDetect against five baselines:
FedAvg, FedProx, Krum, Trimmed Mean, and FLTrust-style trust aggregation.

| Property | FedAvg | FedProx | Krum | Trimmed Mean | FLTrust-style | QI-FedDetect |
|----------|--------|---------|------|--------------|---------------|--------------|
| Poisoning defense | No | No | Yes | Yes | Yes | Yes |
| Communication overhead | O(nd) | O(nd) | O(nd) | O(nd) | O(nd) | O(nd) |
| Quantum-inspired | No | No | No | No | No | Yes |
| IoT intrusion-detection setting | Yes | Yes | Yes | Yes | Yes | Yes |
| Formal detection guarantee | No | No | Partial | Partial | No | Yes |
