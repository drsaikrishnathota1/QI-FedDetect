# QI-FedDetect: Formal Algorithm Specification

## Algorithm 1: QI-FedDetect Aggregation
G = {g1, g2, ..., gn}  — local gradients from n clients
w = {w1, w2, ..., wn}  — client sample weights
t in (0,1)             — fidelity threshold (default: 0.5)
k_min                  — minimum honest clients required
## Complexity Analysis

| Phase | Complexity |
|-------|------------|
| Amplitude encoding | O(n x d) |
| Fidelity matrix | O(n^2 x d) |
| Outlier detection | O(n^2) |
| Aggregation | O(n x d) |
| Total | O(n^2 x d) |

Communication complexity: O(n x d) — identical to FedAvg, zero overhead.

## Comparison Table

| Property | FedAvg | FedProx | QI-FedDetect |
|----------|--------|---------|--------------|
| Poisoning defense | No | No | Yes |
| Communication overhead | O(nd) | O(nd) | O(nd) |
| Quantum-inspired | No | No | Yes |
| Military IoT threat model | No | No | Yes |
| Formal detection guarantee | No | No | Yes |
