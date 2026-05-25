"""
qife.py — Quantum-Inspired Feature Encoder (QIFE).

Implements a classically-simulated variational quantum circuit (VQC) that encodes
raw network traffic features into a quantum-amplitude feature vector.

Circuit structure per layer l:
    1. Parameterised Y-rotations:  R_y(W_l @ x + b_l)  applied to each qubit
    2. CNOT ring entanglement:     qubit i -> qubit (i+1) % n_qubits

Output: expectation values of Pauli-Z observables on each qubit → z ∈ R^{n_qubits}

Reference: Farhi & Neven (2018), arXiv:1802.06002
"""

import math
import torch
import torch.nn as nn


class QIFE(nn.Module):
    """
    Quantum-Inspired Feature Encoder.

    Args:
        input_dim  : dimensionality of raw input features (d)
        n_qubits   : number of simulated qubits (default: 6, so 2^6 = 64 amplitudes)
        n_layers   : number of VQC encoding layers L (default: 3)
    """

    def __init__(self, input_dim: int, n_qubits: int = 6, n_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.output_dim = n_qubits          # one Pauli-Z expectation per qubit

        # Trainable encoding weights W_l ∈ R^{n_qubits × input_dim} and bias b_l ∈ R^{n_qubits}
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(n_qubits, input_dim) * 0.1)
            for _ in range(n_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(n_qubits))
            for _ in range(n_layers)
        ])

    # ------------------------------------------------------------------
    # Statevector simulation helpers
    # ------------------------------------------------------------------

    def _ry_gate(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Build batched R_y(theta) rotation matrices.
        theta : (batch, n_qubits)
        returns: (batch, n_qubits, 2, 2)
        """
        c = torch.cos(theta / 2)          # (batch, n_qubits)
        s = torch.sin(theta / 2)
        # [[cos, -sin], [sin, cos]]
        row0 = torch.stack([ c, -s], dim=-1)   # (batch, n_qubits, 2)
        row1 = torch.stack([ s,  c], dim=-1)
        return torch.stack([row0, row1], dim=-2)   # (batch, n_qubits, 2, 2)

    def _apply_ry_layer(self, state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Apply independent R_y(theta_i) to qubit i of the statevector.
        state : (batch, 2^n_qubits)  complex amplitudes
        theta : (batch, n_qubits)
        """
        batch = state.shape[0]
        n = self.n_qubits
        state = state.view(batch, *([2] * n))   # (batch, 2, 2, ..., 2)

        for q in range(n):
            gate = self._ry_gate(theta[:, q])   # (batch, 2, 2)
            # Move qubit q to axis 1 for matmul, then move back
            axes = list(range(n + 1))
            axes[1], axes[q + 1] = axes[q + 1], axes[1]
            state = state.permute(axes)         # qubit q now at dim 1
            # state shape: (batch, 2, 2, ...) → flatten all but batch and qubit-q
            s = state.reshape(batch, 2, -1)     # (batch, 2, rest)
            # gate: (batch, 2, 2)  ×  s: (batch, 2, rest) → (batch, 2, rest)
            s = torch.bmm(gate, s)
            state = s.reshape(batch, *state.shape[1:])
            state = state.permute(axes)         # undo permutation

        return state.reshape(batch, -1)

    def _apply_cnot_ring(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply CNOT ring: qubit i controls qubit (i+1) % n_qubits, for i in 0..n-1.
        Operates on real-valued statevectors (Ry gates keep states real).
        state : (batch, 2^n_qubits)
        """
        batch = state.shape[0]
        n = self.n_qubits
        state = state.view(batch, *([2] * n))

        for ctrl in range(n):
            tgt = (ctrl + 1) % n
            # CNOT: flip target qubit when control qubit == 1
            axes = list(range(n + 1))
            # bring ctrl to dim 1, tgt to dim 2
            axes[1], axes[ctrl + 1] = axes[ctrl + 1], axes[1]
            state = state.permute(axes)
            axes2 = list(range(n + 1))
            tgt_new = axes.index(tgt + 1)          # find where tgt ended up
            axes2[2], axes2[tgt_new] = axes2[tgt_new], axes2[2]
            state = state.permute(axes2)
            # state[:, ctrl_val, tgt_val, ...]
            s = state.clone()
            # swap |ctrl=1, tgt=0⟩ ↔ |ctrl=1, tgt=1⟩
            s[:, 1, 0], s[:, 1, 1] = state[:, 1, 1].clone(), state[:, 1, 0].clone()
            state = s
            # undo permutations
            state = state.permute([axes2.index(i) for i in range(n + 1)])
            state = state.permute([axes.index(i) for i in range(n + 1)])

        return state.reshape(batch, -1)

    def _measure_pauli_z(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute <Z_i> = Σ_{s: s_i=0} |a_s|^2 − Σ_{s: s_i=1} |a_s|^2 for each qubit i.
        state : (batch, 2^n_qubits)
        returns: (batch, n_qubits)
        """
        batch = state.shape[0]
        n = self.n_qubits
        probs = state ** 2                          # real statevector, so no .abs()
        probs = probs.view(batch, *([2] * n))
        expectations = []
        for q in range(n):
            # Sum over all axes except qubit q
            axes = list(range(1, n + 1))
            axes.remove(q + 1)
            p0 = probs.select(q + 1, 0).sum(dim=list(range(1, n)))
            p1 = probs.select(q + 1, 1).sum(dim=list(range(1, n)))
            expectations.append(p0 - p1)
        return torch.stack(expectations, dim=1)    # (batch, n_qubits)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features x into quantum-amplitude feature vector z.

        x : (batch, input_dim)
        returns z : (batch, n_qubits)  values in [-1, 1]
        """
        batch = x.shape[0]

        # Initialise |0...0⟩ statevector — first basis element is 1, rest 0
        state = torch.zeros(batch, 2 ** self.n_qubits, device=x.device, dtype=x.dtype)
        state[:, 0] = 1.0

        for l in range(self.n_layers):
            # Compute rotation angles: theta_l = W_l @ x + b_l
            theta = x @ self.W[l].T + self.b[l]   # (batch, n_qubits)
            state = self._apply_ry_layer(state, theta)
            state = self._apply_cnot_ring(state)

        z = self._measure_pauli_z(state)           # (batch, n_qubits)
        return z
