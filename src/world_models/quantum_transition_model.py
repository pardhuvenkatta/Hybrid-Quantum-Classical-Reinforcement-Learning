import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class QuantumTransitionModel:
    """
    Quantum Conditional Transition World Model (4 Qubits, 16 Basis States).
    - Encoding: U_enc(x) = tensor product of R_y(pi * x_j) for 4D feature x(s,a)
    - Variational: L blocks of R_y, R_z rotations + CNOT entangling chain
    - Born rule measurement: P_q(s' = j | s, a; theta) = |<j|psi_theta(s,a)>|^2
    """
    def __init__(self, num_blocks: int = 3, lr: float = 0.05, eps_smoothing: float = 0.02):
        self.num_qubits = 4
        self.hilbert_dim = 16
        self.num_blocks = num_blocks
        self.lr = lr
        self.eps_smoothing = eps_smoothing

        np.random.seed(42)
        self.params = np.random.uniform(0, 2 * np.pi, (num_blocks, 4, 2))
        self.parameter_count = self.params.size
        self.circuit_depth = num_blocks * 3
        self.two_qubit_gates = num_blocks * 4

        self.total_circuit_executions = 0
        self.total_shots_executed = 0

    def _encode_input(self, state: Any, action: int) -> np.ndarray:
        if isinstance(state, tuple):
            r, c = state[0], state[1]
        else:
            r, c = state // 4, state % 4
        b0 = float((action >> 1) & 1)
        b1 = float(action & 1)
        return np.array([r / 3.0, c / 3.0, b0, b1], dtype=float)

    def _single_qubit_ry(self, theta: float) -> np.ndarray:
        cos_t, sin_t = np.cos(theta / 2.0), np.sin(theta / 2.0)
        return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=complex)

    def _single_qubit_rz(self, theta: float) -> np.ndarray:
        return np.array([[np.exp(-1j * theta / 2.0), 0], [0, np.exp(1j * theta / 2.0)]], dtype=complex)

    def _apply_gate(self, psi: np.ndarray, gate: np.ndarray, target_qubit: int) -> np.ndarray:
        shape = [2] * self.num_qubits
        psi_tensor = psi.reshape(shape)
        psi_tensor = np.moveaxis(psi_tensor, target_qubit, 0)
        orig_shape = psi_tensor.shape
        psi_flat = psi_tensor.reshape(2, -1)
        psi_new = np.dot(gate, psi_flat).reshape(orig_shape)
        psi_new = np.moveaxis(psi_new, 0, target_qubit)
        return psi_new.reshape(-1)

    def _apply_cnot(self, psi: np.ndarray, control: int, target: int) -> np.ndarray:
        psi_out = psi.copy()
        N = len(psi)
        for i in range(N):
            c_bit = (i >> (self.num_qubits - 1 - control)) & 1
            if c_bit == 1:
                flipped_i = i ^ (1 << (self.num_qubits - 1 - target))
                if i < flipped_i:
                    psi_out[i], psi_out[flipped_i] = psi[flipped_i], psi[i]
        return psi_out

    def forward_statevector(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        if params is None:
            params = self.params

        self.total_circuit_executions += 1
        psi = np.zeros(self.hilbert_dim, dtype=complex)
        psi[0] = 1.0

        for j in range(self.num_qubits):
            ry = self._single_qubit_ry(np.pi * x[j])
            psi = self._apply_gate(psi, ry, j)

        for b in range(self.num_blocks):
            for j in range(self.num_qubits):
                ry = self._single_qubit_ry(params[b, j, 0])
                rz = self._single_qubit_rz(params[b, j, 1])
                psi = self._apply_gate(psi, ry, j)
                psi = self._apply_gate(psi, rz, j)

            for j in range(self.num_qubits):
                c = j
                t = (j + 1) % self.num_qubits
                psi = self._apply_cnot(psi, c, t)

        return psi

    def predict_next_state_distribution(self, state: Any, action: int, shots: Optional[int] = None) -> np.ndarray:
        x = self._encode_input(state, action)
        psi = self.forward_statevector(x)
        probs = np.abs(psi)**2
        probs = probs / np.sum(probs)
        if shots is not None and shots > 0:
            counts = np.random.multinomial(shots, probs)
            probs = counts / float(shots)
            self.total_shots_executed += shots
        probs = (1.0 - self.eps_smoothing) * probs + self.eps_smoothing / self.hilbert_dim
        return probs

    def train_on_batch(self, states: List[Any], actions: List[int], next_states: List[Any], steps: int = 1):
        if len(states) == 0:
            return 0.0

        # Subsample up to 10 transitions for fast gradient update
        if len(states) > 10:
            idx_sub = np.random.choice(len(states), 10, replace=False)
            states_sub = [states[i] for i in idx_sub]
            actions_sub = [actions[i] for i in idx_sub]
            next_states_sub = [next_states[i] for i in idx_sub]
        else:
            states_sub, actions_sub, next_states_sub = states, actions, next_states

        shift = np.pi / 2.0
        total_loss = 0.0

        for _ in range(steps):
            grad = np.zeros_like(self.params)
            for b in range(self.num_blocks):
                for q in range(self.num_qubits):
                    for p_idx in range(2):
                        params_plus = self.params.copy()
                        params_plus[b, q, p_idx] += shift

                        params_minus = self.params.copy()
                        params_minus[b, q, p_idx] -= shift

                        loss_plus, loss_minus = 0.0, 0.0
                        for s, a, ns in zip(states_sub, actions_sub, next_states_sub):
                            x = self._encode_input(s, a)
                            ns_idx = ns[0] * 4 + ns[1] if isinstance(ns, tuple) else ns % 16

                            psi_p = self.forward_statevector(x, params_plus)
                            p_p = np.abs(psi_p[ns_idx])**2
                            p_p = (1.0 - self.eps_smoothing) * p_p + self.eps_smoothing / 16
                            loss_plus -= np.log(p_p)

                            psi_m = self.forward_statevector(x, params_minus)
                            p_m = np.abs(psi_m[ns_idx])**2
                            p_m = (1.0 - self.eps_smoothing) * p_m + self.eps_smoothing / 16
                            loss_minus -= np.log(p_m)

                        grad[b, q, p_idx] = (loss_plus - loss_minus) / (2.0 * len(states_sub))

            self.params -= self.lr * grad

        return total_loss

    def compute_nll(self, states: List[Any], actions: List[int], next_states: List[Any]) -> float:
        if len(states) == 0:
            return 0.0
        nlls = []
        for s, a, ns in zip(states, actions, next_states):
            probs = self.predict_next_state_distribution(s, a)
            ns_idx = ns[0] * 4 + ns[1] if isinstance(ns, tuple) else ns % 16
            nlls.append(-np.log(probs[ns_idx]))
        return float(np.mean(nlls))
