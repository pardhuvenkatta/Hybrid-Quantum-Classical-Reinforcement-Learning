import numpy as np
from typing import Tuple, Dict, Any, List
from src.environment.gridworld import GridWorld, ACTIONS
from src.quantum.quantum_encoding import StateActionEncoder

class QuantumRNNAgent:
    """
    Quantum-Gated Recurrent World Model (Q-RNN).
    Uses a VQC hidden state evolution |h_t> = VQC(|h_{t-1}>, s_t, a_t) for recurrent trajectory evaluation.
    Reference: Hollmann, DLR / Univ. Ulm (2026).
    """
    def __init__(
        self,
        env: GridWorld,
        horizon: int = 3,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1
    ):
        self.env = env
        self.horizon = horizon
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.encoder = StateActionEncoder(grid_width=env.width, grid_height=env.height, depth=horizon)
        self.num_qubits = self.encoder.num_qubits
        self.hilbert_dim = self.encoder.hilbert_dim

        # VQC Recurrent hidden weights
        self.vqc_rnn_weights = np.random.randn(self.num_qubits * 2) * 0.1
        self.q_table = np.zeros((env.num_states, env.num_actions))
        self.parameter_count = int(self.q_table.size + self.vqc_rnn_weights.size)

    def _quantum_recurrent_step(self, pos: Tuple[int, int], prev_h: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        r, c = pos
        theta_r = np.pi * (r / max(1, self.env.height - 1))
        theta_c = np.pi * (c / max(1, self.env.width - 1))

        # Quantum hidden state update via VQC rotation
        h_next = prev_h.copy()
        for i in range(self.hilbert_dim):
            w = self.vqc_rnn_weights[i % len(self.vqc_rnn_weights)]
            phase = theta_r * np.cos(w + action) + theta_c * np.sin(w + action)
            h_next[i] *= np.exp(1j * phase)

        norm = np.linalg.norm(h_next)
        if norm > 0:
            h_next = h_next / norm

        # Pred reward from measurement expectation
        predicted_reward = float(np.real(np.vdot(h_next, prev_h))) - 0.5
        return h_next, predicted_reward

    def select_action(self, pos: Tuple[int, int]) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(ACTIONS))

        state_idx = self.env.pos_to_state(pos)
        action_values = np.zeros(self.env.num_actions)
        h_0 = np.ones(self.hilbert_dim, dtype=complex) / np.sqrt(self.hilbert_dim)

        for a in ACTIONS:
            h_t, r_pred = self._quantum_recurrent_step(pos, h_0, a)
            action_values[a] = r_pred + self.q_table[state_idx, a]

        max_v = np.max(action_values)
        best_a = np.where(action_values == max_v)[0]
        return int(np.random.choice(best_a))

    def update(self, state_pos: Tuple[int, int], action: int, reward: float, next_pos: Tuple[int, int], done: bool):
        s = self.env.pos_to_state(state_pos)
        s_next = self.env.pos_to_state(next_pos)
        best_next_q = 0.0 if done else np.max(self.q_table[s_next])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[s, action]
        self.q_table[s, action] += self.lr * td_error
