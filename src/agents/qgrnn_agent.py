import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from src.quantum.quantum_encoding import StateActionEncoder

class QGRNNAgent:
    """
    Architecture 3: Quantum-Gated Recurrent World Models (QGRNN).
    Embeds a VQC inside the recurrent cell (QGRU/QLSTM) of a model-based RL agent.
    Reference: Hollmann, DLR / Univ. Ulm (2026).
    """
    def __init__(
        self,
        env: Any,
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

        self.num_actions = getattr(env, "num_actions", 4)
        self.num_states = getattr(env, "num_states", 16)

        grid_w = getattr(env, "width", 4)
        grid_h = getattr(env, "height", 4)
        self.encoder = StateActionEncoder(grid_width=grid_w, grid_height=grid_h, depth=horizon)
        self.num_qubits = self.encoder.num_qubits
        self.hilbert_dim = self.encoder.hilbert_dim

        # VQC Recurrent hidden weights
        np.random.seed(42)
        self.vqc_rnn_weights = np.random.randn(self.num_qubits * 2) * 0.1
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.parameter_count = len(self.vqc_rnn_weights) + (self.num_states * self.num_actions)

    def _state_to_idx(self, state: Any) -> int:
        if hasattr(self.env, "pos_to_state") and isinstance(state, tuple):
            return self.env.pos_to_state(state)
        elif isinstance(state, int):
            return state % self.num_states
        elif isinstance(state, tuple):
            return (state[0] * 4 + state[1]) % self.num_states
        return 0

    def _quantum_recurrent_step(self, state: Any, prev_h: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        s_idx = self._state_to_idx(state)
        theta_r = np.pi * (s_idx / self.num_states)

        h_next = prev_h.copy()
        for i in range(self.hilbert_dim):
            w = self.vqc_rnn_weights[i % len(self.vqc_rnn_weights)]
            phase = theta_r * np.cos(w + action)
            h_next[i] *= np.exp(1j * phase)

        norm = np.linalg.norm(h_next)
        if norm > 0:
            h_next = h_next / norm

        predicted_reward = float(np.real(np.vdot(h_next, prev_h))) - 0.5
        return h_next, predicted_reward

    def select_action(self, state: Any) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(range(self.num_actions)))

        s_idx = self._state_to_idx(state)
        action_values = np.zeros(self.num_actions)
        h_0 = np.ones(self.hilbert_dim, dtype=complex) / np.sqrt(self.hilbert_dim)

        for a in range(self.num_actions):
            h_t, r_pred = self._quantum_recurrent_step(state, h_0, a)
            action_values[a] = r_pred + self.q_table[s_idx, a]

        max_v = np.max(action_values)
        best_a = np.where(action_values == max_v)[0]
        return int(np.random.choice(best_a))

    def update(self, *args, **kwargs):
        if len(args) >= 5:
            state, action, reward, next_state, done = args[0], args[1], args[2], args[3], args[4]
        elif "state" in kwargs:
            state = kwargs["state"]
            action = kwargs.get("action", 0)
            reward = kwargs.get("reward", 0.0)
            next_state = kwargs.get("next_state", state)
            done = kwargs.get("done", False)
        else:
            return

        s = self._state_to_idx(state)
        s_next = self._state_to_idx(next_state)
        best_next_q = 0.0 if done else np.max(self.q_table[s_next])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[s, action]
        self.q_table[s, action] += self.lr * td_error
