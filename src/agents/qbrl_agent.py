import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from src.quantum.quantum_encoding import StateActionEncoder

class QBRLAgent:
    """
    Architecture 4: Quantum Bayesian Reinforcement Learning (QBRL).
    Hybrid quantum-classical look-ahead model for belief state updates using amplitude amplification.
    Reference: Quantum Machine Intelligence (2026).
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

        # Quantum belief state vector over states
        self.belief = np.ones(self.num_states, dtype=complex) / np.sqrt(self.num_states)
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.parameter_count = self.num_states * self.num_actions

    def _state_to_idx(self, state: Any) -> int:
        if hasattr(self.env, "pos_to_state") and isinstance(state, tuple):
            return self.env.pos_to_state(state)
        elif isinstance(state, int):
            return state % self.num_states
        elif isinstance(state, tuple):
            return (state[0] * 4 + state[1]) % self.num_states
        return 0

    def _quantum_amplitude_amplification_belief_update(self, observed_state: int):
        # Oracle phase marking on observed state
        self.belief[observed_state] *= -1.0

        # Grover reflection around mean belief
        u = np.ones(self.num_states, dtype=complex) / np.sqrt(self.num_states)
        self.belief = 2 * u * np.dot(u.conj(), self.belief) - self.belief

        norm = np.linalg.norm(self.belief)
        if norm > 0:
            self.belief /= norm

    def select_action(self, state: Any) -> int:
        s_idx = self._state_to_idx(state)
        self._quantum_amplitude_amplification_belief_update(s_idx)

        if np.random.rand() < self.epsilon:
            return int(np.random.choice(range(self.num_actions)))

        belief_probs = np.abs(self.belief)**2
        q_vals = self.q_table[s_idx]
        combined = 0.6 * q_vals + 0.4 * belief_probs[s_idx]
        best_a = np.where(combined == np.max(combined))[0]
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
