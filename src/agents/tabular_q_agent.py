import numpy as np
from typing import Tuple, Any, Optional

class TabularQAgent:
    """
    Model-Free Tabular Q-Learning Agent.
    Updates Q(s, a) exclusively from real environment experience tuples.
    """
    def __init__(
        self,
        num_states: int = 16,
        num_actions: int = 4,
        lr: float = 0.10,
        gamma: float = 0.99,
        epsilon: float = 0.10
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))
        self.parameter_count = self.q_table.size

    def _state_to_idx(self, state: Any) -> int:
        if isinstance(state, tuple):
            return state[0] * 4 + state[1]
        elif isinstance(state, int):
            return state % self.num_states
        return 0

    def select_action(self, state: Any) -> int:
        state_idx = self._state_to_idx(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_vals = self.q_table[state_idx]
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        s = self._state_to_idx(state)
        s_next = self._state_to_idx(next_state)
        
        target = reward if done else reward + self.gamma * np.max(self.q_table[s_next])
        td_error = target - self.q_table[s, action]
        self.q_table[s, action] += self.lr * td_error
