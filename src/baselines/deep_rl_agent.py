import numpy as np
from typing import Tuple, Dict, Any
from src.environment.gridworld import GridWorld, ACTIONS

class DeepRLAgent:
    """
    Function-approximated Deep Q-Learning style baseline operating on state-action features.
    Evaluated over 50 episodes to compare against classical tabular and QWM agents.
    """
    def __init__(
        self,
        env: GridWorld,
        lr: float = 0.05,
        gamma: float = 0.95,
        epsilon: float = 0.15
    ):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # Linear function approximation over normalized (row, col) coordinates + one-hot action features
        self.num_features = 2 + 4  # pos features (x, y normalized) + 4-dim action one-hot
        self.weights = np.random.randn(self.num_features) * 0.01

    def _get_features(self, pos: Tuple[int, int], action: int) -> np.ndarray:
        feats = np.zeros(self.num_features)
        feats[0] = pos[0] / float(self.env.height)
        feats[1] = pos[1] / float(self.env.width)
        feats[2 + action] = 1.0
        return feats

    def get_q_value(self, pos: Tuple[int, int], action: int) -> float:
        phi = self._get_features(pos, action)
        return float(np.dot(self.weights, phi))

    def select_action(self, current_pos: Tuple[int, int]) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(ACTIONS))
        q_vals = [self.get_q_value(current_pos, a) for a in ACTIONS]
        max_v = max(q_vals)
        best_a = [a for a in ACTIONS if q_vals[a] == max_v]
        return int(np.random.choice(best_a))

    def update(
        self,
        state_pos: Tuple[int, int],
        action: int,
        reward: float,
        next_pos: Tuple[int, int],
        done: bool
    ):
        phi = self._get_features(state_pos, action)
        q_curr = np.dot(self.weights, phi)
        if done:
            target = reward
        else:
            q_next_max = max(self.get_q_value(next_pos, a) for a in ACTIONS)
            target = reward + self.gamma * q_next_max
        td_err = target - q_curr
        self.weights += self.lr * td_err * phi
