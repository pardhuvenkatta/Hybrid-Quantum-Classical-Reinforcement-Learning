import numpy as np
from typing import Tuple, Dict, Any, Optional

class Grid4x4Env:
    """
    Stochastic 4x4 Grid Environment for HQWM-FSP study.
    - 16 discrete states: pos (r, c) for r in 0..3, c in 0..3.
    - Actions: 0 (Up), 1 (Down), 2 (Left), 3 (Right).
    - Intended action succeeds with probability 1 - slip_prob (default 0.90);
      otherwise a uniform random action is taken.
    - Boundaries: Staying in place if moving off grid.
    - Reward: +10.0 at Goal (3, 3) terminating episode; -0.1 step penalty.
    - Maximum episode length: 50 steps.
    """
    def __init__(self, slip_prob: float = 0.10, max_steps: int = 50):
        self.grid_size = 4
        self.num_states = 16
        self.num_actions = 4
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.goal_pos = (3, 3)
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos
        self.current_step = 0
        self.gamma = 0.99

        # Action velocity vectors: (dr, dc)
        self.action_effects = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.grid_size + pos[1]

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        return (state // self.grid_size, state % self.grid_size)

    def reset(self, seed: Optional[int] = None) -> Tuple[int, int]:
        if seed is not None:
            np.random.seed(seed)
        self.current_pos = self.start_pos
        self.current_step = 0
        return self.current_pos

    def _get_next_pos(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        dr, dc = self.action_effects[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            return (nr, nc)
        return pos  # Stay in place on boundary collision

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        self.current_step += 1

        # Stochastic slip mechanism
        if np.random.rand() < self.slip_prob:
            actual_action = np.random.randint(self.num_actions)
        else:
            actual_action = action

        next_pos = self._get_next_pos(self.current_pos, actual_action)
        self.current_pos = next_pos

        done = False
        if next_pos == self.goal_pos:
            reward = 10.0
            done = True
        else:
            reward = -0.1

        if self.current_step >= self.max_steps:
            done = True

        info = {
            "reached_goal": (next_pos == self.goal_pos),
            "actual_action": actual_action
        }
        return next_pos, reward, done, info

    def simulate_step(self, pos: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Deterministic look-ahead step without environment state mutation."""
        next_pos = self._get_next_pos(pos, action)
        if next_pos == self.goal_pos:
            return next_pos, 10.0, True
        return next_pos, -0.1, False
