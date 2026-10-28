"""
Uniform random baseline agent for RL benchmark tasks.
"""
import numpy as np
from typing import Any

class RandomAgent:
    """
    Uniform-random policy baseline.
    Selects actions with equal probability across the discrete action space.
    """
    def __init__(self, env: Any, horizon: int = 1):
        self.env = env
        self.horizon = horizon
        self.num_actions = getattr(env, "num_actions", 4)
        self.parameter_count = 0

    def select_action(self, state: Any) -> int:
        return int(np.random.randint(0, self.num_actions))

    def update(self, *args, **kwargs):
        pass
