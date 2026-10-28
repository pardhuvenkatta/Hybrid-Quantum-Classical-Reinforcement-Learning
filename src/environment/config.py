import numpy as np
from typing import Tuple, List, Set, Dict, Any
from src.environment.gridworld import GridWorld

class CanonicalEnvironmentConfig:
    """
    Single Canonical Environment Configuration for all benchmark scripts.
    Grid Size: 6x6
    Start Position: (0, 0)
    Goal Position: (5, 5)
    Obstacles: [(1, 1), (1, 2), (2, 4), (3, 1), (3, 2), (4, 4)]
    Traps: [(2, 2), (4, 1)]
    Step Penalty: -0.05
    Revisit Penalty: -0.15
    Goal Reward: +1.0
    Obstacle Penalty: -0.3
    Trap Penalty: -0.8
    Max Steps: 40
    """
    GRID_WIDTH = 6
    GRID_HEIGHT = 6
    START_POS = (0, 0)
    GOAL_POS = (5, 5)
    OBSTACLES = [(1, 1), (1, 2), (2, 4), (3, 1), (3, 2), (4, 4)]
    TRAPS = [(2, 2), (4, 1)]
    STEP_PENALTY = -0.05
    REVISIT_PENALTY = -0.15
    GOAL_REWARD = 1.0
    OBSTACLE_PENALTY = -0.3
    TRAP_PENALTY = -0.8
    MAX_STEPS = 40

    @classmethod
    def create_env(cls) -> GridWorld:
        return GridWorld(
            width=cls.GRID_WIDTH,
            height=cls.GRID_HEIGHT,
            start_pos=cls.START_POS,
            goal_pos=cls.GOAL_POS,
            obstacles=cls.OBSTACLES,
            traps=cls.TRAPS,
            step_penalty=cls.STEP_PENALTY,
            revisit_penalty=cls.REVISIT_PENALTY,
            goal_reward=cls.GOAL_REWARD,
            obstacle_penalty=cls.OBSTACLE_PENALTY,
            trap_penalty=cls.TRAP_PENALTY,
            max_steps=cls.MAX_STEPS
        )
