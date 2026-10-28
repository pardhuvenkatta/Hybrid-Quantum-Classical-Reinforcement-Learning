import numpy as np
from typing import Tuple, List, Set, Dict, Any, Optional

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTION_NAMES = {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN", LEFT: "LEFT"}
ACTION_VECTORS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1)
}

class GridWorld:
    """
    Horizon-Sensitive Maze GridWorld Environment for Benchmark Evaluation.
    Includes state revisit penalties, traps, obstacles, and dead-end topologies.
    """
    def __init__(
        self,
        width: int = 6,
        height: int = 6,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Tuple[int, int] = (5, 5),
        obstacles: Optional[List[Tuple[int, int]]] = None,
        traps: Optional[List[Tuple[int, int]]] = None,
        step_penalty: float = -0.05,
        revisit_penalty: float = -0.15,
        goal_reward: float = 1.0,
        obstacle_penalty: float = -0.3,
        trap_penalty: float = -0.8,
        slip_prob: float = 0.0,
        max_steps: int = 40
    ):
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles: Set[Tuple[int, int]] = set(obstacles) if obstacles is not None else {(1, 1), (1, 2), (2, 4), (3, 1), (3, 2), (4, 4)}
        self.traps: Set[Tuple[int, int]] = set(traps) if traps is not None else {(2, 2), (4, 1)}
        
        self.step_penalty = step_penalty
        self.revisit_penalty = revisit_penalty
        self.goal_reward = goal_reward
        self.obstacle_penalty = obstacle_penalty
        self.trap_penalty = trap_penalty
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        
        self.current_pos = self.start_pos
        self.visited_positions: Set[Tuple[int, int]] = {self.start_pos}
        self.steps_taken = 0
        self.done = False

    @property
    def num_states(self) -> int:
        return self.width * self.height

    @property
    def num_actions(self) -> int:
        return 4

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.width + pos[1]

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        return (state // self.width, state % self.width)

    def reset(self) -> Tuple[int, int]:
        self.current_pos = self.start_pos
        self.visited_positions = {self.start_pos}
        self.steps_taken = 0
        self.done = False
        return self.current_pos

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_terminal_pos(self, pos: Tuple[int, int]) -> bool:
        return pos == self.goal_pos or pos in self.traps

    def _is_terminal_pos(self, pos: Tuple[int, int]) -> bool:
        return self.is_terminal_pos(pos)

    def simulate_step(self, pos: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Simulate step transition for world model trajectory rollout evaluations.
        """
        dr, dc = ACTION_VECTORS[action]
        target_pos = (pos[0] + dr, pos[1] + dc)
        
        if not self.is_valid(target_pos) or target_pos in self.obstacles:
            next_pos = pos
            reward = self.obstacle_penalty
            done = False
        elif target_pos == self.goal_pos:
            next_pos = target_pos
            reward = self.goal_reward
            done = True
        elif target_pos in self.traps:
            next_pos = target_pos
            reward = self.trap_penalty
            done = True
        else:
            next_pos = target_pos
            reward = self.step_penalty
            done = False

        return next_pos, reward, done

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        if self.done:
            return self.current_pos, 0.0, True, {"msg": "Episode already completed"}

        actual_action = action
        if self.slip_prob > 0.0 and np.random.rand() < self.slip_prob:
            actual_action = np.random.choice([a for a in ACTIONS if a != action])

        next_pos, reward, done = self.simulate_step(self.current_pos, actual_action)
        
        # Check revisit penalty (only for valid moves to a distinct previously visited position)
        if next_pos in self.visited_positions and next_pos != self.goal_pos and next_pos not in self.obstacles and next_pos != self.current_pos:
            reward += self.revisit_penalty
            
        self.visited_positions.add(next_pos)
        self.current_pos = next_pos
        self.steps_taken += 1

        if self.steps_taken >= self.max_steps and not done:
            done = True
            
        self.done = done
        info = {
            "steps": self.steps_taken,
            "actual_action": actual_action,
            "reached_goal": (self.current_pos == self.goal_pos)
        }
        
        return self.current_pos, reward, done, info
