import numpy as np
import itertools
from typing import Tuple, List, Dict, Any, Optional
from src.environment.gridworld import GridWorld, ACTIONS

class ClassicalWorldModel:
    """
    Classical World Model for evaluating multi-step future trajectories.
    Performs explicit tree rollouts up to horizon H matching the candidate trajectory space 4^H.
    """
    def __init__(self, env: GridWorld, horizon: int = 3, gamma: float = 0.95):
        self.env = env
        self.horizon = horizon
        self.gamma = gamma
        self.total_evaluations = 0

    def heuristic_value(self, pos: Tuple[int, int]) -> float:
        gr, gc = self.env.goal_pos
        pr, pc = pos
        dist = abs(gr - pr) + abs(gc - pc)
        if pos == self.env.goal_pos:
            return self.env.goal_reward
        if pos in self.env.obstacles:
            return self.env.obstacle_penalty
        if pos in self.env.traps:
            return self.env.trap_penalty
        max_dist = self.env.width + self.env.height
        return - (dist / max_dist)

    def generate_candidate_trajectories(self) -> List[Tuple[int, ...]]:
        return list(itertools.product(ACTIONS, repeat=self.horizon))

    def evaluate_trajectory(
        self, start_pos: Tuple[int, int], action_seq: Tuple[int, ...]
    ) -> float:
        self.total_evaluations += 1
        current_p = start_pos
        total_return = 0.0
        discount = 1.0

        for a in action_seq:
            next_p, reward, done = self.env.simulate_step(current_p, a)
            total_return += discount * reward
            discount *= self.gamma
            current_p = next_p
            if done:
                break

        if current_p != self.env.goal_pos:
            total_return += discount * self.heuristic_value(current_p)

        return total_return

    def evaluate_futures(
        self, start_pos: Tuple[int, int], max_evals: Optional[int] = None
    ) -> Dict[Tuple[int, ...], float]:
        candidates = self.generate_candidate_trajectories()
        if max_evals is not None and len(candidates) > max_evals:
            indices = np.random.choice(len(candidates), size=max_evals, replace=False)
            candidates = [candidates[i] for i in indices]

        results = {}
        for seq in candidates:
            ret = self.evaluate_trajectory(start_pos, seq)
            results[seq] = ret

        return results
