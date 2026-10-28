import numpy as np
import math
from typing import Tuple, Dict, Any, List, Optional
from src.environment.gridworld import GridWorld, ACTIONS

class MCTSNode:
    def __init__(self, state_pos: Tuple[int, int], parent=None, action_from_parent: Optional[int] = None):
        self.state_pos = state_pos
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visits = 0
        self.total_reward = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(ACTIONS)

    def ucb1_score(self, action: int, c_param: float = 1.414) -> float:
        child = self.children.get(action)
        if child is None or child.visits == 0:
            return float('inf')
        return (child.total_reward / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)


class MCTSAgent:
    """
    Monte Carlo Tree Search (MCTS) with UCB1 exploration matching rollout budget 4^H.
    """
    def __init__(
        self,
        env: GridWorld,
        horizon: int = 3,
        c_puct: float = 1.414,
        eval_budget: Optional[int] = None,
        gamma: float = 0.95
    ):
        self.env = env
        self.horizon = horizon
        self.c_puct = c_puct
        self.eval_budget = eval_budget if eval_budget is not None else 4**horizon
        self.gamma = gamma

    def select_action(self, current_pos: Tuple[int, int]) -> int:
        root = MCTSNode(state_pos=current_pos)

        # Run simulations equal to eval_budget (4^H)
        for _ in range(self.eval_budget):
            node = root
            sim_pos = current_pos
            depth = 0

            # 1. Selection
            while node.is_fully_expanded() and depth < self.horizon:
                best_action = max(ACTIONS, key=lambda a: node.ucb1_score(a, c_param=self.c_puct))
                node = node.children[best_action]
                sim_pos = node.state_pos
                depth += 1

            # 2. Expansion
            if depth < self.horizon and not self.env._is_terminal_pos(sim_pos):
                untried_actions = [a for a in ACTIONS if a not in node.children]
                if untried_actions:
                    action = int(np.random.choice(untried_actions))
                    next_pos, r, d = self.env.simulate_step(sim_pos, action)
                    child_node = MCTSNode(state_pos=next_pos, parent=node, action_from_parent=action)
                    node.children[action] = child_node
                    node = child_node
                    sim_pos = next_pos
                    depth += 1

            # 3. Rollout / Simulation
            rollout_reward = 0.0
            curr_sim = sim_pos
            discount = 1.0
            for d in range(depth, self.horizon):
                if self.env._is_terminal_pos(curr_sim):
                    break
                rand_action = int(np.random.choice(ACTIONS))
                curr_sim, r, done = self.env.simulate_step(curr_sim, rand_action)
                rollout_reward += discount * r
                discount *= self.gamma

            # 4. Backpropagation
            curr_node = node
            while curr_node is not None:
                curr_node.visits += 1
                curr_node.total_reward += rollout_reward
                curr_node = curr_node.parent

        # Select action with highest visit count (robust choice)
        if not root.children:
            return int(np.random.choice(ACTIONS))

        best_a = max(root.children.keys(), key=lambda a: root.children[a].visits)
        return int(best_a)

    def update(self, state_pos: Tuple[int, int], action: int, reward: float, next_pos: Tuple[int, int], done: bool):
        pass  # MCTS plans online using environment rollouts
