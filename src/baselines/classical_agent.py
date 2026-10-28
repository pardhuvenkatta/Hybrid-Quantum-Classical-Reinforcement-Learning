import numpy as np
from typing import Tuple, Dict, Any, Optional
from src.environment.gridworld import GridWorld, ACTIONS
from src.baselines.classical_world_model import ClassicalWorldModel

class ClassicalAgent:
    """
    Model-based Classical RL Agent evaluating horizon-H trajectory rollouts.
    Matches the trajectory search space 4^H with the Quantum World Model.
    """
    def __init__(
        self,
        env: GridWorld,
        horizon: int = 3,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        eval_budget: Optional[int] = None
    ):
        self.env = env
        self.horizon = horizon
        self.world_model = ClassicalWorldModel(env=env, horizon=horizon, gamma=gamma)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eval_budget = eval_budget if eval_budget is not None else 4**horizon

        # Q-table: num_states x num_actions
        self.q_table = np.zeros((env.num_states, env.num_actions))
        self.parameter_count = int(self.q_table.size)

    def get_action_values_from_world_model(
        self, current_pos: Tuple[int, int]
    ) -> np.ndarray:
        trajectory_scores = self.world_model.evaluate_futures(
            start_pos=current_pos, max_evals=self.eval_budget
        )
        
        num_act = self.env.num_actions
        action_scores = {a: [] for a in range(num_act)}
        for seq, score in trajectory_scores.items():
            first_action = seq[0] % num_act
            action_scores[first_action].append(score)

        action_values = np.zeros(num_act)
        for a in range(num_act):
            if len(action_scores[a]) > 0:
                action_values[a] = np.max(action_scores[a])
            else:
                s_idx = self.env.pos_to_state(current_pos)
                action_values[a] = self.q_table[s_idx, a] if s_idx < len(self.q_table) else 0.0

        return action_values

    def select_action(self, current_pos: Tuple[int, int]) -> int:
        num_act = self.env.num_actions
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(range(num_act)))

        state_idx = self.env.pos_to_state(current_pos)
        model_action_values = self.get_action_values_from_world_model(current_pos)
        
        q_vals = self.q_table[state_idx][:num_act] if state_idx < len(self.q_table) else np.zeros(num_act)
        combined_values = 0.8 * model_action_values + 0.2 * q_vals
        
        max_val = np.max(combined_values)
        best_actions = np.where(combined_values == max_val)[0]
        return int(np.random.choice(best_actions))

    def update(
        self,
        state_pos: Tuple[int, int],
        action: int,
        reward: float,
        next_pos: Tuple[int, int],
        done: bool
    ):
        s = self.env.pos_to_state(state_pos)
        s_next = self.env.pos_to_state(next_pos)

        best_next_q = 0.0 if done else np.max(self.q_table[s_next])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[s, action]
        self.q_table[s, action] += self.lr * td_error
