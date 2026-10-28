import numpy as np
from typing import Tuple, Dict, Any
from src.environment.gridworld import GridWorld, ACTIONS
from src.quantum.quantum_world_model import QuantumWorldModel

class ZengActionSearchAgent:
    """
    Implementation of Zeng et al. (2023, 2024) Grover Action Search Collapse Baseline.
    Uses VQC + Grover Oracle, but collapses the trajectory measurement to the single
    most probable immediate action a*, without keeping full trajectory distribution features.
    """
    def __init__(
        self,
        env: GridWorld,
        horizon: int = 3,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        shots: int = 128,
        eval_budget: int = 64
    ):
        self.env = env
        self.horizon = horizon
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.qwm = QuantumWorldModel(env=env, horizon=horizon, gamma=gamma, shots=shots)
        self.q_table = np.zeros((env.num_states, env.num_actions))

    def select_action(self, current_pos: Tuple[int, int]) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(ACTIONS))

        state_idx = self.env.pos_to_state(current_pos)
        
        # QWM Grover measurement distribution P(tau)
        traj_probs, top_seqs = self.qwm.evaluate_futures_quantum(current_pos)

        # Zeng et al. collapse: pick top measured trajectory state and collapse to first action a*
        best_seq = max(traj_probs.keys(), key=lambda seq: traj_probs[seq])
        collapsed_action = best_seq[0]

        # Combine with Q-table
        q_val = self.q_table[state_idx, collapsed_action]
        
        # Select action matching collapsed single-action Grover decision
        return int(collapsed_action)

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

        # Update QWM parameters
        self.qwm.train_on_experience(state_pos, action, reward, next_pos, done)
