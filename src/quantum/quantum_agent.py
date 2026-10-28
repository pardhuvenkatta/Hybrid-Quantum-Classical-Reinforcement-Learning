import numpy as np
from typing import Tuple, Dict, Any, Optional
from src.environment.gridworld import GridWorld, ACTIONS
from src.quantum.quantum_world_model import QuantumWorldModel

class QuantumAgent:
    """
    Hybrid Quantum-Classical AI Agent.
    Interfaces Quantum World Model measurement distribution P(\tau) into policy decision updates.
    """
    def __init__(
        self,
        env: GridWorld,
        horizon: int = 3,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        shots: int = 128,
        enable_oracle: bool = True,
        enable_interference: bool = True,
        oracle_mode: str = "exact",
        beta: float = 1.0,
        noise_model: Optional[Any] = None
    ):
        self.env = env
        self.noise_model = noise_model
        self.world_model = QuantumWorldModel(
            env=env,
            horizon=horizon,
            gamma=gamma,
            enable_oracle=enable_oracle,
            enable_interference=enable_interference,
            oracle_mode=oracle_mode,
            beta=beta
        )
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.shots = shots

        # Policy Q-table prior: num_states x num_actions
        self.q_table = np.zeros((env.num_states, env.num_actions))
        circuit_depth = max(1, 2 * horizon)
        self.parameter_count = int(self.q_table.size + (self.world_model.num_qubits * circuit_depth))

    def get_action_probabilities_from_quantum_distribution(
        self, current_pos: Tuple[int, int]
    ) -> np.ndarray:
        trajectory_probs = self.world_model.evaluate_futures_distribution(
            start_pos=current_pos, shots=self.shots, noise_model=self.noise_model
        )

        num_act = self.env.num_actions
        action_probs = np.zeros(num_act)
        for seq, prob in trajectory_probs.items():
            first_action = seq[0] % num_act
            action_probs[first_action] += prob

        total_p = np.sum(action_probs)
        if total_p > 0:
            action_probs /= total_p
        else:
            action_probs = np.ones(self.env.num_actions) / self.env.num_actions

        return action_probs

    def select_action(self, current_pos: Tuple[int, int]) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(ACTIONS)

        state_idx = self.env.pos_to_state(current_pos)
        quantum_action_probs = self.get_action_probabilities_from_quantum_distribution(current_pos)

        # Softmax Q-table prior
        q_vals = self.q_table[state_idx]
        exp_q = np.exp(q_vals - np.max(q_vals))
        q_probs = exp_q / np.sum(exp_q)

        # Combined policy distribution: 80% Quantum Model Distribution + 20% Q-prior
        combined_dist = 0.8 * quantum_action_probs + 0.2 * q_probs
        
        max_p = np.max(combined_dist)
        best_actions = np.where(combined_dist == max_p)[0]
        return np.random.choice(best_actions)

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
