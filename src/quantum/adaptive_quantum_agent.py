import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from src.environment.gridworld import GridWorld, ACTIONS
from src.quantum.quantum_agent import QuantumAgent
from src.quantum.uncertainty_estimator import UncertaintyEstimator
from src.quantum.adaptive_controller import AdaptiveQPUController

class AdaptiveQuantumAgent(QuantumAgent):
    """
    AI-Controlled Adaptive Quantum Agent.
    Dynamically allocates QPU resources (marked branches K, circuit depth d, shots M)
    based on prediction uncertainty estimates U(s) in [0, 1].
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
        noise_model: Optional[Any] = None
    ):
        super().__init__(
            env=env,
            horizon=horizon,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            shots=shots,
            enable_oracle=enable_oracle,
            enable_interference=enable_interference,
            noise_model=noise_model
        )
        self.noise_model = noise_model
        num_states = getattr(env, "num_states", 16)
        num_actions = getattr(env, "num_actions", 4)
        self.uncertainty_estimator = UncertaintyEstimator(
            num_states=num_states,
            num_actions=num_actions
        )
        self.controller = AdaptiveQPUController(hilbert_dim=self.world_model.hilbert_dim)
        self.parameter_count = self.q_table.size

        # Resource & Uncertainty Tracking Metrics
        self.total_qpu_calls = 0
        self.total_shots_used = 0
        self.total_circuit_layers = 0
        
        self.uncertainty_history: List[float] = []
        self.shots_history: List[int] = []
        self.depth_history: List[int] = []
        self.branches_marked_history: List[int] = []
        self.last_state_idx: Optional[int] = None

    def select_action(self, current_pos: Tuple[int, int]) -> int:
        state_idx = self.env.pos_to_state(current_pos)
        self.last_state_idx = state_idx

        # 1. Estimate state uncertainty U(s) in [0, 1]
        u = self.uncertainty_estimator.estimate_uncertainty(state_idx, self.q_table)

        # 2. Allocate quantum resources via AI controller
        res = self.controller.allocate_resources(u, hilbert_dim=self.world_model.hilbert_dim)

        # 3. Evaluate quantum world model distribution with dynamic resources
        trajectory_probs = self.world_model.evaluate_futures_distribution(
            start_pos=current_pos,
            shots=res["measurement_shots"],
            num_branches=res["num_branches"],
            circuit_depth=res["circuit_depth"],
            enable_interference=res["exploration_level"],
            noise_model=self.noise_model
        )

        # 4. Record execution tracking metrics
        self.total_qpu_calls += 1
        self.total_shots_used += res["measurement_shots"]
        self.total_circuit_layers += res["circuit_depth"]

        self.uncertainty_history.append(u)
        self.shots_history.append(res["measurement_shots"])
        self.depth_history.append(res["circuit_depth"])
        self.branches_marked_history.append(res["num_branches"])

        # 5. Policy decision combination: 80% Quantum Model Distribution + 20% Q-prior
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

        if np.random.rand() < self.epsilon:
            return np.random.choice(ACTIONS)

        # Softmax Q-table prior
        q_vals = self.q_table[state_idx]
        exp_q = np.exp(q_vals - np.max(q_vals))
        q_probs = exp_q / np.sum(exp_q)

        combined_dist = 0.8 * action_probs + 0.2 * q_probs
        max_p = np.max(combined_dist)
        best_actions = np.where(combined_dist == max_p)[0]
        return int(np.random.choice(best_actions))

    def on_episode_end(self, episode_return: float):
        """Update uncertainty estimator experience after episode completion."""
        if self.last_state_idx is not None:
            self.uncertainty_estimator.update_state_experience(self.last_state_idx, episode_return)
