import numpy as np
from typing import Any, Optional, Dict, List, Tuple
from src.quantum.quantum_agent import QuantumAgent

class QPlannerClassicalLearnerAgent(QuantumAgent):
    """
    Architecture 5: Quantum Dynamic Planner + Classical Learner.
    Clean separation of concerns:
    - QPU acts purely as a non-parametric trajectory planner/evaluator.
    - Classical AI acts purely as a parametric policy learner (softmax policy pi_theta).
    """
    def __init__(
        self,
        env: Any,
        horizon: int = 3,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        shots: int = 128,
        noise_model: Optional[Any] = None
    ):
        super().__init__(
            env=env,
            horizon=horizon,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            shots=shots
        )
        self.noise_model = noise_model
        self.num_states = getattr(env, "num_states", 16)
        self.num_actions = getattr(env, "num_actions", getattr(getattr(env, "action_space", None), "n", 4))
        self.policy_theta = np.zeros((self.num_states, self.num_actions))
        self.parameter_count = self.policy_theta.size
        self.total_qpu_calls = 0
        self.total_shots_used = 0

    def _state_to_idx(self, state: Any) -> int:
        if hasattr(self.env, "pos_to_state") and isinstance(state, tuple):
            return self.env.pos_to_state(state)
        elif isinstance(state, int):
            return state % self.num_states
        elif isinstance(state, tuple):
            return (state[0] * 4 + state[1]) % self.num_states
        return 0

    def _get_classical_policy_probs(self, state_idx: int) -> np.ndarray:
        logits = self.policy_theta[state_idx]
        exp_l = np.exp(logits - np.max(logits))
        return exp_l / np.sum(exp_l)

    def select_action(self, state: Any) -> int:
        state_idx = self._state_to_idx(state)

        self.total_qpu_calls += 1
        self.total_shots_used += self.shots

        init_state = self.world_model.encode_state(state_idx)
        oracle_phases = np.zeros(self.world_model.hilbert_dim)
        for a in range(self.num_actions):
            oracle_phases[a % self.world_model.hilbert_dim] = np.pi * (self.policy_theta[state_idx, a] > 0)

        qpu_state = self.world_model.apply_phase_oracle(init_state, oracle_phases)
        qpu_state = self.world_model.apply_diffusion(qpu_state)
        qpu_probs = self.world_model.measure_probabilities(qpu_state, shots=self.shots)

        if self.noise_model is not None and hasattr(self.noise_model, "apply_to_probabilities"):
            qpu_probs = self.noise_model.apply_to_probabilities(qpu_probs)

        qpu_values = np.zeros(self.num_actions)
        for idx, p in enumerate(qpu_probs):
            qpu_values[idx % self.num_actions] += p

        policy_probs = self._get_classical_policy_probs(state_idx)
        combined_probs = 0.6 * policy_probs + 0.4 * (qpu_values / (np.sum(qpu_values) + 1e-8))
        combined_probs /= np.sum(combined_probs)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions) % self.num_actions

        return int(np.random.choice(self.num_actions, p=combined_probs)) % self.num_actions

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        super().update(state, action, reward, next_state, done)
        state_idx = self._state_to_idx(state)
        probs = self._get_classical_policy_probs(state_idx)
        one_hot = np.zeros(self.num_actions)
        one_hot[action % self.num_actions] = 1.0
        grad = one_hot - probs
        self.policy_theta[state_idx] += self.lr * reward * grad

    def on_episode_end(self, total_reward: float):
        pass
