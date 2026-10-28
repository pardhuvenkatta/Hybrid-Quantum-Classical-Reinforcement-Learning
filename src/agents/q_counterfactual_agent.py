import numpy as np
from typing import Any, Optional, Dict, List, Tuple
from src.quantum.quantum_agent import QuantumAgent

class QCounterfactualAgent(QuantumAgent):
    """
    Architecture 1: Quantum Counterfactual Decision Engine.
    Classical AI generates N candidate actions -> QPU evaluates counterfactual outcomes
    for each action using a VQC state evolution -> Classical AI selects action with highest
    counterfactual value.
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
        self.parameter_count = self.q_table.size + self.world_model.hilbert_dim * 2
        self.total_qpu_calls = 0
        self.total_shots_used = 0
        self.counterfactual_predictions: List[float] = []
        self.actual_returns: List[float] = []
        self.counterfactual_accuracy = 0.0

    def _state_to_idx(self, state: Any) -> int:
        if hasattr(self.env, "pos_to_state") and isinstance(state, tuple):
            return self.env.pos_to_state(state)
        elif isinstance(state, int):
            return state % self.num_states
        elif isinstance(state, tuple):
            return (state[0] * 4 + state[1]) % self.num_states
        return 0

    def select_action(self, state: Any) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions) % self.num_actions

        state_idx = self._state_to_idx(state)
        counterfactual_values = np.zeros(self.num_actions)

        for a in range(self.num_actions):
            self.total_qpu_calls += 1
            self.total_shots_used += self.shots

            init_state = self.world_model.encode_state(state_idx)
            oracle_phases = np.zeros(self.world_model.hilbert_dim)
            oracle_phases[a % self.world_model.hilbert_dim] = np.pi
            cf_state = self.world_model.apply_phase_oracle(init_state, oracle_phases)
            cf_state = self.world_model.apply_diffusion(cf_state)
            probs = self.world_model.measure_probabilities(cf_state, shots=self.shots)

            if self.noise_model is not None and hasattr(self.noise_model, "apply_to_probabilities"):
                probs = self.noise_model.apply_to_probabilities(probs)

            expected_cf_val = np.sum(probs * np.linspace(-1.0, 1.0, len(probs))) + self.q_table[state_idx, a]
            counterfactual_values[a] = expected_cf_val

        best_a = int(np.argmax(counterfactual_values)) % self.num_actions
        self.counterfactual_predictions.append(float(counterfactual_values[best_a]))
        return best_a

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        super().update(state, action, reward, next_state, done)
        self.actual_returns.append(reward)
        if len(self.counterfactual_predictions) > 0 and len(self.actual_returns) > 0:
            pred_arr = np.array(self.counterfactual_predictions[-100:])
            act_arr = np.array(self.actual_returns[-100:])
            if len(pred_arr) == len(act_arr) and len(pred_arr) > 1:
                norm_pred = (pred_arr - np.mean(pred_arr)) / (np.std(pred_arr) + 1e-8)
                norm_act = (act_arr - np.mean(act_arr)) / (np.std(act_arr) + 1e-8)
                self.counterfactual_accuracy = float(np.mean(np.abs(norm_pred - norm_act) < 1.0))

    def on_episode_end(self, total_reward: float):
        pass
