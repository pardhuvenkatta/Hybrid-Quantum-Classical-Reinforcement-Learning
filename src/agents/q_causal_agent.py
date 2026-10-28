import numpy as np
from typing import Any, Optional, Dict, List, Tuple
from src.quantum.quantum_agent import QuantumAgent

class QCausalAgent(QuantumAgent):
    """
    Architecture 4: Quantum Causal Decision Architecture.
    Classical causal model maintains state-action-reward graph -> candidate interventions do(a) ->
    QPU evaluates quantum Bayesian network causal inference -> AI selects intervention with best outcome.
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
        self.causal_weights = np.ones((self.num_states, self.num_actions))
        self.parameter_count = self.q_table.size + self.causal_weights.size
        self.total_qpu_calls = 0
        self.total_shots_used = 0
        self.causal_predictions: List[float] = []
        self.causal_outcomes: List[float] = []
        self.causal_intervention_accuracy = 0.0

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
        causal_values = np.zeros(self.num_actions)

        for a in range(self.num_actions):
            self.total_qpu_calls += 1
            self.total_shots_used += self.shots

            intervention_weight = self.causal_weights[state_idx, a]
            init_state = self.world_model.encode_state(state_idx)
            causal_phases = np.zeros(self.world_model.hilbert_dim)
            causal_phases[a % self.world_model.hilbert_dim] = np.pi * intervention_weight

            qbn_state = self.world_model.apply_phase_oracle(init_state, causal_phases)
            qbn_state = self.world_model.apply_diffusion(qbn_state)
            probs = self.world_model.measure_probabilities(qbn_state, shots=self.shots)

            if self.noise_model is not None and hasattr(self.noise_model, "apply_to_probabilities"):
                probs = self.noise_model.apply_to_probabilities(probs)

            q_val = self.q_table[state_idx, a]
            quantum_causal_return = np.sum(probs * np.linspace(0.0, 1.0, len(probs))) * intervention_weight + q_val
            causal_values[a] = quantum_causal_return

        best_a = int(np.argmax(causal_values)) % self.num_actions
        self.causal_predictions.append(float(causal_values[best_a]))
        return best_a

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        super().update(state, action, reward, next_state, done)
        state_idx = self._state_to_idx(state)
        if reward > 0:
            self.causal_weights[state_idx, action % self.num_actions] *= 1.05
        else:
            self.causal_weights[state_idx, action % self.num_actions] *= 0.95
        self.causal_weights[state_idx] = np.clip(self.causal_weights[state_idx], 0.1, 2.0)

        self.causal_outcomes.append(reward)
        if len(self.causal_predictions) > 1 and len(self.causal_outcomes) > 1:
            preds = np.array(self.causal_predictions[-100:])
            outs = np.array(self.causal_outcomes[-100:])
            if len(preds) == len(outs):
                corr = np.corrcoef(preds, outs)[0, 1]
                self.causal_intervention_accuracy = float(np.nan_to_num(corr, nan=0.0))

    def on_episode_end(self, total_reward: float):
        pass
