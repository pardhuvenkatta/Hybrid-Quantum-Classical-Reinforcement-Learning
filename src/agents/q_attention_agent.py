import numpy as np
from typing import Any, Optional, Dict, List, Tuple
from src.quantum.quantum_agent import QuantumAgent

class QAttentionAgent(QuantumAgent):
    """
    Architecture 6: Quantum Attention / Action Selection Layer.
    Classical encoder compresses raw observation to 8 features -> Quantum Attention Layer (VQC)
    evaluates entangling feature inner products -> Measurement yields attention weights -> Action logits.
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
        self.feature_dim = 8
        self.w_enc = np.random.normal(0, 0.1, (self.num_states, self.feature_dim))
        self.w_head = np.random.normal(0, 0.1, (self.feature_dim, self.num_actions))
        self.parameter_count = self.w_enc.size + self.w_head.size + self.q_table.size
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

    def select_action(self, state: Any) -> int:
        state_idx = self._state_to_idx(state)

        features = self.w_enc[state_idx]
        norm_features = (features - np.mean(features)) / (np.std(features) + 1e-8)

        self.total_qpu_calls += 1
        self.total_shots_used += self.shots

        init_state = self.world_model.encode_state(state_idx)
        attn_phases = np.zeros(self.world_model.hilbert_dim)
        for i in range(self.world_model.hilbert_dim):
            attn_phases[i] = np.pi * np.sin(norm_features[i % self.feature_dim])

        state_vec = self.world_model.apply_phase_oracle(init_state, attn_phases)
        state_vec = self.world_model.apply_diffusion(state_vec)
        probs = self.world_model.measure_probabilities(state_vec, shots=self.shots)

        if self.noise_model is not None and hasattr(self.noise_model, "apply_to_probabilities"):
            probs = self.noise_model.apply_to_probabilities(probs)

        attn_weights = probs[:self.feature_dim] / (np.sum(probs[:self.feature_dim]) + 1e-8)
        weighted_features = norm_features * attn_weights
        logits = weighted_features @ self.w_head + self.q_table[state_idx]
        action_probs = np.exp(logits - np.max(logits))
        action_probs /= np.sum(action_probs)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions) % self.num_actions

        return int(np.random.choice(self.num_actions, p=action_probs)) % self.num_actions

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        super().update(state, action, reward, next_state, done)
        state_idx = self._state_to_idx(state)
        feat = self.w_enc[state_idx]
        self.w_enc[state_idx] += self.lr * reward * np.sign(feat)
        self.w_head[:, action % self.num_actions] += self.lr * reward * feat

    def on_episode_end(self, total_reward: float):
        pass
