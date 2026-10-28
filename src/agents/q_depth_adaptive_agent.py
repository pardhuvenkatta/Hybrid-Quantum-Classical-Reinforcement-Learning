import numpy as np
from typing import Any, Optional, Dict, List, Tuple
from src.quantum.quantum_agent import QuantumAgent

class QDepthAdaptiveAgent(QuantumAgent):
    """
    Architecture 2: Adaptive Quantum Circuit Complexity.
    Classical AI estimates decision difficulty -> maps to circuit depth (Shallow: 4 layers/4 qubits,
    Medium: 8 layers/6 qubits, Deep: 12 layers/8 qubits) -> QPU executes with chosen depth -> Action.
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
        self.state_visit_counts: Dict[int, int] = {}
        self.parameter_count = self.q_table.size + 12 * 8  # VQC parameters for deep config
        self.total_qpu_calls = 0
        self.total_shots_used = 0
        self.depth_history: List[int] = []
        self.depth_distribution = {"Shallow (L=4)": 0, "Medium (L=8)": 0, "Deep (L=12)": 0}

    def _state_to_idx(self, state: Any) -> int:
        if hasattr(self.env, "pos_to_state") and isinstance(state, tuple):
            return self.env.pos_to_state(state)
        elif isinstance(state, int):
            return state % self.num_states
        elif isinstance(state, tuple):
            return (state[0] * 4 + state[1]) % self.num_states
        return 0

    def _estimate_difficulty(self, state_idx: int) -> float:
        visits = self.state_visit_counts.get(state_idx, 0)
        novelty = 1.0 / np.sqrt(visits + 1)
        q_vals = self.q_table[state_idx]
        sorted_q = np.sort(q_vals)[::-1]
        margin = (sorted_q[0] - sorted_q[1]) / (abs(sorted_q[0]) + 1e-5) if len(sorted_q) > 1 else 1.0
        difficulty = 0.6 * novelty + 0.4 * (1.0 - margin)
        return float(np.clip(difficulty, 0.0, 1.0))

    def select_action(self, state: Any) -> int:
        state_idx = self._state_to_idx(state)
        self.state_visit_counts[state_idx] = self.state_visit_counts.get(state_idx, 0) + 1

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions) % self.num_actions

        difficulty = self._estimate_difficulty(state_idx)
        if difficulty < 0.35:
            depth = 4
            num_qubits = 4
            shots = 64
            self.depth_distribution["Shallow (L=4)"] += 1
        elif difficulty < 0.70:
            depth = 8
            num_qubits = 6
            shots = 128
            self.depth_distribution["Medium (L=8)"] += 1
        else:
            depth = 12
            num_qubits = 8
            shots = 256
            self.depth_distribution["Deep (L=12)"] += 1

        self.depth_history.append(depth)
        self.total_qpu_calls += 1
        self.total_shots_used += shots

        init_state = self.world_model.encode_state(state_idx)
        state_vec = init_state
        for _ in range(depth // 2):
            phases = np.random.uniform(0, 2*np.pi, len(state_vec))
            state_vec = self.world_model.apply_phase_oracle(state_vec, phases)
            state_vec = self.world_model.apply_diffusion(state_vec)

        probs = self.world_model.measure_probabilities(state_vec, shots=shots)
        if self.noise_model is not None and hasattr(self.noise_model, "apply_to_probabilities"):
            probs = self.noise_model.apply_to_probabilities(probs)

        q_bias = np.exp(self.q_table[state_idx] - np.max(self.q_table[state_idx]))
        q_bias /= np.sum(q_bias)

        action_probs = np.zeros(self.num_actions)
        for i, p in enumerate(probs):
            action_probs[i % self.num_actions] += p
        action_probs = 0.5 * action_probs + 0.5 * q_bias
        action_probs /= np.sum(action_probs)

        return int(np.random.choice(self.num_actions, p=action_probs)) % self.num_actions

    def on_episode_end(self, total_reward: float):
        pass
