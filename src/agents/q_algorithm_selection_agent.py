import numpy as np
from typing import Any, Optional, Dict, List, Tuple
from src.quantum.quantum_agent import QuantumAgent

class QAlgorithmSelectionAgent(QuantumAgent):
    """
    Architecture 7: Quantum Algorithm Selection Agent.
    Classical AI analyzes problem state -> Selector maps state to quantum algorithm primitive
    (Grover, VQC Continuous, Amplitude Amplification, Error-Mitigated Sampling) -> QPU executes -> Action.
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
        self.parameter_count = self.q_table.size + 32
        self.total_qpu_calls = 0
        self.total_shots_used = 0
        self.algorithm_distribution = {
            "Grover Search": 0,
            "VQC Optimization": 0,
            "Amplitude Amplification": 0,
            "Robust Sampling": 0
        }

    def _state_to_idx(self, state: Any) -> int:
        if hasattr(self.env, "pos_to_state") and isinstance(state, tuple):
            return self.env.pos_to_state(state)
        elif isinstance(state, int):
            return state % self.num_states
        elif isinstance(state, tuple):
            return (state[0] * 4 + state[1]) % self.num_states
        return 0

    def _select_algorithm(self, state_idx: int) -> str:
        visits = self.state_visit_counts.get(state_idx, 0)
        q_vals = self.q_table[state_idx]
        q_std = float(np.std(q_vals))

        if self.noise_model is not None and getattr(self.noise_model, "noise_level", 0.0) > 0.01:
            return "Robust Sampling"
        elif q_std > 0.5:
            return "Amplitude Amplification"
        elif visits < 5:
            return "Grover Search"
        else:
            return "VQC Optimization"

    def select_action(self, state: Any) -> int:
        state_idx = self._state_to_idx(state)
        self.state_visit_counts[state_idx] = self.state_visit_counts.get(state_idx, 0) + 1

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions) % self.num_actions

        algo = self._select_algorithm(state_idx)
        self.algorithm_distribution[algo] += 1
        self.total_qpu_calls += 1
        self.total_shots_used += self.shots

        init_state = self.world_model.encode_state(state_idx)

        if algo == "Grover Search":
            best_a = int(np.argmax(self.q_table[state_idx])) % self.world_model.hilbert_dim
            oracle_phases = np.zeros(self.world_model.hilbert_dim)
            oracle_phases[best_a] = np.pi
            state_vec = self.world_model.apply_phase_oracle(init_state, oracle_phases)
            state_vec = self.world_model.apply_diffusion(state_vec)
        elif algo == "Amplitude Amplification":
            phases = np.where(self.q_table[state_idx] >= np.mean(self.q_table[state_idx]), np.pi, 0.0)
            padded_phases = np.zeros(self.world_model.hilbert_dim)
            padded_phases[:len(phases)] = phases
            state_vec = self.world_model.apply_phase_oracle(init_state, padded_phases)
            state_vec = self.world_model.apply_diffusion(state_vec)
            state_vec = self.world_model.apply_diffusion(state_vec)
        elif algo == "VQC Optimization":
            rotations = self.q_table[state_idx] * np.pi
            padded_rotations = np.zeros(self.world_model.hilbert_dim)
            padded_rotations[:len(rotations)] = rotations
            state_vec = self.world_model.apply_phase_oracle(init_state, padded_rotations)
        else:
            state_vec = init_state.copy()

        probs = self.world_model.measure_probabilities(state_vec, shots=self.shots)
        if self.noise_model is not None and hasattr(self.noise_model, "apply_to_probabilities"):
            probs = self.noise_model.apply_to_probabilities(probs)

        action_probs = np.zeros(self.num_actions)
        for idx, p in enumerate(probs):
            action_probs[idx % self.num_actions] += p
        action_probs /= np.sum(action_probs)

        return int(np.random.choice(self.num_actions, p=action_probs)) % self.num_actions

    def on_episode_end(self, total_reward: float):
        pass
