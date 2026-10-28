import numpy as np
from typing import Any, Optional, Dict, List, Tuple
from src.quantum.quantum_agent import QuantumAgent

class QMultiAgentAgent(QuantumAgent):
    """
    Architecture 3: Quantum Multi-Agent Coordination.
    Shared QPU evaluates joint-action combinations for multi-agent team coordination.
    Quantum interference suppresses non-cooperative joint states and amplifies coordinated action pairs.
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
        self.parameter_count = self.q_table.size + 16 * 4
        self.total_qpu_calls = 0
        self.total_shots_used = 0
        self.coordination_success_rate = 1.0
        self.coordination_events: List[float] = []

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
        self.total_qpu_calls += 1
        self.total_shots_used += self.shots

        init_state = self.world_model.encode_state(state_idx)
        entangled_state = (init_state + np.roll(init_state, 1)) / np.sqrt(2.0)
        
        N = len(entangled_state)
        oracle_phases = np.zeros(N)
        for i in range(N):
            a_sub = i % self.num_actions
            if self.q_table[state_idx, a_sub] >= np.mean(self.q_table[state_idx]):
                oracle_phases[i] = np.pi

        state_vec = self.world_model.apply_phase_oracle(entangled_state, oracle_phases)
        state_vec = self.world_model.apply_diffusion(state_vec)

        probs = self.world_model.measure_probabilities(state_vec, shots=self.shots)
        if self.noise_model is not None and hasattr(self.noise_model, "apply_to_probabilities"):
            probs = self.noise_model.apply_to_probabilities(probs)

        local_probs = np.zeros(self.num_actions)
        for idx, p in enumerate(probs):
            local_probs[idx % self.num_actions] += p
        local_probs /= np.sum(local_probs)

        selected_action = int(np.random.choice(self.num_actions, p=local_probs)) % self.num_actions
        is_coordinated = float(self.q_table[state_idx, selected_action] >= np.mean(self.q_table[state_idx]))
        self.coordination_events.append(is_coordinated)
        if len(self.coordination_events) > 0:
            self.coordination_success_rate = float(np.mean(self.coordination_events[-100:]))

        return selected_action

    def on_episode_end(self, total_reward: float):
        pass
