import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from src.environment.gridworld import GridWorld, ACTIONS
from src.quantum.quantum_encoding import StateActionEncoder

class FixedQuantumWorldModel:
    """
    Fixed Quantum World Model with exact Grover Phase Marking:
    Mark top trajectory returns with phase shift pi (e^{i pi} = -1),
    followed by Grover Diffusion D = 2|u><u| - I to achieve exact quantum amplitude amplification.
    """
    def __init__(
        self,
        env: GridWorld,
        horizon: int = 3,
        gamma: float = 0.95,
        percentile_threshold: float = 75.0,
        enable_oracle: bool = True,
        enable_interference: bool = True
    ):
        self.env = env
        self.horizon = horizon
        self.gamma = gamma
        self.percentile_threshold = percentile_threshold
        self.enable_oracle = enable_oracle
        self.enable_interference = enable_interference
        
        self.encoder = StateActionEncoder(grid_width=env.width, grid_height=env.height, depth=horizon)
        self.num_qubits = self.encoder.num_qubits
        self.hilbert_dim = self.encoder.hilbert_dim
        self.total_circuit_evaluations = 0

    def evaluate_classical_trajectory_value(
        self, start_pos: Tuple[int, int], action_seq: Tuple[int, ...]
    ) -> float:
        current_p = start_pos
        total_return = 0.0
        discount = 1.0

        for a in action_seq:
            next_p, reward, done = self.env.simulate_step(current_p, a)
            total_return += discount * reward
            discount *= self.gamma
            current_p = next_p
            if done:
                break

        gr, gc = self.env.goal_pos
        pr, pc = current_p
        dist = abs(gr - pr) + abs(gc - pc)
        max_dist = self.env.width + self.env.height
        
        if current_p == self.env.goal_pos:
            heuristic = self.env.goal_reward
        elif current_p in self.env.obstacles:
            heuristic = self.env.obstacle_penalty
        elif current_p in self.env.traps:
            heuristic = self.env.trap_penalty
        else:
            heuristic = - (dist / max_dist)

        total_return += discount * heuristic
        return total_return

    def evaluate_futures_distribution(
        self, start_pos: Tuple[int, int], shots: int = 128, noise_model: Optional[Any] = None
    ) -> Dict[Tuple[int, ...], float]:
        self.total_circuit_evaluations += 1
        N_states = self.hilbert_dim
        
        # 1. State vector initialization in uniform superposition: H^{\otimes n} |0>
        psi = np.ones(N_states, dtype=complex) / np.sqrt(N_states)

        # 2. Evaluate candidate trajectory values
        values = np.zeros(N_states)
        for idx in range(N_states):
            seq = self.encoder.index_to_trajectory(idx)
            values[idx] = self.evaluate_classical_trajectory_value(start_pos, seq)

        # 3. Quantum Phase Oracle U_world: Phase Flip e^{i pi} = -1 on top trajectories
        if self.enable_oracle:
            v_min, v_max = np.min(values), np.max(values)
            if v_max > v_min:
                thresh = np.percentile(values, self.percentile_threshold)
                oracle_phases = np.where(values >= thresh, np.pi, 0.0)
                psi = psi * np.exp(1j * oracle_phases)

        # 4. Grover Diffusion / Interference Operator: D = 2|u><u| - I
        if self.enable_interference:
            u = np.ones(N_states, dtype=complex) / np.sqrt(N_states)
            psi = 2 * u * np.dot(u.conj(), psi) - psi

        # 5. Measurement Probabilities P(\tau) = |\alpha_\tau|^2
        probs = np.abs(psi)**2
        probs = probs / np.sum(probs)

        counts = np.random.multinomial(shots, probs)

        trajectory_probs = {}
        for idx, count in enumerate(counts):
            seq = self.encoder.index_to_trajectory(idx)
            trajectory_probs[seq] = count / shots

        return trajectory_probs
