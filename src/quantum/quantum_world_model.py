import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from src.environment.gridworld import GridWorld, ACTIONS
from src.quantum.quantum_encoding import StateActionEncoder

try:
    import qiskit
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

class QuantumWorldModel:
    """
    Quantum World Model for Trajectory Generation & Evaluation Distribution Estimation.
    Uses exact Phase Marking (e^{i pi} = -1) on high-value trajectory states and Grover Diffusion
    for Quantum Amplitude Amplification over candidate future trajectories.
    """
    def __init__(
        self,
        env: GridWorld,
        horizon: int = 3,
        gamma: float = 0.95,
        percentile_threshold: float = 75.0,
        enable_oracle: bool = True,
        enable_interference: bool = True,
        oracle_mode: str = "exact",
        beta: float = 1.0
    ):
        self.env = env
        self.horizon = horizon
        self.gamma = gamma
        self.percentile_threshold = percentile_threshold
        self.enable_oracle = enable_oracle
        self.enable_interference = enable_interference
        self.oracle_mode = oracle_mode
        self.beta = beta
        
        self.encoder = StateActionEncoder(
            grid_width=getattr(env, "width", 8),
            grid_height=getattr(env, "height", 8),
            depth=horizon
        )
        self.num_qubits = self.encoder.num_qubits
        self.hilbert_dim = self.encoder.hilbert_dim
        self.total_circuit_evaluations = 0
        self.last_circuit_depth = 0
        self.weights = np.random.randn(16) * 0.1

    def encode_state(self, state_idx: int) -> np.ndarray:
        """Initialize state vector in uniform superposition with state-specific phase shift."""
        psi = np.ones(self.hilbert_dim, dtype=complex) / np.sqrt(self.hilbert_dim)
        phase_shift = np.exp(1j * (state_idx % self.hilbert_dim) * 2 * np.pi / self.hilbert_dim)
        return psi * phase_shift

    def apply_phase_oracle(self, psi: np.ndarray, phases: np.ndarray) -> np.ndarray:
        """Apply diagonal phase oracle U_phase = diag(e^{i * phases})."""
        padded_phases = np.zeros(self.hilbert_dim)
        padded_phases[:min(len(phases), self.hilbert_dim)] = phases[:min(len(phases), self.hilbert_dim)]
        return psi * np.exp(1j * padded_phases)

    def apply_diffusion(self, psi: np.ndarray) -> np.ndarray:
        """Apply Grover diffusion operator D = 2|u><u| - I."""
        u = np.ones(self.hilbert_dim, dtype=complex) / np.sqrt(self.hilbert_dim)
        return 2 * u * np.dot(u.conj(), psi) - psi

    def measure_probabilities(self, psi: np.ndarray, shots: int = 128) -> np.ndarray:
        """Compute computational basis measurement probabilities P(i) = |psi_i|^2 with sampling variance."""
        probs = np.abs(psi)**2
        probs = probs / np.sum(probs)
        counts = np.random.multinomial(shots, probs)
        return counts / float(shots)

    def get_top_marked_indices(self, start_pos: Tuple[int, int], num_branches: Optional[int] = None) -> List[int]:
        N_states = self.hilbert_dim
        values = np.zeros(N_states)
        for idx in range(N_states):
            seq = self.encoder.index_to_trajectory(idx)
            values[idx] = self.evaluate_classical_trajectory_value(start_pos, seq)
        v_min, v_max = np.min(values), np.max(values)
        if v_max > v_min:
            if num_branches is not None:
                k = min(max(1, num_branches), N_states - 1)
            else:
                k = max(1, N_states // 8)
            top_indices = np.argsort(values)[-k:].tolist()
            return top_indices
        return []

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

        gr, gc = getattr(self.env, "goal_pos", (0, 0))
        pr, pc = current_p if isinstance(current_p, tuple) and len(current_p) == 2 else (0, 0)
        dist = abs(gr - pr) + abs(gc - pc)
        max_dist = getattr(self.env, "width", 8) + getattr(self.env, "height", 8)
        
        goal_p = getattr(self.env, "goal_pos", None)
        obstacles = getattr(self.env, "obstacles", [])
        traps = getattr(self.env, "traps", [])
        
        if current_p == goal_p:
            heuristic = getattr(self.env, "goal_reward", 1.0)
        elif current_p in obstacles:
            heuristic = getattr(self.env, "obstacle_penalty", -0.5)
        elif current_p in traps:
            heuristic = getattr(self.env, "trap_penalty", -1.0)
        else:
            heuristic = - (dist / max_dist)

        total_return += discount * heuristic
        return total_return

    def evaluate_futures_distribution(
        self,
        start_pos: Tuple[int, int],
        shots: int = 128,
        num_branches: Optional[int] = None,
        circuit_depth: int = 1,
        enable_interference: Optional[bool] = None,
        noise_model: Optional[Any] = None
    ) -> Dict[Tuple[int, ...], float]:
        self.total_circuit_evaluations += 1
        N_states = self.hilbert_dim
        use_interference = enable_interference if enable_interference is not None else self.enable_interference
        
        psi = np.ones(N_states, dtype=complex) / np.sqrt(N_states)

        values = np.zeros(N_states)
        for idx in range(N_states):
            seq = self.encoder.index_to_trajectory(idx)
            values[idx] = self.evaluate_classical_trajectory_value(start_pos, seq)

        depth_iterations = max(1, circuit_depth)
        for _ in range(depth_iterations):
            if self.enable_oracle:
                if self.oracle_mode == "exact":
                    top_indices = self.get_top_marked_indices(start_pos, num_branches=num_branches)
                    if len(top_indices) > 0:
                        psi[top_indices] *= -1.0
                elif self.oracle_mode == "continuous":
                    v_min, v_max = np.min(values), np.max(values)
                    if v_max > v_min:
                        norm_v = (values - v_min) / (v_max - v_min)
                    else:
                        norm_v = np.zeros_like(values)
                    oracle_phases = self.beta * np.pi * norm_v
                    psi = psi * np.exp(1j * oracle_phases)

            if use_interference:
                u = np.ones(N_states, dtype=complex) / np.sqrt(N_states)
                psi = 2 * u * np.dot(u.conj(), psi) - psi

            if noise_model is not None and hasattr(noise_model, 'apply_statevector_noise'):
                psi = noise_model.apply_statevector_noise(psi)

        probs = np.abs(psi)**2
        probs = probs / np.sum(probs)

        counts = np.random.multinomial(shots, probs)

        if noise_model is not None and hasattr(noise_model, 'apply_measurement_noise'):
            counts = noise_model.apply_measurement_noise(counts, shots)

        trajectory_probs = {}
        for idx, count in enumerate(counts):
            seq = self.encoder.index_to_trajectory(idx)
            trajectory_probs[seq] = count / shots

        return trajectory_probs
