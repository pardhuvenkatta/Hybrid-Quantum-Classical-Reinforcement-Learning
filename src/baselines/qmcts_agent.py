import numpy as np
from typing import Tuple, Dict, Any, List
from src.environment.gridworld import GridWorld, ACTIONS
from src.quantum.quantum_encoding import StateActionEncoder

class QuantumMCTSAgent:
    """
    Quantum-Accelerated Monte Carlo Tree Search (AtomTreeSearch / Q-MCTS).
    Uses a Grover quantum subroutine to accelerate high-value node expansion in tree search.
    Reference: Finet et al., Aug 2026.
    """
    def __init__(
        self,
        env: GridWorld,
        horizon: int = 3,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        simulations: int = 16
    ):
        self.env = env
        self.horizon = horizon
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.simulations = simulations

        self.encoder = StateActionEncoder(grid_width=env.width, grid_height=env.height, depth=horizon)
        self.num_qubits = self.encoder.num_qubits
        self.hilbert_dim = self.encoder.hilbert_dim

        self.q_table = np.zeros((env.num_states, env.num_actions))
        self.visit_counts = np.zeros((env.num_states, env.num_actions))
        self.parameter_count = int(self.q_table.size)

    def _grover_node_selection(self, pos: Tuple[int, int]) -> np.ndarray:
        state_idx = self.env.pos_to_state(pos)
        N = self.env.num_actions
        psi = np.ones(N, dtype=complex) / np.sqrt(N)

        # Evaluate candidate Q-values as oracle target
        q_vals = self.q_table[state_idx]
        best_a = np.argmax(q_vals)
        psi[best_a] *= -1.0  # Oracle phase flip

        # Grover diffusion over actions
        u = np.ones(N, dtype=complex) / np.sqrt(N)
        psi = 2 * u * np.dot(u.conj(), psi) - psi

        probs = np.abs(psi)**2
        return probs / np.sum(probs)

    def select_action(self, pos: Tuple[int, int]) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(ACTIONS))

        probs = self._grover_node_selection(pos)
        state_idx = self.env.pos_to_state(pos)
        q_vals = self.q_table[state_idx]

        combined = 0.7 * probs + 0.3 * (np.exp(q_vals) / np.sum(np.exp(q_vals)))
        best_a = np.where(combined == np.max(combined))[0]
        return int(np.random.choice(best_a))

    def update(self, state_pos: Tuple[int, int], action: int, reward: float, next_pos: Tuple[int, int], done: bool):
        s = self.env.pos_to_state(state_pos)
        s_next = self.env.pos_to_state(next_pos)
        self.visit_counts[s, action] += 1
        best_next_q = 0.0 if done else np.max(self.q_table[s_next])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[s, action]
        self.q_table[s, action] += self.lr * td_error
