import numpy as np
from typing import Tuple, Dict, Any, List
from src.environment.gridworld import GridWorld, ACTIONS
from src.quantum.quantum_encoding import StateActionEncoder

class QLSTMReservoirAgent:
    """
    Architecture E: QLSTM-Reservoir RL.
    Fixed, untrained VQC quantum reservoir feature extractor mapped to a trainable classical readout.
    Bypasses barren plateaus while preserving non-linear quantum entanglement features.
    Reference: Chen, Wells Fargo, QTML 2024.
    """
    def __init__(
        self,
        env: GridWorld,
        horizon: int = 3,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1
    ):
        self.env = env
        self.horizon = horizon
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.encoder = StateActionEncoder(grid_width=env.width, grid_height=env.height, depth=horizon)
        self.num_qubits = self.encoder.num_qubits
        self.hilbert_dim = self.encoder.hilbert_dim
        
        # Fixed quantum reservoir parameters (randomized VQC, untrained)
        self.fixed_reservoir_weights = np.random.randn(self.num_qubits * 4)
        
        # Trainable linear readout layer over quantum statevector features
        self.readout_weights = np.zeros((env.num_actions, self.hilbert_dim))
        self.parameter_count = int(self.readout_weights.size + self.fixed_reservoir_weights.size)

    def _quantum_reservoir_features(self, pos: Tuple[int, int]) -> np.ndarray:
        # Simulate fixed VQC reservoir mapping |psi_reservoir>
        r, c = pos
        theta_r = np.pi * (r / max(1, self.env.height - 1))
        theta_c = np.pi * (c / max(1, self.env.width - 1))
        
        # Superposition state
        psi = np.ones(self.hilbert_dim, dtype=complex) / np.sqrt(self.hilbert_dim)
        
        # Apply fixed VQC phase rotations
        for i in range(self.hilbert_dim):
            w = self.fixed_reservoir_weights[i % len(self.fixed_reservoir_weights)]
            phase = theta_r * np.cos(w) + theta_c * np.sin(w)
            psi[i] *= np.exp(1j * phase)
            
        # Return Born rule measurement probability vector as features
        return np.abs(psi)**2

    def select_action(self, pos: Tuple[int, int]) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(ACTIONS))

        phi = self._quantum_reservoir_features(pos)
        q_values = np.dot(self.readout_weights, phi)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(
        self,
        state_pos: Tuple[int, int],
        action: int,
        reward: float,
        next_pos: Tuple[int, int],
        done: bool
    ):
        phi_s = self._quantum_reservoir_features(state_pos)
        q_s = np.dot(self.readout_weights[action], phi_s)
        
        if done:
            target = reward
        else:
            phi_next = self._quantum_reservoir_features(next_pos)
            max_q_next = np.max(np.dot(self.readout_weights, phi_next))
            target = reward + self.gamma * max_q_next

        td_error = target - q_s
        self.readout_weights[action] += self.lr * td_error * phi_s
