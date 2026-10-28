import numpy as np
from typing import Tuple, List, Dict

class StateActionEncoder:
    """
    Formal Quantum Trajectory Encoding Scheme:
    For depth d with branching factor b=4 (4 actions: UP, RIGHT, DOWN, LEFT),
    n = 2 * d qubits represent the trajectory Hilbert space H_traj of dimension 2^n = 4^d.
    
    Basis State Mapping:
    |tau> = |a_1> (x) |a_2> (x) ... (x) |a_d>
    where each action a_i in {0,1,2,3} is encoded into 2 computational qubits:
    00: UP (0), 01: RIGHT (1), 10: DOWN (2), 11: LEFT (3).
    """
    def __init__(self, grid_width: int, grid_height: int, depth: int = 3, branching_factor: int = 4):
        self.width = grid_width
        self.height = grid_height
        self.depth = depth
        self.branching_factor = branching_factor
        self.bits_per_action = int(np.ceil(np.log2(branching_factor)))
        self.num_qubits = self.depth * self.bits_per_action
        self.hilbert_dim = 2**self.num_qubits

    def pos_to_state_embedding(self, pos: Tuple[int, int]) -> np.ndarray:
        """
        State Embedding |s_t>: Normalized angle vector for VQC parameter binding.
        """
        r, c = pos
        theta_r = np.pi * (r / max(1, self.height - 1))
        theta_c = np.pi * (c / max(1, self.width - 1))
        return np.array([theta_r, theta_c], dtype=float)

    def action_seq_to_bitstring(self, action_seq: Tuple[int, ...]) -> str:
        bits = []
        for a in action_seq:
            bits.append(format(a, f'0{self.bits_per_action}b'))
        return "".join(bits)

    def bitstring_to_action_seq(self, bitstr: str) -> Tuple[int, ...]:
        actions = []
        for i in range(0, len(bitstr), self.bits_per_action):
            chunk = bitstr[i:i+self.bits_per_action]
            actions.append(int(chunk, 2))
        return tuple(actions)

    def index_to_trajectory(self, index: int) -> Tuple[int, ...]:
        bitstr = format(index, f'0{self.num_qubits}b')
        return self.bitstring_to_action_seq(bitstr)
