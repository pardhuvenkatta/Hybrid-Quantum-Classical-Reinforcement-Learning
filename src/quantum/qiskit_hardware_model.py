import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from typing import Tuple, Dict, Any, List
from src.environment.gridworld import GridWorld, ACTIONS
from src.quantum.quantum_encoding import StateActionEncoder
from src.quantum.quantum_world_model import QuantumWorldModel

class QiskitHardwareSimulator:
    """
    Qiskit Circuit Translator and NISQ Noisy Simulation Engine for QWM.
    Converts QWM numpy statevector state transitions into explicit Qiskit QuantumCircuits
    and runs them under Qiskit Aer with customizable depolarizing and readout noise models.
    """
    def __init__(
        self,
        env: GridWorld,
        horizon: int = 3,
        shots: int = 1024,
        depol_error: float = 0.0,
        readout_err: float = 0.0
    ):
        self.env = env
        self.horizon = horizon
        self.shots = shots
        self.num_qubits = 2 * horizon
        self.encoder = StateActionEncoder(grid_width=env.width, grid_height=env.height, depth=horizon)
        self.depol_error = depol_error
        self.readout_err = readout_err
        
        # Build noise model
        self.noise_model = NoiseModel()
        if depol_error > 0:
            error_1q = depolarizing_error(depol_error, 1)
            error_2q = depolarizing_error(depol_error * 2, 2)
            self.noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'ry', 'rz', 'x', 'z', 's'])
            self.noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
        if readout_err > 0:
            ro_error = ReadoutError([[1 - readout_err, readout_err], [readout_err, 1 - readout_err]])
            self.noise_model.add_all_qubit_readout_error(ro_error)

        if depol_error > 0 or readout_err > 0:
            self.simulator = AerSimulator(noise_model=self.noise_model)
        else:
            self.simulator = AerSimulator()

    def build_qiskit_circuit(self, qwm: QuantumWorldModel, start_pos: Tuple[int, int]) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # 1. Superposition H^n
        for q in range(self.num_qubits):
            qc.h(q)
            
        # 2. VQC Angle Embedding (RY, RZ, CZ)
        for h_step in range(self.horizon):
            q0 = 2 * h_step
            q1 = 2 * h_step + 1
            param_idx = (h_step * 4) % len(qwm.weights)
            qc.ry(qwm.weights[param_idx], q0)
            qc.rz(qwm.weights[(param_idx + 1) % len(qwm.weights)], q1)
            qc.cz(q0, q1)

        # 3. Grover Phase Marking Oracle & Diffusion
        top_indices = qwm.get_top_marked_indices(start_pos)
        if len(top_indices) > 0:
            for target_idx in top_indices:
                bitstr = format(target_idx, f'0{self.num_qubits}b')
                for q, bit in enumerate(bitstr):
                    if bit == '0':
                        qc.x(q)
                if self.num_qubits == 1:
                    qc.z(0)
                else:
                    qc.h(self.num_qubits - 1)
                    qc.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
                    qc.h(self.num_qubits - 1)
                for q, bit in enumerate(bitstr):
                    if bit == '0':
                        qc.x(q)

            # Grover Diffusion D = 2|u><u| - I
            for q in range(self.num_qubits):
                qc.h(q)
                qc.x(q)
            if self.num_qubits > 1:
                qc.h(self.num_qubits - 1)
                qc.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
                qc.h(self.num_qubits - 1)
            else:
                qc.z(0)
            for q in range(self.num_qubits):
                qc.x(q)
                qc.h(q)

        # 4. Measurement
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def execute_noisy_simulation(self, qwm: QuantumWorldModel, start_pos: Tuple[int, int]) -> Dict[Tuple[int, ...], float]:
        qc = self.build_qiskit_circuit(qwm, start_pos)
        transpiled_qc = qiskit.transpile(qc, self.simulator)
        result = self.simulator.run(transpiled_qc, shots=self.shots).result()
        counts = result.get_counts()

        # Convert counts to sequence probability map
        seq_probs = {}
        total_shots = sum(counts.values())
        for bitstr, count in counts.items():
            idx = int(bitstr, 2)
            seq = self.encoder.index_to_trajectory(idx)
            seq_probs[seq] = count / float(total_shots)
            
        return seq_probs
