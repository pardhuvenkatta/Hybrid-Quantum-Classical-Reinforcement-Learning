import pytest
import numpy as np
from src.environment.gridworld import GridWorld
from src.quantum.quantum_encoding import StateActionEncoder
from src.quantum.quantum_world_model import QuantumWorldModel
from src.quantum.quantum_agent import QuantumAgent

def test_quantum_encoding():
    encoder = StateActionEncoder(grid_width=5, grid_height=5, depth=2)
    assert encoder.num_qubits == 4
    
    bitstr = encoder.action_seq_to_bitstring((0, 1)) # UP, RIGHT
    assert bitstr == "0001"
    
    seq = encoder.bitstring_to_action_seq("0001")
    assert seq == (0, 1)

def test_quantum_world_model():
    env = GridWorld(width=5, height=5)
    qwm = QuantumWorldModel(env=env, horizon=2)
    assert qwm.num_qubits == 4

    trajectory_probs = qwm.evaluate_futures_distribution(start_pos=(0, 0), shots=64)
    assert len(trajectory_probs) == 16
    total_p = sum(trajectory_probs.values())
    assert pytest.approx(total_p, abs=1e-3) == 1.0

def test_quantum_agent_action_selection():
    env = GridWorld(width=5, height=5)
    agent = QuantumAgent(env=env, horizon=2, shots=64)
    action = agent.select_action((0, 0))
    assert action in [0, 1, 2, 3]

    agent.update((0, 0), action, -0.02, (0, 1), False)
    assert agent.q_table.shape == (25, 4)

def test_quantum_continuous_oracle():
    env = GridWorld(width=5, height=5)
    qwm = QuantumWorldModel(env=env, horizon=2, oracle_mode="continuous", beta=1.0)
    assert qwm.oracle_mode == "continuous"
    trajectory_probs = qwm.evaluate_futures_distribution(start_pos=(0, 0), shots=64)
    assert len(trajectory_probs) == 16
    total_p = sum(trajectory_probs.values())
    assert pytest.approx(total_p, abs=1e-3) == 1.0

    agent = QuantumAgent(env=env, horizon=2, shots=64, oracle_mode="continuous", beta=1.0)
    act = agent.select_action((0, 0))
    assert act in [0, 1, 2, 3]
