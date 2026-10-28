import numpy as np
import pytest
from src.environment.gridworld import GridWorld
from src.quantum.fixed_oracle import FixedQuantumWorldModel

def test_fixed_quantum_world_model_initialization():
    env = GridWorld(6, 6)
    qwm = FixedQuantumWorldModel(env=env, horizon=3)
    assert qwm.num_qubits == 6
    assert qwm.hilbert_dim == 64

def test_fixed_quantum_world_model_futures_distribution():
    env = GridWorld(6, 6)
    qwm = FixedQuantumWorldModel(env=env, horizon=2)
    dist = qwm.evaluate_futures_distribution(start_pos=(0, 0), shots=100)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 1.0) < 1e-4
