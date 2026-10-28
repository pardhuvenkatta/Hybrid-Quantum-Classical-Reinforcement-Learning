import pytest
import numpy as np

from src.environments.grid_4x4 import Grid4x4Env
from src.agents.tabular_q_agent import TabularQAgent
from src.world_models.classical_transition_model import ClassicalTransitionModel
from src.world_models.quantum_transition_model import QuantumTransitionModel
from src.world_models.hybrid_transition_model import HybridTransitionModel
from src.planners.sampling_planner import SamplingPlanner

def test_grid_4x4_environment():
    env = Grid4x4Env(slip_prob=0.10)
    pos = env.reset(seed=42)
    assert pos == (0, 0)
    next_pos, reward, done, info = env.step(1)  # Move down
    assert isinstance(next_pos, tuple)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

def test_tabular_q_agent():
    agent = TabularQAgent()
    action = agent.select_action((0, 0))
    assert 0 <= action < 4
    agent.update((0, 0), action, -0.1, (1, 0), False)
    assert agent.q_table.shape == (16, 4)

def test_classical_world_model():
    model = ClassicalTransitionModel()
    probs = model.predict_next_state_distribution((0, 0), 1)
    assert len(probs) == 16
    assert np.isclose(np.sum(probs), 1.0)
    loss = model.train_on_batch([(0, 0)], [1], [(1, 0)], steps=1)
    assert loss >= 0.0

def test_quantum_world_model():
    model = QuantumTransitionModel(num_blocks=2)
    probs = model.predict_next_state_distribution((0, 0), 1)
    assert len(probs) == 16
    assert np.isclose(np.sum(probs), 1.0)
    loss = model.train_on_batch([(0, 0)], [1], [(1, 0)], steps=1)
    assert loss >= 0.0

def test_hybrid_world_model():
    cm = ClassicalTransitionModel()
    qm = QuantumTransitionModel(num_blocks=2)
    hm = HybridTransitionModel(cm, qm, initial_alpha=0.5)

    probs = hm.predict_next_state_distribution((0, 0), 1)
    assert len(probs) == 16
    assert np.isclose(np.sum(probs), 1.0)

    val_alpha = hm.update_validation_alpha([(0, 0)], [1], [(1, 0)])
    assert 0.0 <= val_alpha <= 1.0

def test_sampling_planner():
    cm = ClassicalTransitionModel()
    agent = TabularQAgent()
    planner = SamplingPlanner(num_samples=4, horizon=3)

    action = planner.select_action((0, 0), cm, agent.q_table)
    assert 0 <= action < 4
