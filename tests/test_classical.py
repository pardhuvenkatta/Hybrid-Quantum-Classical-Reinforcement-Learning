import pytest
from src.environment.gridworld import GridWorld
from src.baselines.classical_world_model import ClassicalWorldModel
from src.baselines.classical_agent import ClassicalAgent

def test_classical_world_model():
    env = GridWorld(width=5, height=5)
    model = ClassicalWorldModel(env=env, horizon=2)
    trajectories = model.generate_candidate_trajectories()
    assert len(trajectories) == 16  # 4^2 candidate paths

    evals = model.evaluate_futures((0, 0))
    assert len(evals) == 16
    assert model.total_evaluations == 16

def test_classical_agent_action_selection():
    env = GridWorld(width=5, height=5)
    agent = ClassicalAgent(env=env, horizon=2, eval_budget=16)
    action = agent.select_action((0, 0))
    assert action in [0, 1, 2, 3]

    agent.update((0, 0), action, -0.02, (0, 1), False)
    assert agent.q_table.shape == (25, 4)
