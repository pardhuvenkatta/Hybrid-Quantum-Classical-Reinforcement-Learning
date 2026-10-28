import pytest
import numpy as np
from src.environments.standard_benchmarks import create_benchmark_env
from src.quantum.noise_models import depolarizing_noise_model, combined_nisq_noise_model

from src.agents.q_counterfactual_agent import QCounterfactualAgent
from src.agents.q_depth_adaptive_agent import QDepthAdaptiveAgent
from src.agents.q_multiagent_agent import QMultiAgentAgent
from src.agents.q_causal_agent import QCausalAgent
from src.agents.q_planner_classical_learner_agent import QPlannerClassicalLearnerAgent
from src.agents.q_attention_agent import QAttentionAgent
from src.agents.q_algorithm_selection_agent import QAlgorithmSelectionAgent

AGENT_CLASSES = [
    QCounterfactualAgent,
    QDepthAdaptiveAgent,
    QMultiAgentAgent,
    QCausalAgent,
    QPlannerClassicalLearnerAgent,
    QAttentionAgent,
    QAlgorithmSelectionAgent
]

BENCHMARKS = ["CartPole-v1", "Acrobot-v1", "FrozenLake-v1", "Maze-8x8"]

@pytest.mark.parametrize("agent_cls", AGENT_CLASSES)
@pytest.mark.parametrize("bench_name", BENCHMARKS)
def test_new_agent_instantiation_and_step(agent_cls, bench_name):
    env = create_benchmark_env(bench_name)
    agent = agent_cls(env=env, horizon=3)

    assert hasattr(agent, "parameter_count")
    assert agent.parameter_count > 0

    state = env.reset()
    action = agent.select_action(state)
    num_actions = getattr(env, "num_actions", getattr(getattr(env, "action_space", None), "n", 4))
    assert 0 <= action < num_actions

    next_state, reward, done, info = env.step(action)
    agent.update(state, action, reward, next_state, done)
    agent.on_episode_end(reward)

def test_new_agent_noise_compatibility():
    noise = combined_nisq_noise_model()
    env = create_benchmark_env("Maze-8x8")
    for agent_cls in AGENT_CLASSES:
        agent = agent_cls(env=env, horizon=3, noise_model=noise)
        state = env.reset()
        action = agent.select_action(state)
        assert 0 <= action < getattr(env, "num_actions", 4)
