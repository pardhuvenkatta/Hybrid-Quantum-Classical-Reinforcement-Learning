import pytest
import numpy as np
from src.environments.standard_benchmarks import create_benchmark_env
from src.agents.ppo_q_agent import PPOQAgent
from src.agents.q_spirl_agent import QSpiRLAgent
from src.agents.qgrnn_agent import QGRNNAgent
from src.agents.qbrl_agent import QBRLAgent
from src.agents.hqcpg_agent import HQCPGAgent
from src.agents.adaptive_quantum_agent import AdaptiveQuantumAgent

def test_all_six_grand_agents():
    env = create_benchmark_env("maze", width=4, height=4)
    agents = [
        PPOQAgent(env=env, horizon=2),
        QSpiRLAgent(env=env, horizon=2),
        QGRNNAgent(env=env, horizon=2),
        QBRLAgent(env=env, horizon=2),
        HQCPGAgent(env=env, horizon=2),
        AdaptiveQuantumAgent(env=env, horizon=2)
    ]

    for agent in agents:
        action = agent.select_action((0, 0))
        assert action in range(env.num_actions)
        agent.update((0, 0), action, -0.05, (0, 1), False)
        assert hasattr(agent, "parameter_count")
