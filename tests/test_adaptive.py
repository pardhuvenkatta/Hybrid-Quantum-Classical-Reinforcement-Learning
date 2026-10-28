import pytest
import numpy as np
from src.environments.standard_benchmarks import create_benchmark_env, GridWorldMaze, CartPoleEnv
from src.quantum.noise_models import noise_free_model, depolarizing_noise_model, combined_nisq_noise_model
from src.quantum.uncertainty_estimator import UncertaintyEstimator
from src.quantum.adaptive_controller import AdaptiveQPUController
from src.quantum.adaptive_quantum_agent import AdaptiveQuantumAgent

def test_standard_benchmarks():
    maze = create_benchmark_env("maze", width=5, height=5)
    assert maze.num_states == 25
    pos = maze.reset()
    assert pos == (0, 0)
    next_p, r, done, info = maze.step(1)
    assert next_p == (0, 1)

    cartpole = create_benchmark_env("cartpole")
    assert cartpole.num_actions == 2
    c_pos = cartpole.reset()
    assert isinstance(c_pos, tuple)
    next_c, c_r, c_done, _ = cartpole.step(1)
    assert isinstance(c_r, float)

def test_noise_models():
    n_free = noise_free_model()
    assert not n_free.is_noisy()

    n_comb = combined_nisq_noise_model()
    assert n_comb.is_noisy()
    psi = np.ones(16, dtype=complex) / np.sqrt(16)
    psi_n = n_comb.apply_statevector_noise(psi)
    assert len(psi_n) == 16
    assert pytest.approx(np.linalg.norm(psi_n), abs=1e-3) == 1.0

def test_uncertainty_estimator():
    estimator = UncertaintyEstimator(num_states=16)
    q_table = np.zeros((16, 4))
    u1 = estimator.estimate_uncertainty(0, q_table)
    assert 0.0 <= u1 <= 1.0

    estimator.update_state_experience(0, 1.0)
    u2 = estimator.estimate_uncertainty(0, q_table)
    assert 0.0 <= u2 <= 1.0

def test_adaptive_qpu_controller():
    controller = AdaptiveQPUController(hilbert_dim=16)
    r_low = controller.allocate_resources(0.1)
    assert r_low["category"] == "low"
    assert r_low["num_branches"] == 1
    assert r_low["circuit_depth"] == 1
    assert r_low["measurement_shots"] == 64

    r_high = controller.allocate_resources(0.9)
    assert r_high["category"] == "high"
    assert r_high["circuit_depth"] == 3
    assert r_high["measurement_shots"] == 256

def test_adaptive_quantum_agent():
    env = GridWorldMaze(width=5, height=5)
    agent = AdaptiveQuantumAgent(env=env, horizon=2)
    action = agent.select_action((0, 0))
    assert action in [0, 1, 2, 3]
    assert agent.total_qpu_calls == 1
    assert agent.total_shots_used in [64, 128, 256]
    agent.on_episode_end(0.5)
