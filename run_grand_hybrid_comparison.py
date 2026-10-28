"""
Grand hybrid comparison benchmark (7 architectures x 4 benchmarks + noise + maze scaling).

Scope note: This runner does NOT sweep planning horizons. Each phase uses a fixed
horizon (Phase 1: H=3, Phase 2: H=4, Phase 3: H=4). For horizon sweeps, use
run_canonical_experiments.py (H=2..6), run_adaptive_comparison.py (H=2..6),
or run_extended_benchmarks.py (H=1..6).
"""

import os
import sys
import json
import time
import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any

from src.environment.config import CanonicalEnvironmentConfig
from src.environments.standard_benchmarks import create_benchmark_env, GridWorldMaze
from src.quantum.noise_models import noise_free_model, depolarizing_noise_model, combined_nisq_noise_model
from src.baselines.classical_agent import ClassicalAgent
from src.quantum.quantum_agent import QuantumAgent
from src.agents.ppo_q_agent import PPOQAgent
from src.agents.q_spirl_agent import QSpiRLAgent
from src.agents.qgrnn_agent import QGRNNAgent
from src.agents.qbrl_agent import QBRLAgent
from src.agents.hqcpg_agent import HQCPGAgent
from src.quantum.adaptive_quantum_agent import AdaptiveQuantumAgent

def calculate_cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / s_pooled

def get_ci95(std_val, N=20):
    return 1.96 * (std_val / np.sqrt(N))

def calculate_convergence_speed(episodes_rewards, target_ratio=0.90):
    """
    Episodes to reach target_ratio × peak performance.
    For positive-reward tasks (CartPole): peak × target_ratio (higher is better).
    For negative-reward tasks (Acrobot): |peak| × target_ratio (less negative is better).
    """
    conv_episodes = []
    for ep_r in episodes_rewards:
        peak = np.max(ep_r)
        if peak > 0:
            target = peak * target_ratio
            reached_ep = next(
                (i + 1 for i, r in enumerate(ep_r) if r >= target),
                len(ep_r),
            )
        else:
            peak_mag = abs(np.min(ep_r))
            target_mag = peak_mag * target_ratio
            reached_ep = next(
                (i + 1 for i, r in enumerate(ep_r) if abs(r) <= target_mag),
                len(ep_r),
            )
        conv_episodes.append(reached_ep)
    return float(np.mean(conv_episodes))

def instantiate_agent(arch_name: str, env: Any, horizon: int, noise_model: Any = None):
    """Return (agent, noise_applied) where noise_applied indicates whether the
    noise_model kwarg was actually consumed by this architecture's constructor."""
    if arch_name == "Classical_Baseline":
        return ClassicalAgent(env=env, horizon=horizon, eval_budget=4**horizon), False
    elif arch_name == "Fixed_QWM_Baseline":
        return QuantumAgent(env=env, horizon=horizon, shots=128, noise_model=noise_model), True
    elif arch_name == "PPO_Q":
        return PPOQAgent(env=env, horizon=horizon), False
    elif arch_name == "Q_SpiRL":
        return QSpiRLAgent(env=env, horizon=horizon), False
    elif arch_name == "QGRNN":
        return QGRNNAgent(env=env, horizon=horizon), False
    elif arch_name == "QBRL":
        return QBRLAgent(env=env, horizon=horizon), False
    elif arch_name == "HQCPG":
        return HQCPGAgent(env=env, horizon=horizon), False
    elif arch_name == "Adaptive_Q_Ours":
        return AdaptiveQuantumAgent(env=env, horizon=horizon, noise_model=noise_model), True
    else:
        raise ValueError(f"Unknown architecture name: {arch_name}")

def run_grand_hybrid_comparison_benchmark():
    start_time = time.time()
    print("=========================================================================")
    print("  GRAND BENCHMARK OF 6 HYBRID QUANTUM-CLASSICAL ARCHITECTURES (N=20)   ")
    print("=========================================================================")

    num_seeds = 20
    seeds = list(range(100, 100 + num_seeds))
    episodes = 50

    architecture_names = [
        "Classical_Baseline",
        "PPO_Q",
        "Q_SpiRL",
        "QGRNN",
        "QBRL",
        "HQCPG",
        "Adaptive_Q_Ours"
    ]

    results = {
        "metadata": {
            "num_seeds": num_seeds,
            "episodes": episodes,
            "architectures_evaluated": architecture_names,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "primary_benchmarks": {},
        "noise_robustness": {},
        "maze_size_scaling": {}
    }

    grid_cfg = CanonicalEnvironmentConfig

    # 1. Primary Benchmarks (CartPole-v1, Acrobot-v1, FrozenLake-v1, Maze-8x8)
    print("\n--- Phase 1: Primary Benchmarks ---")
    benchmarks = ["CartPole-v1", "Acrobot-v1", "FrozenLake-v1", "Maze-8x8"]

    for bench in benchmarks:
        print(f"\nEvaluating Benchmark: {bench}...")
        results["primary_benchmarks"][bench] = {}
        b_data = {arch: {"rewards": [], "success": [], "ep_hist": [], "param_count": 0} for arch in architecture_names}

        for seed in seeds:
            for arch in architecture_names:
                np.random.seed(seed)
                env = create_benchmark_env(bench)
                agent, _ = instantiate_agent(arch, env, horizon=3)
                b_data[arch]["param_count"] = getattr(agent, "parameter_count", getattr(agent, "hilbert_dim", 16))

                ep_r, ep_s = [], []
                for ep in range(episodes):
                    pos = env.reset()
                    total_r, done = 0.0, False
                    while not done:
                        a = agent.select_action(pos)
                        next_p, r, done, info = env.step(a)
                        agent.update(pos, a, r, next_p, done)
                        total_r += r
                        pos = next_p
                    if hasattr(agent, "on_episode_end"):
                        agent.on_episode_end(total_r)
                    ep_r.append(total_r)
                    ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)

                b_data[arch]["rewards"].append(np.mean(ep_r))
                b_data[arch]["success"].append(float(np.mean(ep_s)))
                b_data[arch]["ep_hist"].append(ep_r)

        # Aggregate metrics & pairwise t-tests against Classical & Best Hybrid
        classical_arr = np.array(b_data["Classical_Baseline"]["rewards"])
        hybrid_means = {arch: np.mean(b_data[arch]["rewards"]) for arch in architecture_names if arch != "Classical_Baseline"}
        best_hybrid_arch = max(hybrid_means, key=hybrid_means.get)
        best_hybrid_arr = np.array(b_data[best_hybrid_arch]["rewards"])

        for arch in architecture_names:
            arr = np.array(b_data[arch]["rewards"])
            succ = np.array(b_data[arch]["success"])

            t_vs_c, p_vs_c = stats.ttest_rel(arr, classical_arr)
            d_vs_c = calculate_cohens_d(arr, classical_arr)

            t_vs_best, p_vs_best = stats.ttest_rel(arr, best_hybrid_arr)
            d_vs_best = calculate_cohens_d(arr, best_hybrid_arr)

            results["primary_benchmarks"][bench][arch] = {
                "mean_reward": float(np.mean(arr)),
                "std_reward": float(np.std(arr)),
                "ci95": float(get_ci95(np.std(arr))),
                "success_rate": float(np.mean(succ)),
                "success_definition": "reached_goal",
                # conv_speed omitted: recompute on next full run with fixed function
                "parameter_count": b_data[arch]["param_count"],
                "stats_vs_classical": {"t_stat": float(t_vs_c), "p_val": float(p_vs_c), "cohens_d": float(d_vs_c)},
                "stats_vs_best_hybrid": {"t_stat": float(t_vs_best), "p_val": float(p_vs_best), "cohens_d": float(d_vs_best)}
            }
        results["primary_benchmarks"][bench]["winning_architecture"] = best_hybrid_arch if np.mean(best_hybrid_arr) > np.mean(classical_arr) else "Classical_Baseline"
        print(f"  {bench} Winner: {results['primary_benchmarks'][bench]['winning_architecture']} (Best Hybrid: {best_hybrid_arch})")

    # 2. Noise Robustness (Noise-Free, Depolarizing p=0.005, Combined NISQ Noise)
    print("\n--- Phase 2: NISQ Noise Robustness (H=4 Canonical Maze) ---")
    noise_models = {
        "Noise-Free": noise_free_model(),
        "Depolarizing_p0.005": depolarizing_noise_model(0.005),
        "Combined_NISQ_Noise": combined_nisq_noise_model()
    }

    for n_name, n_model in noise_models.items():
        print(f"\nEvaluating Noise Condition: {n_name}...")
        results["noise_robustness"][n_name] = {}
        n_data = {arch: {"rewards": [], "success": []} for arch in architecture_names}
        noise_applied_map = {}

        for seed in seeds:
            for arch in architecture_names:
                np.random.seed(seed)
                env = grid_cfg.create_env()
                agent, noise_applied = instantiate_agent(arch, env, horizon=4, noise_model=n_model)
                noise_applied_map[arch] = noise_applied

                ep_r, ep_s = [], []
                for ep in range(episodes):
                    pos = env.reset()
                    total_r, done = 0.0, False
                    while not done:
                        a = agent.select_action(pos)
                        next_p, r, done, info = env.step(a)
                        agent.update(pos, a, r, next_p, done)
                        total_r += r
                        pos = next_p
                    if hasattr(agent, "on_episode_end"):
                        agent.on_episode_end(total_r)
                    ep_r.append(total_r)
                    ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)

                n_data[arch]["rewards"].append(np.mean(ep_r))
                n_data[arch]["success"].append(float(np.mean(ep_s)))

        classical_arr = np.array(n_data["Classical_Baseline"]["rewards"])
        for arch in architecture_names:
            arr = np.array(n_data[arch]["rewards"])
            succ = np.array(n_data[arch]["success"])
            t_vs_c, p_vs_c = stats.ttest_rel(arr, classical_arr)
            d_vs_c = calculate_cohens_d(arr, classical_arr)
            results["noise_robustness"][n_name][arch] = {
                "mean_reward": float(np.mean(arr)),
                "std_reward": float(np.std(arr)),
                "success_rate": float(np.mean(succ)),
                "success_definition": "reached_goal",
                "cohens_d_vs_classical": float(d_vs_c),
                "p_val_vs_classical": float(p_vs_c),
                "noise_model_applied": noise_applied_map[arch]
            }

    # 3. Maze Size Scaling (4x4 to 16x16)
    print("\n--- Phase 3: Maze Size Scaling (4x4 to 16x16) ---")
    sizes = [4, 5, 8, 11, 16]
    for sz in sizes:
        print(f"\nEvaluating Maze Size: {sz}x{sz}...")
        results["maze_size_scaling"][f"{sz}x{sz}"] = {}
        m_data = {arch: [] for arch in architecture_names}

        for seed in seeds:
            for arch in architecture_names:
                np.random.seed(seed)
                env = GridWorldMaze(width=sz, height=sz)
                agent, _ = instantiate_agent(arch, env, horizon=4)

                ep_r = []
                for ep in range(episodes):
                    pos = env.reset()
                    total_r, done = 0.0, False
                    while not done:
                        a = agent.select_action(pos)
                        next_p, r, done, info = env.step(a)
                        agent.update(pos, a, r, next_p, done)
                        total_r += r
                        pos = next_p
                    if hasattr(agent, "on_episode_end"):
                        agent.on_episode_end(total_r)
                    ep_r.append(total_r)

                m_data[arch].append(np.mean(ep_r))

        for arch in architecture_names:
            results["maze_size_scaling"][f"{sz}x{sz}"][arch] = float(np.mean(m_data[arch]))

    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/grand_hybrid_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSuccessfully saved grand hybrid comparison JSON to: {out_path}")

    elapsed = time.time() - start_time
    print(f"Grand hybrid benchmark completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_grand_hybrid_comparison_benchmark()
