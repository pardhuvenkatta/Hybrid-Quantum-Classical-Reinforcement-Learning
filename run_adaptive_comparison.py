import os
import sys
import json
import time
import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Optional, Tuple

from src.environment.config import CanonicalEnvironmentConfig
from src.environments.standard_benchmarks import create_benchmark_env, GridWorldMaze, FrozenLakeEnv, CartPoleEnv, AcrobotEnv
from src.quantum.noise_models import noise_free_model, depolarizing_noise_model, combined_nisq_noise_model
from src.baselines.classical_agent import ClassicalAgent
from src.quantum.quantum_agent import QuantumAgent
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

def calculate_convergence_speed(episodes_rewards: List[List[float]], target_ratio: float = 0.90) -> float:
    """Calculate average number of episodes to reach 90% of peak performance."""
    conv_episodes = []
    for ep_r in episodes_rewards:
        peak = np.max(ep_r)
        target = peak * target_ratio if peak > 0 else peak / target_ratio
        reached_ep = len(ep_r)
        for ep_idx, r in enumerate(ep_r):
            if r >= target:
                reached_ep = ep_idx + 1
                break
        conv_episodes.append(reached_ep)
    return float(np.mean(conv_episodes))

def run_adaptive_comparison_benchmark():
    start_time = time.time()
    print("=========================================================================")
    print("  ARCHITECTURE 2: AI-CONTROLLED QUANTUM FUTURE EXPLORATION BENCHMARK    ")
    print("=========================================================================")

    num_seeds = 20
    seeds = list(range(100, 100 + num_seeds))
    episodes = 50
    horizons = [2, 3, 4, 5, 6]

    results = {
        "metadata": {
            "num_seeds": num_seeds,
            "horizons": horizons,
            "episodes": episodes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "horizon_scaling": {},
        "primary_benchmarks": {},
        "noise_robustness": {},
        "maze_size_scaling": {}
    }

    grid_cfg = CanonicalEnvironmentConfig

    # 1. Horizon Scaling Benchmark H=2..6 on Canonical GridWorld
    print("\n--- Phase 1: Horizon Scaling (Canonical GridWorld, H=2..6, 20 Seeds) ---")
    for H in horizons:
        futures = 4**H
        print(f"\n[HORIZON H={H}] 4^H Budget = {futures}...")

        c_rewards, fq_rewards, ad_rewards = [], [], []
        c_succ, fq_succ, ad_succ = [], [], []
        c_ep_history, fq_ep_history, ad_ep_history = [], [], []

        fq_qpu_calls_list = []
        ad_qpu_calls, ad_shots, ad_depth, ad_branches = [], [], [], []
        ad_unc_list, ad_shots_list, ad_branches_list = [], [], []

        for seed in seeds:
            # A. Classical Baseline
            np.random.seed(seed)
            env_c = grid_cfg.create_env()
            c_agent = ClassicalAgent(env=env_c, horizon=H, eval_budget=futures)
            c_ep_r, c_ep_s = [], []
            for ep in range(episodes):
                pos = env_c.reset()
                total_r, done = 0.0, False
                while not done:
                    a = c_agent.select_action(pos)
                    next_p, r, done, info = env_c.step(a)
                    c_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                c_ep_r.append(total_r)
                c_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            c_rewards.append(np.mean(c_ep_r))
            c_succ.append(float(np.mean(c_ep_s)))
            c_ep_history.append(c_ep_r)

            # B. Fixed-Q Quantum Baseline (K=N/8, depth=2, M=128)
            np.random.seed(seed)
            env_fq = grid_cfg.create_env()
            fq_agent = QuantumAgent(env=env_fq, horizon=H, shots=128)
            fq_ep_r, fq_ep_s = [], []
            for ep in range(episodes):
                pos = env_fq.reset()
                total_r, done = 0.0, False
                while not done:
                    a = fq_agent.select_action(pos)
                    next_p, r, done, info = env_fq.step(a)
                    fq_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                fq_ep_r.append(total_r)
                fq_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            fq_rewards.append(np.mean(fq_ep_r))
            fq_succ.append(float(np.mean(fq_ep_s)))
            fq_ep_history.append(fq_ep_r)
            if hasattr(fq_agent.world_model, "total_circuit_evaluations"):
                fq_calls = fq_agent.world_model.total_circuit_evaluations
            elif hasattr(fq_agent, "total_qpu_calls"):
                fq_calls = fq_agent.total_qpu_calls
            else:
                raise AttributeError(
                    "Fixed-Q agent has no QPU-call counter "
                    "(neither world_model.total_circuit_evaluations nor total_qpu_calls); "
                    "cannot report measured qpu_calls. Instrument the agent before running this benchmark."
                )
            fq_qpu_calls_list.append(fq_calls)
            if fq_calls == 0:
                raise RuntimeError(
                    f"Fixed-Q agent reports 0 QPU circuit evaluations after running "
                    f"{episodes} episodes (seed={seed}, H={H}). The counter exists on the "
                    f"agent but is not being incremented during planning. Check that "
                    f"QuantumWorldModel.evaluate_futures_distribution() (or the equivalent "
                    f"circuit-evaluation path) increments .total_circuit_evaluations. "
                    f"Reporting qpu_calls=0 would silently invalidate the QPU-cost comparison."
                )

            # C. Adaptive-Q Agent (AI-Controlled Resources)
            np.random.seed(seed)
            env_ad = grid_cfg.create_env()
            ad_agent = AdaptiveQuantumAgent(env=env_ad, horizon=H)
            ad_ep_r, ad_ep_s = [], []
            for ep in range(episodes):
                pos = env_ad.reset()
                total_r, done = 0.0, False
                while not done:
                    a = ad_agent.select_action(pos)
                    next_p, r, done, info = env_ad.step(a)
                    ad_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                ad_agent.on_episode_end(total_r)
                ad_ep_r.append(total_r)
                ad_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            ad_rewards.append(np.mean(ad_ep_r))
            ad_succ.append(float(np.mean(ad_ep_s)))
            ad_ep_history.append(ad_ep_r)

            ad_qpu_calls.append(ad_agent.total_qpu_calls)
            ad_shots.append(ad_agent.total_shots_used / max(1, ad_agent.total_qpu_calls))
            ad_depth.append(ad_agent.total_circuit_layers / max(1, ad_agent.total_qpu_calls))
            ad_branches.append(np.mean(ad_agent.branches_marked_history) if ad_agent.branches_marked_history else 1)

            ad_unc_list.extend(ad_agent.uncertainty_history)
            ad_shots_list.extend(ad_agent.shots_history)
            ad_branches_list.extend(ad_agent.branches_marked_history)

            print(f"  Seed {seed} -> Class: {c_rewards[-1]:.4f} | Fixed-Q: {fq_rewards[-1]:.4f} | Adaptive-Q: {ad_rewards[-1]:.4f} (Avg Shots: {ad_shots[-1]:.1f})")

        c_arr, fq_arr, ad_arr = np.array(c_rewards), np.array(fq_rewards), np.array(ad_rewards)

        # Correlation between uncertainty and allocated resources
        if len(ad_unc_list) > 1 and np.std(ad_unc_list) > 0 and np.std(ad_shots_list) > 0:
            r_shots, p_shots = stats.pearsonr(ad_unc_list, ad_shots_list)
            r_branches, p_branches = stats.pearsonr(ad_unc_list, ad_branches_list)
        else:
            r_shots, p_shots, r_branches, p_branches = 0.0, 1.0, 0.0, 1.0

        # Statistical Pairwise Comparisons
        t_ad_fq, p_ad_fq = stats.ttest_rel(ad_arr, fq_arr)
        d_ad_fq = calculate_cohens_d(ad_arr, fq_arr)

        t_ad_c, p_ad_c = stats.ttest_rel(ad_arr, c_arr)
        d_ad_c = calculate_cohens_d(ad_arr, c_arr)

        t_fq_c, p_fq_c = stats.ttest_rel(fq_arr, c_arr)
        d_fq_c = calculate_cohens_d(fq_arr, c_arr)

        # NOTE: Fixed-Q QPU calls are instrumented and measured via
        # `fq_agent.world_model.total_circuit_evaluations`; Adaptive-Q reports
        # measured values via `ad_agent.total_qpu_calls`.
        results["horizon_scaling"][f"H{H}"] = {
            "classical": {"mean": float(np.mean(c_arr)), "std": float(np.std(c_arr)), "ci95": float(get_ci95(np.std(c_arr))), "success": float(np.mean(c_succ)), "conv_speed": calculate_convergence_speed(c_ep_history)},
            "fixed_q": {"mean": float(np.mean(fq_arr)), "std": float(np.std(fq_arr)), "ci95": float(get_ci95(np.std(fq_arr))), "success": float(np.mean(fq_succ)), "conv_speed": calculate_convergence_speed(fq_ep_history), "qpu_calls": float(np.mean(fq_qpu_calls_list)) if fq_qpu_calls_list else 0.0, "avg_shots": 128.0, "avg_depth": 2.0, "avg_branches": max(1, 4**H // 8)},
            "adaptive_q": {"mean": float(np.mean(ad_arr)), "std": float(np.std(ad_arr)), "ci95": float(get_ci95(np.std(ad_arr))), "success": float(np.mean(ad_succ)), "conv_speed": calculate_convergence_speed(ad_ep_history), "qpu_calls": float(np.mean(ad_qpu_calls)), "avg_shots": float(np.mean(ad_shots)), "avg_depth": float(np.mean(ad_depth)), "avg_branches": float(np.mean(ad_branches))},
            "correlations": {"r_uncertainty_shots": float(r_shots), "p_uncertainty_shots": float(p_shots), "r_uncertainty_branches": float(r_branches), "p_uncertainty_branches": float(p_branches)},
            "stats": {
                "adaptive_vs_fixed": {"t_stat": float(t_ad_fq), "p_val": float(p_ad_fq), "cohens_d": float(d_ad_fq)},
                "adaptive_vs_classical": {"t_stat": float(t_ad_c), "p_val": float(p_ad_c), "cohens_d": float(d_ad_c)},
                "fixed_vs_classical": {"t_stat": float(t_fq_c), "p_val": float(p_fq_c), "cohens_d": float(d_fq_c)}
            }
        }

    # 2. Standard Environment Benchmarks (CartPole-v1, Acrobot-v1, FrozenLake-v1, 8x8 Maze)
    print("\n--- Phase 2: Standard Environment Benchmarks ---")
    standard_benchmarks = ["CartPole-v1", "Acrobot-v1", "FrozenLake-v1", "Maze-8x8"]

    for bench_name in standard_benchmarks:
        print(f"\nBenchmarking {bench_name}...")
        c_rewards, fq_rewards, ad_rewards = [], [], []
        c_succ, fq_succ, ad_succ = [], [], []

        for seed in seeds:
            # Classical
            np.random.seed(seed)
            env_c = create_benchmark_env(bench_name)
            c_agent = ClassicalAgent(env=env_c, horizon=3, eval_budget=64)
            c_ep_r, c_ep_s = [], []
            for ep in range(episodes):
                pos = env_c.reset()
                total_r, done = 0.0, False
                while not done:
                    a = c_agent.select_action(pos)
                    next_p, r, done, info = env_c.step(a)
                    c_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                c_ep_r.append(total_r)
                c_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            c_rewards.append(np.mean(c_ep_r))
            c_succ.append(float(np.mean(c_ep_s)))

            # Fixed-Q
            np.random.seed(seed)
            env_fq = create_benchmark_env(bench_name)
            fq_agent = QuantumAgent(env=env_fq, horizon=3, shots=128)
            fq_ep_r, fq_ep_s = [], []
            for ep in range(episodes):
                pos = env_fq.reset()
                total_r, done = 0.0, False
                while not done:
                    a = fq_agent.select_action(pos)
                    next_p, r, done, info = env_fq.step(a)
                    fq_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                fq_ep_r.append(total_r)
                fq_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            fq_rewards.append(np.mean(fq_ep_r))
            fq_succ.append(float(np.mean(fq_ep_s)))

            # Adaptive-Q
            np.random.seed(seed)
            env_ad = create_benchmark_env(bench_name)
            ad_agent = AdaptiveQuantumAgent(env=env_ad, horizon=3)
            ad_ep_r, ad_ep_s = [], []
            for ep in range(episodes):
                pos = env_ad.reset()
                total_r, done = 0.0, False
                while not done:
                    a = ad_agent.select_action(pos)
                    next_p, r, done, info = env_ad.step(a)
                    ad_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                ad_agent.on_episode_end(total_r)
                ad_ep_r.append(total_r)
                ad_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            ad_rewards.append(np.mean(ad_ep_r))
            ad_succ.append(float(np.mean(ad_ep_s)))

        c_arr, fq_arr, ad_arr = np.array(c_rewards), np.array(fq_rewards), np.array(ad_rewards)
        t_ad_fq, p_ad_fq = stats.ttest_rel(ad_arr, fq_arr)
        d_ad_fq = calculate_cohens_d(ad_arr, fq_arr)

        results["primary_benchmarks"][bench_name] = {
            "classical": {"mean": float(np.mean(c_arr)), "std": float(np.std(c_arr)), "success": float(np.mean(c_succ))},
            "fixed_q": {"mean": float(np.mean(fq_arr)), "std": float(np.std(fq_arr)), "success": float(np.mean(fq_succ))},
            "adaptive_q": {"mean": float(np.mean(ad_arr)), "std": float(np.std(ad_arr)), "success": float(np.mean(ad_succ))},
            "stats": {"adaptive_vs_fixed": {"t_stat": float(t_ad_fq), "p_val": float(p_ad_fq), "cohens_d": float(d_ad_fq)}}
        }

    # 3. Noise Robustness Benchmark (Canonical Maze H=4)
    print("\n--- Phase 3: NISQ Quantum Noise Robustness (H=4) ---")
    noise_conditions = {
        "Noise-Free": noise_free_model(),
        "Depolarizing_p0.005": depolarizing_noise_model(0.005),
        "Combined_NISQ_Noise": combined_nisq_noise_model()
    }

    for n_name, n_model in noise_conditions.items():
        print(f"\nEvaluating Noise Condition: {n_name}...")
        fq_rewards, ad_rewards = [], []
        fq_succ, ad_succ = [], []

        for seed in seeds:
            # Fixed-Q with Noise
            np.random.seed(seed)
            env_fq = grid_cfg.create_env()
            fq_agent = QuantumAgent(env=env_fq, horizon=4, shots=128, noise_model=n_model)
            fq_ep_r, fq_ep_s = [], []
            for ep in range(episodes):
                pos = env_fq.reset()
                total_r, done = 0.0, False
                while not done:
                    a = fq_agent.select_action(pos)
                    next_p, r, done, info = env_fq.step(a)
                    fq_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                fq_ep_r.append(total_r)
                fq_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            fq_rewards.append(np.mean(fq_ep_r))
            fq_succ.append(float(np.mean(fq_ep_s)))

            # Adaptive-Q with Noise
            np.random.seed(seed)
            env_ad = grid_cfg.create_env()
            ad_agent = AdaptiveQuantumAgent(env=env_ad, horizon=4, noise_model=n_model)
            ad_ep_r, ad_ep_s = [], []
            for ep in range(episodes):
                pos = env_ad.reset()
                total_r, done = 0.0, False
                while not done:
                    a = ad_agent.select_action(pos)
                    next_p, r, done, info = env_ad.step(a)
                    ad_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                ad_agent.on_episode_end(total_r)
                ad_ep_r.append(total_r)
                ad_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            ad_rewards.append(np.mean(ad_ep_r))
            ad_succ.append(float(np.mean(ad_ep_s)))

        fq_arr, ad_arr = np.array(fq_rewards), np.array(ad_rewards)
        t_ad_fq, p_ad_fq = stats.ttest_rel(ad_arr, fq_arr)
        d_ad_fq = calculate_cohens_d(ad_arr, fq_arr)

        results["noise_robustness"][n_name] = {
            "fixed_q": {"mean": float(np.mean(fq_arr)), "std": float(np.std(fq_arr)), "success": float(np.mean(fq_succ))},
            "adaptive_q": {"mean": float(np.mean(ad_arr)), "std": float(np.std(ad_arr)), "success": float(np.mean(ad_succ))},
            "stats": {"adaptive_vs_fixed": {"t_stat": float(t_ad_fq), "p_val": float(p_ad_fq), "cohens_d": float(d_ad_fq)}}
        }

    # 4. Maze Size Scaling (4x4 to 16x16, H=4)
    print("\n--- Phase 4: Maze Size Scaling (4x4 to 16x16) ---")
    sizes = [4, 5, 8, 11, 16]
    for sz in sizes:
        print(f"\nEvaluating Maze Size: {sz}x{sz}...")
        c_rewards, fq_rewards, ad_rewards = [], [], []

        for seed in seeds:
            # Classical
            np.random.seed(seed)
            env_c = GridWorldMaze(width=sz, height=sz)
            c_agent = ClassicalAgent(env=env_c, horizon=4, eval_budget=256)
            c_ep_r = []
            for ep in range(episodes):
                pos = env_c.reset()
                total_r, done = 0.0, False
                while not done:
                    a = c_agent.select_action(pos)
                    next_p, r, done, info = env_c.step(a)
                    c_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                c_ep_r.append(total_r)
            c_rewards.append(np.mean(c_ep_r))

            # Fixed-Q
            np.random.seed(seed)
            env_fq = GridWorldMaze(width=sz, height=sz)
            fq_agent = QuantumAgent(env=env_fq, horizon=4, shots=128)
            fq_ep_r = []
            for ep in range(episodes):
                pos = env_fq.reset()
                total_r, done = 0.0, False
                while not done:
                    a = fq_agent.select_action(pos)
                    next_p, r, done, info = env_fq.step(a)
                    fq_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                fq_ep_r.append(total_r)
            fq_rewards.append(np.mean(fq_ep_r))

            # Adaptive-Q
            np.random.seed(seed)
            env_ad = GridWorldMaze(width=sz, height=sz)
            ad_agent = AdaptiveQuantumAgent(env=env_ad, horizon=4)
            ad_ep_r = []
            for ep in range(episodes):
                pos = env_ad.reset()
                total_r, done = 0.0, False
                while not done:
                    a = ad_agent.select_action(pos)
                    next_p, r, done, info = env_ad.step(a)
                    ad_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                ad_agent.on_episode_end(total_r)
                ad_ep_r.append(total_r)
            ad_rewards.append(np.mean(ad_ep_r))

        c_arr, fq_arr, ad_arr = np.array(c_rewards), np.array(fq_rewards), np.array(ad_rewards)
        results["maze_size_scaling"][f"{sz}x{sz}"] = {
            "classical": float(np.mean(c_arr)),
            "fixed_q": float(np.mean(fq_arr)),
            "adaptive_q": float(np.mean(ad_arr))
        }

    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/adaptive_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSuccessfully saved benchmark JSON results to: {out_path}")

    elapsed = time.time() - start_time
    print(f"Adaptive comparison benchmark completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_adaptive_comparison_benchmark()
