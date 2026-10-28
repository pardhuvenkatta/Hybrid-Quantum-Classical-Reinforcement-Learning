"""
All-architecture hybrid benchmark (7 architectures x 5 horizons).

Produces `artifacts/all_architectures_benchmark_results.json`.

This benchmark evaluates 7 architectures across 5 planning horizons (H=2..6)
on Canonical GridWorld, providing the extended architectural horizon scaling
data. Table 2 in `paper_draft.md §5.2` reports multi-domain evaluations
from `artifacts/grand_hybrid_comparison_results.json`.
"""

import os
import sys
import json
import time
import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any

from src.environment.config import CanonicalEnvironmentConfig
from src.baselines.classical_agent import ClassicalAgent
from src.quantum.quantum_agent import QuantumAgent
from src.quantum.adaptive_quantum_agent import AdaptiveQuantumAgent
from src.agents.ppo_q_agent import PPOQAgent
from src.baselines.qlstm_reservoir_agent import QLSTMReservoirAgent
from src.baselines.qrnn_agent import QuantumRNNAgent
from src.baselines.qmcts_agent import QuantumMCTSAgent

def calculate_cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / s_pooled

def get_ci95(std_val, N=20):
    return 1.96 * (std_val / np.sqrt(N))

N_COMPARISONS = 6

def run_all_architectures_benchmark():
    start_time = time.time()
    print("=========================================================================")
    print("  COMPREHENSIVE ALL-ARCHITECTURE HYBRID QUANTUM BENCHMARK (N=20 SEEDS)  ")
    print("=========================================================================")

    num_seeds = 20
    seeds = list(range(100, 100 + num_seeds))
    episodes = 50
    horizons = [2, 3, 4, 5, 6]
    grid_cfg = CanonicalEnvironmentConfig

    architecture_names = [
        "Classical_Baseline",
        "Fixed_QWM_Baseline",
        "Adaptive_QWM_Ours",
        "PPO_Q",
        "QLSTM_Reservoir",
        "Q_RNN",
        "Q_MCTS"
    ]

    results = {
        "metadata": {
            "num_seeds": num_seeds,
            "horizons": horizons,
            "episodes": episodes,
            "grid_size": grid_cfg.GRID_WIDTH,
            "architectures_evaluated": architecture_names,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "scaling_benchmark": {}
    }

    for H in horizons:
        qubits = 2 * H
        futures = 4**H
        print(f"\n[HORIZON H={H}] Qubits={qubits}, Path Space={futures}...")

        h_data = {arch: {"rewards": [], "success": []} for arch in architecture_names}

        for seed in seeds:
            # 1. Classical Baseline
            np.random.seed(seed)
            env = grid_cfg.create_env()
            agent = ClassicalAgent(env=env, horizon=H, eval_budget=futures)
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
                ep_r.append(total_r)
                ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            h_data["Classical_Baseline"]["rewards"].append(np.mean(ep_r))
            h_data["Classical_Baseline"]["success"].append(float(np.mean(ep_s)))

            # 2. Fixed-QWM Baseline
            np.random.seed(seed)
            env = grid_cfg.create_env()
            agent = QuantumAgent(env=env, horizon=H, shots=128)
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
                ep_r.append(total_r)
                ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            h_data["Fixed_QWM_Baseline"]["rewards"].append(np.mean(ep_r))
            h_data["Fixed_QWM_Baseline"]["success"].append(float(np.mean(ep_s)))

            # 3. Adaptive-QWM Ours
            np.random.seed(seed)
            env = grid_cfg.create_env()
            agent = AdaptiveQuantumAgent(env=env, horizon=H)
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
                agent.on_episode_end(total_r)
                ep_r.append(total_r)
                ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            h_data["Adaptive_QWM_Ours"]["rewards"].append(np.mean(ep_r))
            h_data["Adaptive_QWM_Ours"]["success"].append(float(np.mean(ep_s)))

            # 4. PPO-Q
            np.random.seed(seed)
            env = grid_cfg.create_env()
            agent = PPOQAgent(env=env, horizon=H)
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
                ep_r.append(total_r)
                ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            h_data["PPO_Q"]["rewards"].append(np.mean(ep_r))
            h_data["PPO_Q"]["success"].append(float(np.mean(ep_s)))

            # 5. QLSTM-Reservoir
            np.random.seed(seed)
            env = grid_cfg.create_env()
            agent = QLSTMReservoirAgent(env=env, horizon=H)
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
                ep_r.append(total_r)
                ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            h_data["QLSTM_Reservoir"]["rewards"].append(np.mean(ep_r))
            h_data["QLSTM_Reservoir"]["success"].append(float(np.mean(ep_s)))

            # 6. Q-RNN
            np.random.seed(seed)
            env = grid_cfg.create_env()
            agent = QuantumRNNAgent(env=env, horizon=H)
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
                ep_r.append(total_r)
                ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            h_data["Q_RNN"]["rewards"].append(np.mean(ep_r))
            h_data["Q_RNN"]["success"].append(float(np.mean(ep_s)))

            # 7. Q-MCTS
            np.random.seed(seed)
            env = grid_cfg.create_env()
            agent = QuantumMCTSAgent(env=env, horizon=H)
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
                ep_r.append(total_r)
                ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            h_data["Q_MCTS"]["rewards"].append(np.mean(ep_r))
            h_data["Q_MCTS"]["success"].append(float(np.mean(ep_s)))

            print(f"  Seed {seed} -> Class: {h_data['Classical_Baseline']['rewards'][-1]:.4f} | Adapt-QWM: {h_data['Adaptive_QWM_Ours']['rewards'][-1]:.4f} | Fixed-QWM: {h_data['Fixed_QWM_Baseline']['rewards'][-1]:.4f} | PPO-Q: {h_data['PPO_Q']['rewards'][-1]:.4f} | QLSTM: {h_data['QLSTM_Reservoir']['rewards'][-1]:.4f} | Q-RNN: {h_data['Q_RNN']['rewards'][-1]:.4f} | Q-MCTS: {h_data['Q_MCTS']['rewards'][-1]:.4f}")

        # Aggregate metrics per architecture
        key = f"H{H}"
        results["scaling_benchmark"][key] = {}
        for arch in architecture_names:
            r_arr = np.array(h_data[arch]["rewards"])
            s_arr = np.array(h_data[arch]["success"])
            results["scaling_benchmark"][key][arch] = {
                "mean_reward": float(np.mean(r_arr)),
                "std_reward": float(np.std(r_arr)),
                "ci95": float(get_ci95(np.std(r_arr))),
                "success_rate": float(np.mean(s_arr))
            }

        # Statistical comparisons against Adaptive_QWM_Ours
        results["scaling_benchmark"][key]["stats_vs_adaptive"] = {}
        ad_arr = np.array(h_data["Adaptive_QWM_Ours"]["rewards"])
        for arch in architecture_names:
            if arch != "Adaptive_QWM_Ours":
                other_arr = np.array(h_data[arch]["rewards"])
                t_stat, p_val = stats.ttest_rel(ad_arr, other_arr)
                d_val = calculate_cohens_d(ad_arr, other_arr)
                results["scaling_benchmark"][key]["stats_vs_adaptive"][arch] = {
                    "t_stat": float(t_stat),
                    "p_val": float(p_val),
                    "p_val_bonferroni": float(min(1.0, p_val * N_COMPARISONS)),
                    "cohens_d": float(d_val)
                }

    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/all_architectures_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all-architecture benchmark JSON to: {out_path}")

    elapsed = time.time() - start_time
    print(f"All-architecture benchmark completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_all_architectures_benchmark()
