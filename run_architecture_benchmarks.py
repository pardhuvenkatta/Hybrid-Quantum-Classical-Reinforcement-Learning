"""
Sub-architecture comparison (QWM vs PPO-Q vs QLSTM-Reservoir, H=2..6).

Produces `artifacts/architecture_benchmark_results.json`.

This JSON is consumed DIRECTLY by the discussion of specialized architectures
in `paper_draft.md` — it is NOT read by any `generate_*.py` script. Kept as a
runner for reproducibility; if you want a figure, write a dedicated
`generate_architecture_figures.py` that reads this file.
"""

import os
import json
import time
import numpy as np
import scipy.stats as stats

from src.environment.config import CanonicalEnvironmentConfig
from src.quantum.quantum_agent import QuantumAgent
from src.agents.ppo_q_agent import PPOQAgent
from src.baselines.qlstm_reservoir_agent import QLSTMReservoirAgent

N_COMPARISONS = 10

def calculate_cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / s_pooled

def get_ci95(std_val, N=20):
    return 1.96 * (std_val / np.sqrt(N))

def run_arch_benchmarks():
    start_time = time.time()
    print("=========================================================================")
    print("  HYBRID ARCHITECTURE BENCHMARKING (QWM vs PPO-Q vs QLSTM-Reservoir)   ")
    print("=========================================================================")

    num_seeds = 20
    horizons = [2, 3, 4, 5, 6]
    episodes = 50
    shots = 128
    grid_cfg = CanonicalEnvironmentConfig

    results = {
        "metadata": {
            "num_seeds": num_seeds,
            "horizons": horizons,
            "episodes": episodes,
            "grid_size": grid_cfg.GRID_WIDTH,
            "n_comparisons": N_COMPARISONS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "architectures": {
            "QWM_Baseline": {},
            "PPO_Q": {},
            "QLSTM_Reservoir": {}
        },
        "statistical_comparisons": {}
    }

    for H in horizons:
        qubits = 2 * H
        futures = 4**H
        print(f"\n[BENCHMARKING HORIZON H={H}] {qubits} Qubits, {futures} Path Space...")

        qwm_rewards, ppo_rewards, qlstm_rewards = [], [], []
        qwm_succ, ppo_succ, qlstm_succ = [], [], []

        seeds = list(range(100, 100 + num_seeds))
        for seed in seeds:
            # 1. QWM Baseline Agent
            np.random.seed(seed)
            env_qwm = grid_cfg.create_env()
            qwm_agent = QuantumAgent(env=env_qwm, horizon=H, shots=shots)
            ep_r, ep_s = [], []
            for ep in range(episodes):
                pos = env_qwm.reset()
                total_r = 0.0
                done = False
                while not done:
                    a = qwm_agent.select_action(pos)
                    next_p, r, done, info = env_qwm.step(a)
                    qwm_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                ep_r.append(total_r)
                ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            qwm_rewards.append(np.mean(ep_r))
            qwm_succ.append(float(np.mean(ep_s)))

            # 2. PPO-Q Agent
            np.random.seed(seed)
            env_ppo = grid_cfg.create_env()
            ppo_agent = PPOQAgent(env=env_ppo, horizon=H)
            ep_r, ep_s = [], []
            for ep in range(episodes):
                pos = env_ppo.reset()
                total_r = 0.0
                done = False
                while not done:
                    a = ppo_agent.select_action(pos)
                    next_p, r, done, info = env_ppo.step(a)
                    ppo_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                ep_r.append(total_r)
                ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            ppo_rewards.append(np.mean(ep_r))
            ppo_succ.append(float(np.mean(ep_s)))

            # 3. QLSTM Reservoir Agent
            np.random.seed(seed)
            env_qlstm = grid_cfg.create_env()
            qlstm_agent = QLSTMReservoirAgent(env=env_qlstm, horizon=H)
            ep_r, ep_s = [], []
            for ep in range(episodes):
                pos = env_qlstm.reset()
                total_r = 0.0
                done = False
                while not done:
                    a = qlstm_agent.select_action(pos)
                    next_p, r, done, info = env_qlstm.step(a)
                    qlstm_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                ep_r.append(total_r)
                ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            qlstm_rewards.append(np.mean(ep_r))
            qlstm_succ.append(float(np.mean(ep_s)))

        r_qwm, r_ppo, r_qlstm = np.array(qwm_rewards), np.array(ppo_rewards), np.array(qlstm_rewards)

        results["architectures"]["QWM_Baseline"][f"H{H}"] = {
            "mean_reward": float(np.mean(r_qwm)),
            "std_reward": float(np.std(r_qwm)),
            "ci95": float(get_ci95(np.std(r_qwm), num_seeds)),
            "success_rate": float(np.mean(qwm_succ))
        }

        results["architectures"]["PPO_Q"][f"H{H}"] = {
            "mean_reward": float(np.mean(r_ppo)),
            "std_reward": float(np.std(r_ppo)),
            "ci95": float(get_ci95(np.std(r_ppo), num_seeds)),
            "success_rate": float(np.mean(ppo_succ))
        }

        results["architectures"]["QLSTM_Reservoir"][f"H{H}"] = {
            "mean_reward": float(np.mean(r_qlstm)),
            "std_reward": float(np.std(r_qlstm)),
            "ci95": float(get_ci95(np.std(r_qlstm), num_seeds)),
            "success_rate": float(np.mean(qlstm_succ))
        }

        # Statistical paired t-tests vs QWM Baseline
        t_ppo, p_ppo = stats.ttest_rel(r_ppo, r_qwm)
        d_ppo = calculate_cohens_d(r_ppo, r_qwm)
        
        t_qlstm, p_qlstm = stats.ttest_rel(r_qlstm, r_qwm)
        d_qlstm = calculate_cohens_d(r_qlstm, r_qwm)

        results["statistical_comparisons"][f"H{H}"] = {
            "PPO_Q_vs_QWM": {
                "t_stat": float(t_ppo),
                "p_val": float(p_ppo),
                "p_val_bonferroni": float(min(1.0, p_ppo * N_COMPARISONS)),
                "cohens_d": float(d_ppo)
            },
            "QLSTM_vs_QWM": {
                "t_stat": float(t_qlstm),
                "p_val": float(p_qlstm),
                "p_val_bonferroni": float(min(1.0, p_qlstm * N_COMPARISONS)),
                "cohens_d": float(d_qlstm)
            }
        }

        print(f"H={H} -> QWM Reward: {np.mean(r_qwm):.4f}, PPO-Q Reward: {np.mean(r_ppo):.4f} (d={d_ppo:.4f}), QLSTM Reward: {np.mean(r_qlstm):.4f} (d={d_qlstm:.4f})")

    # Save JSON data
    os.makedirs("artifacts", exist_ok=True)
    json_path = "artifacts/architecture_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved architecture benchmark results JSON to: {json_path}")

    elapsed = time.time() - start_time
    print(f"Architecture benchmarking completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_arch_benchmarks()
