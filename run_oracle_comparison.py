import os
import sys
import json
import time
import numpy as np
import scipy.stats as stats

from src.environment.config import CanonicalEnvironmentConfig
from src.baselines.classical_agent import ClassicalAgent
from src.quantum.quantum_agent import QuantumAgent
from src.parallel.runner import run_seeds_parallel

def calculate_cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / s_pooled

def get_ci95(std_val, N=20):
    return 1.96 * (std_val / np.sqrt(N))

def _oracle_seed_task(seed, H, episodes, shots, futures):
    grid_cfg = CanonicalEnvironmentConfig

    # 1. Classical Baseline
    np.random.seed(seed)
    env_c = grid_cfg.create_env()
    c_agent = ClassicalAgent(env=env_c, horizon=H, eval_budget=futures)
    c_ep_r, c_ep_s = [], []
    for ep in range(episodes):
        pos = env_c.reset()
        total_r = 0.0
        done = False
        while not done:
            a = c_agent.select_action(pos)
            next_p, r, done, info = env_c.step(a)
            c_agent.update(pos, a, r, next_p, done)
            total_r += r
            pos = next_p
        c_ep_r.append(total_r)
        c_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)

    # 2. Quantum Exact Oracle (Grover phase flip pi)
    np.random.seed(seed)
    env_qe = grid_cfg.create_env()
    qe_agent = QuantumAgent(env=env_qe, horizon=H, shots=shots, oracle_mode="exact")
    qe_ep_r, qe_ep_s = [], []
    for ep in range(episodes):
        pos = env_qe.reset()
        total_r = 0.0
        done = False
        while not done:
            a = qe_agent.select_action(pos)
            next_p, r, done, info = env_qe.step(a)
            qe_agent.update(pos, a, r, next_p, done)
            total_r += r
            pos = next_p
        qe_ep_r.append(total_r)
        qe_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)

    # 3. Quantum Continuous Oracle (Phase gradient beta*pi*norm_v)
    np.random.seed(seed)
    env_qc = grid_cfg.create_env()
    qc_agent = QuantumAgent(env=env_qc, horizon=H, shots=shots, oracle_mode="continuous", beta=1.0)
    qc_ep_r, qc_ep_s = [], []
    for ep in range(episodes):
        pos = env_qc.reset()
        total_r = 0.0
        done = False
        while not done:
            a = qc_agent.select_action(pos)
            next_p, r, done, info = env_qc.step(a)
            qc_agent.update(pos, a, r, next_p, done)
            total_r += r
            pos = next_p
        qc_ep_r.append(total_r)
        qc_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)

    return {
        "classical_reward": float(np.mean(c_ep_r)),
        "classical_success": float(np.mean(c_ep_s)),
        "exact_reward": float(np.mean(qe_ep_r)),
        "exact_success": float(np.mean(qe_ep_s)),
        "continuous_reward": float(np.mean(qc_ep_r)),
        "continuous_success": float(np.mean(qc_ep_s))
    }

def run_oracle_comparison():
    start_time = time.time()
    print("=========================================================================")
    print("  QUANTUM WORLD MODEL ORACLE A/B COMPARISON (EXACT VS CONTINUOUS PHASE)  ")
    print("=========================================================================")

    num_seeds = 20
    horizons = [2, 3, 4, 5, 6]
    episodes = 50
    shots = 128
    grid_cfg = CanonicalEnvironmentConfig

    total_comparisons = len(horizons) * 3  # 5 horizons * 3 pairwise tests = 15

    results = {
        "metadata": {
            "num_seeds": num_seeds,
            "horizons": horizons,
            "episodes": episodes,
            "shots": shots,
            "grid_size": grid_cfg.GRID_WIDTH,
            "total_bonferroni_comparisons": total_comparisons,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    print(f"Canonical Environment Setup:")
    print(f"  Grid Size: {grid_cfg.GRID_WIDTH}x{grid_cfg.GRID_HEIGHT}")
    print(f"  Start: {grid_cfg.START_POS}, Goal: {grid_cfg.GOAL_POS}")
    start_seed = 100
    seeds = list(range(start_seed, start_seed + num_seeds))
    print(f"  Seeds: {start_seed} to {start_seed + num_seeds - 1} (N={num_seeds})")
    print(f"  Episodes: {episodes}, Shots: {shots}\n")

    for H in horizons:
        qubits = 2 * H
        futures = 4**H
        print(f"\n[HORIZON H={H}] {qubits} Qubits, {futures} Path Space (4^H)...")

        per_seed = run_seeds_parallel(
            task_fn=_oracle_seed_task,
            seeds=seeds,
            task_kwargs=dict(H=H, episodes=episodes, shots=shots, futures=futures),
            progress=True
        )
        ordered = [per_seed[s] for s in sorted(per_seed)]
        c_rewards = [r["classical_reward"] for r in ordered]
        c_succ    = [r["classical_success"] for r in ordered]
        q_exact_rewards = [r["exact_reward"] for r in ordered]
        q_exact_succ    = [r["exact_success"] for r in ordered]
        q_cont_rewards  = [r["continuous_reward"] for r in ordered]
        q_cont_succ     = [r["continuous_success"] for r in ordered]

        c_arr = np.array(c_rewards)
        qe_arr = np.array(q_exact_rewards)
        qc_arr = np.array(q_cont_rewards)

        # Statistical tests: Exact vs Classical
        t_stat_exact_c, p_exact_c = stats.ttest_rel(qe_arr, c_arr)
        d_exact_c = calculate_cohens_d(qe_arr, c_arr)
        p_bonf_exact_c = min(1.0, float(p_exact_c) * total_comparisons)

        # Statistical tests: Continuous vs Classical
        t_stat_cont_c, p_cont_c = stats.ttest_rel(qc_arr, c_arr)
        d_cont_c = calculate_cohens_d(qc_arr, c_arr)
        p_bonf_cont_c = min(1.0, float(p_cont_c) * total_comparisons)

        # Statistical tests: Exact vs Continuous
        t_stat_exact_cont, p_exact_cont = stats.ttest_rel(qe_arr, qc_arr)
        d_exact_cont = calculate_cohens_d(qe_arr, qc_arr)
        p_bonf_exact_cont = min(1.0, float(p_exact_cont) * total_comparisons)

        key = f"H{H}"
        results[key] = {
            "classical": {
                "mean": float(np.mean(c_arr)),
                "std": float(np.std(c_arr)),
                "ci95": float(get_ci95(np.std(c_arr), num_seeds)),
                "success": float(np.mean(c_succ))
            },
            "quantum_exact": {
                "mean": float(np.mean(qe_arr)),
                "std": float(np.std(qe_arr)),
                "ci95": float(get_ci95(np.std(qe_arr), num_seeds)),
                "success": float(np.mean(q_exact_succ))
            },
            "quantum_continuous": {
                "mean": float(np.mean(qc_arr)),
                "std": float(np.std(qc_arr)),
                "ci95": float(get_ci95(np.std(qc_arr), num_seeds)),
                "success": float(np.mean(q_cont_succ))
            },
            "stats": {
                "exact_vs_classical": {
                    "t_stat": float(t_stat_exact_c),
                    "p_val_uncorrected": float(p_exact_c),
                    "p_val_bonferroni": float(p_bonf_exact_c),
                    "cohens_d": float(d_exact_c)
                },
                "continuous_vs_classical": {
                    "t_stat": float(t_stat_cont_c),
                    "p_val_uncorrected": float(p_cont_c),
                    "p_val_bonferroni": float(p_bonf_cont_c),
                    "cohens_d": float(d_cont_c)
                },
                "exact_vs_continuous": {
                    "t_stat": float(t_stat_exact_cont),
                    "p_val_uncorrected": float(p_exact_cont),
                    "p_val_bonferroni": float(p_bonf_exact_cont),
                    "cohens_d": float(d_exact_cont)
                }
            }
        }

        print(f"\n[SUMMARY H={H}]")
        print(f"  Classical:  {np.mean(c_arr):.4f} +/- {get_ci95(np.std(c_arr)):.4f} | Succ: {np.mean(c_succ)*100:.1f}%")
        print(f"  Q-Exact:    {np.mean(qe_arr):.4f} +/- {get_ci95(np.std(qe_arr)):.4f} | Succ: {np.mean(q_exact_succ)*100:.1f}%")
        print(f"  Q-Cont:     {np.mean(qc_arr):.4f} +/- {get_ci95(np.std(qc_arr)):.4f} | Succ: {np.mean(q_cont_succ)*100:.1f}%")
        print(f"  Exact vs Cont: t={t_stat_exact_cont:.4f}, p_bonf={p_bonf_exact_cont:.4e}, d={d_exact_cont:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/oracle_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved oracle comparison results JSON to: {out_path}")

    elapsed = time.time() - start_time
    print(f"Oracle comparison benchmark completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_oracle_comparison()
