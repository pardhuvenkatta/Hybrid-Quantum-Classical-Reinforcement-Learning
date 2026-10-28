import os
import sys
import json
import time
import argparse
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def _seed_task(seed, H, episodes, shots, futures):
    if episodes <= 0:
        raise ValueError(f"episodes must be >= 1, got {episodes}")

    from src.environment.config import CanonicalEnvironmentConfig
    from src.baselines.classical_agent import ClassicalAgent
    from src.quantum.quantum_agent import QuantumAgent
    import numpy as np

    grid_cfg = CanonicalEnvironmentConfig

    # Classical
    np.random.seed(seed)
    env_c = grid_cfg.create_env()
    c_agent = ClassicalAgent(env=env_c, horizon=H, eval_budget=futures)
    c_ep_r, c_ep_s = [], []
    info = {}
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

    # Quantum
    np.random.seed(seed)
    env_q = grid_cfg.create_env()
    q_agent = QuantumAgent(env=env_q, horizon=H, shots=shots)
    q_ep_r, q_ep_s = [], []
    info = {}
    for ep in range(episodes):
        pos = env_q.reset()
        total_r, done = 0.0, False
        while not done:
            a = q_agent.select_action(pos)
            next_p, r, done, info = env_q.step(a)
            q_agent.update(pos, a, r, next_p, done)
            total_r += r
            pos = next_p
        q_ep_r.append(total_r)
        q_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)

    return {
        "classical_mean_reward": float(np.mean(c_ep_r)),
        "classical_success_rate": float(np.mean(c_ep_s)),
        "quantum_mean_reward": float(np.mean(q_ep_r)),
        "quantum_success_rate": float(np.mean(q_ep_s)),
    }


def _ablation_seed_task(seed, enable_oracle, enable_interference, episodes, shots):
    if episodes <= 0:
        raise ValueError(f"episodes must be >= 1, got {episodes}")

    from src.environment.config import CanonicalEnvironmentConfig
    from src.quantum.quantum_agent import QuantumAgent
    import numpy as np

    grid_cfg = CanonicalEnvironmentConfig
    np.random.seed(seed)
    env_q = grid_cfg.create_env()
    q_agent = QuantumAgent(env=env_q, horizon=4, shots=shots,
                           enable_oracle=enable_oracle,
                           enable_interference=enable_interference)
    q_ep_r, q_ep_s = [], []
    info = {}
    for ep in range(episodes):
        pos = env_q.reset()
        total_r, done = 0.0, False
        while not done:
            a = q_agent.select_action(pos)
            next_p, r, done, info = env_q.step(a)
            q_agent.update(pos, a, r, next_p, done)
            total_r += r
            pos = next_p
        q_ep_r.append(total_r)
        q_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
    return {
        "mean_reward": float(np.mean(q_ep_r)),
        "success_rate": float(np.mean(q_ep_s)),
    }


def run_canonical_suite():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    args = parser.parse_args()

    if args.serial:
        args.workers = 1
    episodes = args.episodes
    horizons = args.horizons

    start_time = time.time()
    print("=========================================================================")
    print("  CANONICAL BENCHMARK & ABLATION EXPERIMENTAL SUITE (N=20 SEEDS, H=2..6) ")
    print("=========================================================================")

    num_seeds = 20
    shots = 128
    grid_cfg = CanonicalEnvironmentConfig

    print(f"Canonical Environment Configuration:")
    print(f"  Grid Size: {grid_cfg.GRID_WIDTH}x{grid_cfg.GRID_HEIGHT}")
    print(f"  Start: {grid_cfg.START_POS}, Goal: {grid_cfg.GOAL_POS}")
    print(f"  Obstacles: {grid_cfg.OBSTACLES}")
    print(f"  Traps: {grid_cfg.TRAPS}")
    print(f"  Step Penalty: {grid_cfg.STEP_PENALTY}, Revisit Penalty: {grid_cfg.REVISIT_PENALTY}, Goal Reward: {grid_cfg.GOAL_REWARD}\n")

    results = {
        "metadata": {
            "num_seeds": num_seeds,
            "horizons": horizons,
            "grid_width": grid_cfg.GRID_WIDTH,
            "grid_height": grid_cfg.GRID_HEIGHT,
            "episodes": episodes,
            "shots": shots,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "scaling_experiments": {},
        "ablation_experiments_H4": {}
    }

    start_seed = 100
    seeds = list(range(start_seed, start_seed + num_seeds))

    # 1. Horizon Scaling Experiments H=2..6
    print(f"--- Phase 1: Scaling Benchmark (H=2..6, 20 Seeds {start_seed}..{start_seed + num_seeds - 1} per Horizon) ---")
    for H in horizons:
        qubits = 2 * H
        futures = 4**H
        print(f"\n[HORIZON H={H}] {qubits} Qubits, {futures} Trajectory Budget (4^H)...")

        per_seed = run_seeds_parallel(
            task_fn=_seed_task,
            seeds=seeds,
            task_kwargs=dict(H=H, episodes=episodes, shots=shots, futures=futures),
            max_workers=args.workers if args else None,
            progress=True,
        )
        ordered = [per_seed[s] for s in sorted(per_seed)]
        c_rewards = [r["classical_mean_reward"] for r in ordered]
        c_succ    = [r["classical_success_rate"] for r in ordered]
        q_rewards = [r["quantum_mean_reward"] for r in ordered]
        q_succ    = [r["quantum_success_rate"] for r in ordered]

        for s_idx, seed in enumerate(seeds):
            print(f"  Seed {seed} -> Class R: {c_rewards[s_idx]:.4f}, Quant R: {q_rewards[s_idx]:.4f}")

        c_r, q_r = np.array(c_rewards), np.array(q_rewards)
        t_stat, p_val_t = stats.ttest_rel(q_r, c_r)
        d = calculate_cohens_d(q_r, c_r)
        p_val_bonf = min(1.0, float(p_val_t) * len(horizons))

        results["scaling_experiments"][f"H{H}"] = {
            "horizon": H,
            "qubits": qubits,
            "candidate_futures": futures,
            "classical_mean_reward": float(np.mean(c_r)),
            "classical_std_reward": float(np.std(c_r)),
            "classical_ci95": float(get_ci95(np.std(c_r), num_seeds)),
            "quantum_mean_reward": float(np.mean(q_r)),
            "quantum_std_reward": float(np.std(q_r)),
            "quantum_ci95": float(get_ci95(np.std(q_r), num_seeds)),
            "classical_success_rate": float(np.mean(c_succ)),
            "quantum_success_rate": float(np.mean(q_succ)),
            "paired_t_stat": float(t_stat),
            "p_val_t": float(p_val_t),
            "p_val_bonferroni": float(p_val_bonf),
            "cohens_d": float(d)
        }

        print(f"\n[SUMMARY H={H}] Class Mean: {np.mean(c_r):.4f} +/- {get_ci95(np.std(c_r)):.4f}, Quant Mean: {np.mean(q_r):.4f} +/- {get_ci95(np.std(q_r)):.4f}")
        print(f"            Cohen's d: {d:.4f}, Bonferroni p-val: {p_val_bonf:.4e}")

    # 2. Ablation Study at H=4
    print("\n--- Phase 2: Ablation Study at H=4 (20 Seeds) ---")
    ablation_configs = {
        "Full_QWM": {"enable_oracle": True, "enable_interference": True},
        "No_Oracle": {"enable_oracle": False, "enable_interference": True},
        "No_Interference": {"enable_oracle": True, "enable_interference": False},
        "No_Oracle_No_Interference": {"enable_oracle": False, "enable_interference": False}
    }

    for name, config in ablation_configs.items():
        print(f"\nAblation Variant: {name}...")
        per_seed = run_seeds_parallel(
            task_fn=_ablation_seed_task,
            seeds=seeds,
            task_kwargs=dict(
                enable_oracle=config["enable_oracle"],
                enable_interference=config["enable_interference"],
                episodes=episodes,
                shots=shots,
            ),
            max_workers=args.workers,
            progress=True,
        )
        q_rewards = [per_seed[s]["mean_reward"] for s in sorted(per_seed)]
        q_succ    = [per_seed[s]["success_rate"] for s in sorted(per_seed)]

        q_r = np.array(q_rewards)
        results["ablation_experiments_H4"][name] = {
            "config": config,
            "mean_reward": float(np.mean(q_r)),
            "std_reward": float(np.std(q_r)),
            "ci95": float(get_ci95(np.std(q_r), num_seeds)),
            "success_rate": float(np.mean(q_succ))
        }
        print(f"Variant {name} -> Mean Reward: {np.mean(q_r):.4f} +/- {get_ci95(np.std(q_r)):.4f} | Succ: {np.mean(q_succ)*100:.1f}%")

    # Save Canonical JSON
    os.makedirs("artifacts", exist_ok=True)
    is_smoke_test = len(horizons) == 1
    if is_smoke_test:
        json_path = "artifacts/canonical_smoke_test.json"
    else:
        json_path = "artifacts/canonical_results.json"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSuccessfully saved canonical results JSON to: {json_path}")

    if is_smoke_test:
        print("Smoke test complete (single horizon). Preserving main figures and canonical_results.json.")
        return

    # 3. Regenerate Figures from Canonical JSON
    fig_dir = "artifacts/figures"
    os.makedirs(fig_dir, exist_ok=True)

    h_vals = horizons

    # Figure 1: Rewards H=2..6
    plt.figure(figsize=(8, 5))
    c_means = [results["scaling_experiments"][f"H{h}"]["classical_mean_reward"] for h in h_vals]
    q_means = [results["scaling_experiments"][f"H{h}"]["quantum_mean_reward"] for h in h_vals]
    c_errs = [results["scaling_experiments"][f"H{h}"]["classical_ci95"] for h in h_vals]
    q_errs = [results["scaling_experiments"][f"H{h}"]["quantum_ci95"] for h in h_vals]

    plt.errorbar(h_vals, c_means, yerr=c_errs, fmt='-o', color='blue', label='Classical Baseline (Tabular Q)', capsize=4, linewidth=2)
    plt.errorbar(h_vals, q_means, yerr=q_errs, fmt='-s', color='purple', label='Quantum Agent (QWM)', capsize=4, linewidth=2)
    plt.title("Reward Scaling across Planning Horizons H=2..6 (N=20 Seeds)", fontsize=11, fontweight='bold')
    plt.xlabel("Planning Horizon H (Qubits n = 2H)", fontsize=10)
    plt.ylabel("Mean Cumulative Reward (95% CI)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "canon_fig1_rewards_H2_H6.png"), dpi=300)
    plt.close()

    # Figure 2: Success Rates H=2..6
    plt.figure(figsize=(8, 5))
    c_succ_vals = [results["scaling_experiments"][f"H{h}"]["classical_success_rate"] * 100 for h in h_vals]
    q_succ_vals = [results["scaling_experiments"][f"H{h}"]["quantum_success_rate"] * 100 for h in h_vals]

    plt.plot(h_vals, c_succ_vals, '-o', color='blue', label='Classical Baseline Success Rate', linewidth=2)
    plt.plot(h_vals, q_succ_vals, '-s', color='purple', label='Quantum Agent Success Rate', linewidth=2)
    plt.title("Goal Success Rate (%) across Horizons H=2..6", fontsize=11, fontweight='bold')
    plt.xlabel("Planning Horizon H", fontsize=10)
    plt.ylabel("Goal Success Rate (%)", fontsize=10)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "canon_fig2_success_rates_H2_H6.png"), dpi=300)
    plt.close()

    # Figure 3: Ablation Bar Chart at H=4
    plt.figure(figsize=(8, 5))
    abl_names = list(results["ablation_experiments_H4"].keys())
    abl_means = [results["ablation_experiments_H4"][k]["mean_reward"] for k in abl_names]
    abl_errs = [results["ablation_experiments_H4"][k]["ci95"] for k in abl_names]
    colors = ['purple', 'crimson', 'orange', 'gray']
    plt.bar(abl_names, abl_means, yerr=abl_errs, color=colors, capsize=5, alpha=0.85)
    plt.title("Ablation Study at Horizon H=4 (256 Candidates, N=20 Seeds)", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Cumulative Reward (95% CI)", fontsize=10)
    plt.xticks(rotation=15)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "canon_fig3_ablation_H4_study.png"), dpi=300)
    plt.close()

    # Figure 6: Cohen's d vs Horizon H=2..6
    plt.figure(figsize=(8, 5))
    d_vals = [results["scaling_experiments"][f"H{h}"]["cohens_d"] for h in h_vals]
    plt.plot(h_vals, d_vals, '-d', color='darkgreen', linewidth=2.5, markersize=8, label="Cohen's d")
    plt.axhline(0.0, color='gray', linestyle='--', label='No Advantage Threshold (d = 0)')
    plt.title("Cohen's d Effect Size vs. Horizon H=2..6", fontsize=11, fontweight='bold')
    plt.xlabel("Planning Horizon H", fontsize=10)
    plt.ylabel("Cohen's d (Quantum vs Classical)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "canon_fig6_cohens_d_vs_horizon.png"), dpi=300)
    plt.close()

    print(f"\nRegenerated figures in {fig_dir}:")
    for fig_file in sorted(os.listdir(fig_dir)):
        print(f"  - {os.path.join(fig_dir, fig_file)}")

    elapsed = time.time() - start_time
    print(f"\nCanonical suite completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_canonical_suite()
