import numpy as np
import scipy.stats as stats
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.evaluation.benchmark import BenchmarkRunner

def calculate_cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / s_pooled

def main():
    print("=========================================================================")
    print("  FIXED CODEBASE: SCALABILITY & ABLATION EXPERIMENTAL RUNNER (H=2..6)    ")
    print("=========================================================================")
    
    num_seeds = 20
    horizons = [2, 3, 4, 5, 6]
    grid_size = 6
    shots = 128
    episodes = 50  # Restored full 50 training episodes
    
    results = {
        "scaling_experiments": {},
        "ablation_experiments_H4": {}
    }

    # 1. Scaling Experiments H=2..6
    print("\n--- Running Fixed Horizon Scaling Experiments (H=2..6, 20 Seeds) ---")
    for H in horizons:
        qubits = 2 * H
        futures = 4**H
        print(f"\nEvaluating Horizon H={H} ({futures} Candidate Futures, {qubits} Qubits)...")
        
        c_rewards, q_rewards = [], []
        c_succ, q_succ = [], []
        c_steps, q_steps = [], []

        for seed in range(100, 100 + num_seeds):
            runner = BenchmarkRunner(
                grid_width=grid_size,
                grid_height=grid_size,
                horizon=H,
                episodes=episodes,
                eval_budget=futures,
                shots=shots,
                seed=seed
            )
            res = runner.run_benchmark()
            c_m, q_m = res["classical_metrics"], res["quantum_metrics"]

            c_rewards.append(c_m.mean_reward)
            q_rewards.append(q_m.mean_reward)
            c_succ.append(c_m.success_rates)
            q_succ.append(q_m.success_rates)
            c_steps.append(c_m.avg_steps)
            q_steps.append(q_m.avg_steps)

        c_r, q_r = np.array(c_rewards), np.array(q_rewards)
        t_stat, p_val_t = stats.ttest_rel(q_r, c_r)
        d = calculate_cohens_d(q_r, c_r)

        results["scaling_experiments"][f"H{H}"] = {
            "horizon": H,
            "qubits": qubits,
            "candidate_futures": futures,
            "classical_mean_reward": float(np.mean(c_r)),
            "classical_std_reward": float(np.std(c_r)),
            "quantum_mean_reward": float(np.mean(q_r)),
            "quantum_std_reward": float(np.std(q_r)),
            "classical_success_rate": float(np.mean(c_succ)),
            "quantum_success_rate": float(np.mean(q_succ)),
            "classical_avg_steps": float(np.mean(c_steps)),
            "quantum_avg_steps": float(np.mean(q_steps)),
            "paired_t_stat": float(t_stat),
            "p_val_t": float(p_val_t),
            "cohens_d": float(d)
        }
        
        print(f"H={H} ({qubits}Q) -> Class Reward: {np.mean(c_r):.4f} +/- {np.std(c_r):.4f}, Quant Reward: {np.mean(q_r):.4f} +/- {np.std(q_r):.4f}")
        print(f"         -> Class Succ: {np.mean(c_succ)*100:.1f}%, Quant Succ: {np.mean(q_succ)*100:.1f}% | p-val: {p_val_t:.4e}, d: {d:.4f}")

    # 2. Ablation Studies at H=4 (256 Candidate Futures, 8 Qubits)
    print("\n--- Running Ablation Studies at H=4 (256 Futures, 8 Qubits) ---")
    ablation_configs = {
        "Full_QWM": {"enable_oracle": True, "enable_interference": True},
        "No_Oracle": {"enable_oracle": False, "enable_interference": True},
        "No_Interference": {"enable_oracle": True, "enable_interference": False},
        "No_Oracle_No_Interference": {"enable_oracle": False, "enable_interference": False}
    }

    for name, config in ablation_configs.items():
        print(f"Running Ablation Variant: {name}...")
        q_rewards, q_succ, q_steps = [], [], []

        for seed in range(100, 100 + num_seeds):
            runner = BenchmarkRunner(
                grid_width=grid_size,
                grid_height=grid_size,
                horizon=4,
                episodes=episodes,
                shots=shots,
                seed=seed,
                enable_oracle=config["enable_oracle"],
                enable_interference=config["enable_interference"]
            )
            q_m = runner.run_agent_experiment("quantum", seed=seed)
            q_rewards.append(q_m.mean_reward)
            q_succ.append(q_m.success_rates)
            q_steps.append(q_m.avg_steps)

        q_r = np.array(q_rewards)
        results["ablation_experiments_H4"][name] = {
            "config": config,
            "mean_reward": float(np.mean(q_r)),
            "std_reward": float(np.std(q_r)),
            "success_rate": float(np.mean(q_succ)),
            "avg_steps": float(np.mean(q_steps))
        }
        print(f"Variant {name} -> Mean Reward: {np.mean(q_r):.4f} +/- {np.std(q_r):.4f} | Success Rate: {np.mean(q_succ)*100:.1f}%")

    # Save JSON data
    os.makedirs("artifacts", exist_ok=True)
    json_path = "artifacts/scaling_and_ablation_results_6x6.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured experimental results: {json_path}")

    # Plot Horizon Scaling Performance
    fig_dir = "artifacts/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    h_vals = horizons
    c_means = [results["scaling_experiments"][f"H{h}"]["classical_mean_reward"] for h in h_vals]
    q_means = [results["scaling_experiments"][f"H{h}"]["quantum_mean_reward"] for h in h_vals]
    c_errs = [results["scaling_experiments"][f"H{h}"]["classical_std_reward"] for h in h_vals]
    q_errs = [results["scaling_experiments"][f"H{h}"]["quantum_std_reward"] for h in h_vals]

    plt.errorbar(h_vals, c_means, yerr=c_errs, fmt='-o', color='blue', label='Classical Baseline', capsize=4, linewidth=2)
    plt.errorbar(h_vals, q_means, yerr=q_errs, fmt='-s', color='purple', label='Quantum Agent (Fixed Grover Oracle)', capsize=4, linewidth=2)
    plt.title("Reward Performance Scaling across Horizons H=2..6 (Fixed Codebase, 6x6 Grid)", fontsize=12, fontweight='bold')
    plt.xlabel("Planning Horizon H (Qubits = 2H)", fontsize=10)
    plt.ylabel("Mean Cumulative Reward", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "saa_scaling_rewards_6x6_H2_H6.png"), dpi=300)
    plt.close()

    # Plot Ablation Bar Chart
    plt.figure(figsize=(8, 5))
    abl_names = list(results["ablation_experiments_H4"].keys())
    abl_means = [results["ablation_experiments_H4"][k]["mean_reward"] for k in abl_names]
    abl_errs = [results["ablation_experiments_H4"][k]["std_reward"] for k in abl_names]

    colors = ['purple', 'crimson', 'orange', 'gray']
    plt.bar(abl_names, abl_means, yerr=abl_errs, color=colors, capsize=5, alpha=0.85)
    plt.title("Ablation Study at Horizon H=4 (Fixed Codebase, 256 Futures, 6x6 Grid)", fontsize=12, fontweight='bold')
    plt.ylabel("Mean Cumulative Reward", fontsize=10)
    plt.xticks(rotation=15)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "saa_ablation_6x6_H4_study.png"), dpi=300)
    plt.close()

    print(f"Saved figures to {fig_dir}: saa_scaling_rewards_6x6_H2_H6.png, saa_ablation_6x6_H4_study.png")

if __name__ == "__main__":
    main()
