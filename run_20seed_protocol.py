import numpy as np
import scipy.stats as stats
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.environment.gridworld import GridWorld
from src.baselines.classical_agent import ClassicalAgent
from src.quantum.quantum_agent import QuantumAgent

# Module-level defaults. Override via `import run_20seed_protocol as m; m.num_seeds = N; ...; m.main()`
# for smoke tests. Do not shadow these names inside main().
num_seeds = 20
horizons = [2, 3, 4, 5, 6]
episodes = 50
shots = 128
grid_size = 6

def calculate_cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / s_pooled

def calculate_ci_diff(x, y, confidence=0.95):
    diff = x - y
    mean_d = np.mean(diff)
    sem_d = stats.sem(diff)
    h = sem_d * stats.t.ppf((1 + confidence) / 2., len(diff) - 1)
    return mean_d - h, mean_d + h

def main():
    """
    Rigorous 20-seed benchmark protocol.

    All parameters (num_seeds, horizons, episodes, shots, grid_size) come from
    module-level constants. Override for smoke tests via
    `import run_20seed_protocol as m; m.num_seeds = 2; m.main()`.
    """
    if num_seeds < 2:
        raise ValueError(
            f"Paired t-tests require num_seeds >= 2, got {num_seeds}. "
            "Cannot compute p-values or Cohen's d with a single seed. "
            "If you want a smoke test, set num_seeds = 2 (not 1)."
        )

    if num_seeds != 20:
        print(
            f"[WARN] Running with num_seeds={num_seeds} (canonical is 20). "
            f"Results will be written to *_results_{num_seeds}seeds.json and figures "
            f"will carry '(N={num_seeds} Seeds)' titles. Do not compare directly "
            f"against canonical 20-seed benchmark results."
        )

    print("=========================================================================")
    print(f"  RIGOROUS {num_seeds}-SEED BENCHMARK PROTOCOL (H=2..6, BONFERRONI CORRECTION)   ")
    print("=========================================================================")
    
    bonferroni_alpha = 0.05 / len(horizons)  # alpha_adj = 0.01
    
    results = {
        "scaling_experiments": {},
        "ablation_experiments_H4": {},
        "learning_curves_H4": {
            "classical_reward_curve": [],
            "quantum_reward_curve": [],
            "classical_success_curve": [],
            "quantum_success_curve": []
        }
    }

    # Track learning curves for H=4 across 20 seeds
    c_ep_rewards_matrix = np.zeros((num_seeds, episodes))
    q_ep_rewards_matrix = np.zeros((num_seeds, episodes))
    c_ep_succ_matrix = np.zeros((num_seeds, episodes))
    q_ep_succ_matrix = np.zeros((num_seeds, episodes))

    # 1. Scaling Experiments H=2..6 across seeds
    print(f"\n--- Running {num_seeds}-Seed Horizon Scaling Experiments (H=2..6) ---")
    for H in horizons:
        qubits = 2 * H
        futures = 4**H
        print(f"\nEvaluating Horizon H={H} ({futures} Candidate Futures, {qubits} Qubits, N={num_seeds} Seeds)...")
        
        c_rewards, q_rewards = [], []
        c_succ, q_succ = [], []
        c_steps, q_steps = [], []

        for seed_idx, seed in enumerate(range(100, 100 + num_seeds)):
            # Classical Agent
            np.random.seed(seed)
            env = GridWorld(width=grid_size, height=grid_size, start_pos=(0, 0), goal_pos=(5, 5))
            c_agent = ClassicalAgent(env=env, horizon=H, eval_budget=futures)
            
            c_ep_r, c_ep_s, c_ep_steps = [], [], []
            for ep in range(episodes):
                pos = env.reset()
                total_r = 0.0
                done = False
                step_cnt = 0
                while not done:
                    step_cnt += 1
                    a = c_agent.select_action(pos)
                    next_p, r, done, info = env.step(a)
                    c_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                c_ep_r.append(total_r)
                c_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
                c_ep_steps.append(step_cnt)
                if H == 4:
                    c_ep_rewards_matrix[seed_idx, ep] = total_r
                    c_ep_succ_matrix[seed_idx, ep] = c_ep_s[-1]

            c_rewards.append(np.mean(c_ep_r))
            c_succ.append(float(np.mean(c_ep_s)))
            c_steps.append(float(np.mean(c_ep_steps)))

            # Quantum Agent
            np.random.seed(seed)
            env = GridWorld(width=grid_size, height=grid_size, start_pos=(0, 0), goal_pos=(5, 5))
            q_agent = QuantumAgent(env=env, horizon=H, shots=shots)
            
            q_ep_r, q_ep_s, q_ep_steps = [], [], []
            for ep in range(episodes):
                pos = env.reset()
                total_r = 0.0
                done = False
                step_cnt = 0
                while not done:
                    step_cnt += 1
                    a = q_agent.select_action(pos)
                    next_p, r, done, info = env.step(a)
                    q_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                q_ep_r.append(total_r)
                q_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
                q_ep_steps.append(step_cnt)
                if H == 4:
                    q_ep_rewards_matrix[seed_idx, ep] = total_r
                    q_ep_succ_matrix[seed_idx, ep] = q_ep_s[-1]

            q_rewards.append(np.mean(q_ep_r))
            q_succ.append(float(np.mean(q_ep_s)))
            q_steps.append(float(np.mean(q_ep_steps)))

        c_r, q_r = np.array(c_rewards), np.array(q_rewards)
        t_stat, p_val_t = stats.ttest_rel(q_r, c_r)
        w_stat, p_val_w = stats.wilcoxon(q_r, c_r)
        d = calculate_cohens_d(q_r, c_r)
        ci_low, ci_high = calculate_ci_diff(q_r, c_r)
        bonf_sig = p_val_t < bonferroni_alpha

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
            "mean_diff": float(np.mean(q_r - c_r)),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "paired_t_stat": float(t_stat),
            "p_val_uncorrected": float(p_val_t),
            "p_val_wilcoxon": float(p_val_w),
            "bonferroni_threshold": bonferroni_alpha,
            "bonferroni_significant": bool(bonf_sig),
            "cohens_d": float(d),
            "classical_mean_steps": float(np.mean(c_steps)),
            "quantum_mean_steps": float(np.mean(q_steps)),
        }
        
        print(f"H={H} ({qubits}Q) -> Class Reward: {np.mean(c_r):.4f} +/- {np.std(c_r):.4f}, Quant Reward: {np.mean(q_r):.4f} +/- {np.std(q_r):.4f}")
        print(
            f"         -> Class Succ: {np.mean(c_succ)*100:.1f}%, Quant Succ: {np.mean(q_succ)*100:.1f}% "
            f"| Class Steps: {np.mean(c_steps):.1f}, Quant Steps: {np.mean(q_steps):.1f} "
            f"| p-uncorrected: {p_val_t:.4e}, Bonf Sig: {bonf_sig}, d: {d:.4f}"
        )

    # 2. Store H=4 Learning Curves
    c_rc_mean = np.mean(c_ep_rewards_matrix, axis=0).tolist()
    q_rc_mean = np.mean(q_ep_rewards_matrix, axis=0).tolist()
    c_sc_mean = np.mean(c_ep_succ_matrix, axis=0).tolist()
    q_sc_mean = np.mean(q_ep_succ_matrix, axis=0).tolist()

    results["learning_curves_H4"]["classical_reward_curve"] = c_rc_mean
    results["learning_curves_H4"]["quantum_reward_curve"] = q_rc_mean
    results["learning_curves_H4"]["classical_success_curve"] = c_sc_mean
    results["learning_curves_H4"]["quantum_success_curve"] = q_sc_mean

    # Episodes to reach 90% performance at H=4
    c_max_r = np.max(c_rc_mean)
    q_max_r = np.max(q_rc_mean)
    c_ep_90 = next((idx+1 for idx, val in enumerate(c_rc_mean) if val >= 0.9 * c_max_r), episodes)
    q_ep_90 = next((idx+1 for idx, val in enumerate(q_rc_mean) if val >= 0.9 * q_max_r), episodes)

    results["learning_curves_H4"]["classical_episodes_to_90"] = c_ep_90
    results["learning_curves_H4"]["quantum_episodes_to_90"] = q_ep_90

    print(f"\n--- H=4 Convergence Analysis ---")
    print(f"Classical Episodes to 90% Max Performance: {c_ep_90}")
    print(f"Quantum Episodes to 90% Max Performance:   {q_ep_90}")

    # 3. Ablation Studies at H=4
    print(f"\n--- Running {num_seeds}-Seed Ablation Studies at H=4 (256 Futures, 8 Qubits) ---")
    ablation_configs = {
        "Full_QWM": {"enable_oracle": True, "enable_interference": True},
        "No_Oracle": {"enable_oracle": False, "enable_interference": True},
        "No_Interference": {"enable_oracle": True, "enable_interference": False},
        "No_Oracle_No_Interference": {"enable_oracle": False, "enable_interference": False}
    }

    for name, config in ablation_configs.items():
        q_rewards, q_succ = [], []
        for seed in range(100, 100 + num_seeds):
            np.random.seed(seed)
            env = GridWorld(width=grid_size, height=grid_size, start_pos=(0, 0), goal_pos=(5, 5))
            q_agent = QuantumAgent(
                env=env,
                horizon=4,
                shots=shots,
                enable_oracle=config["enable_oracle"],
                enable_interference=config["enable_interference"]
            )
            q_ep_r, q_ep_s = [], []
            for ep in range(episodes):
                pos = env.reset()
                total_r = 0.0
                done = False
                while not done:
                    a = q_agent.select_action(pos)
                    next_p, r, done, info = env.step(a)
                    q_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                q_ep_r.append(total_r)
                q_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            q_rewards.append(np.mean(q_ep_r))
            q_succ.append(float(np.mean(q_ep_s)))

        q_r = np.array(q_rewards)
        results["ablation_experiments_H4"][name] = {
            "config": config,
            "mean_reward": float(np.mean(q_r)),
            "std_reward": float(np.std(q_r)),
            "success_rate": float(np.mean(q_succ))
        }
        print(f"Variant {name} -> Mean Reward: {np.mean(q_r):.4f} +/- {np.std(q_r):.4f} | Success Rate: {np.mean(q_succ)*100:.1f}%")

    # Save JSON data
    os.makedirs("artifacts", exist_ok=True)
    results["metadata"] = {
        "num_seeds": num_seeds,
        "canonical_num_seeds": 20,
        "is_canonical_seed_count": bool(num_seeds == 20),
    }
    json_path = f"artifacts/scaling_and_ablation_results_{num_seeds}seeds.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {num_seeds}-seed structured results: {json_path}")

    if num_seeds != 20:
        print(
            f"[WARN] Skipping publication figure generation: num_seeds={num_seeds}, "
            f"canonical requires 20. JSON results are still saved."
        )
        # Skip figure generation but return cleanly
        print(f"\nSkipped figure generation (num_seeds={num_seeds} != 20).")
        return

    # 4. Generate 5 Publication Figures (300 DPI)
    fig_dir = "artifacts/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # Uniform random reference baseline value at H=4
    uniform_ref_reward = results["ablation_experiments_H4"]["No_Oracle_No_Interference"]["mean_reward"]

    # Figure 1: Reward vs. Horizon (H=2..6)
    plt.figure(figsize=(8, 5))
    h_vals = horizons
    c_means = [results["scaling_experiments"][f"H{h}"]["classical_mean_reward"] for h in h_vals]
    q_means = [results["scaling_experiments"][f"H{h}"]["quantum_mean_reward"] for h in h_vals]
    c_errs = [results["scaling_experiments"][f"H{h}"]["classical_std_reward"] for h in h_vals]
    q_errs = [results["scaling_experiments"][f"H{h}"]["quantum_std_reward"] for h in h_vals]

    plt.errorbar(h_vals, c_means, yerr=c_errs, fmt='-o', color='blue', label='Classical Baseline', capsize=4, linewidth=2)
    plt.errorbar(h_vals, q_means, yerr=q_errs, fmt='-s', color='purple', label='Quantum World Model Agent', capsize=4, linewidth=2)
    plt.axhline(uniform_ref_reward, color='red', linestyle='--', label='Uniform Random Baseline')
    plt.title(f"Reward Performance Scaling across Planning Horizons H=2..6 (N={num_seeds} Seeds)", fontsize=12, fontweight='bold')
    plt.xlabel("Planning Horizon H (Qubits = 2H)", fontsize=10)
    plt.ylabel("Mean Cumulative Reward", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "p20_fig1_rewards_H2_H6.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Figure 2: Success Rate vs. Horizon (H=2..6)
    plt.figure(figsize=(8, 5))
    c_succ_vals = [results["scaling_experiments"][f"H{h}"]["classical_success_rate"] * 100 for h in h_vals]
    q_succ_vals = [results["scaling_experiments"][f"H{h}"]["quantum_success_rate"] * 100 for h in h_vals]

    plt.plot(h_vals, c_succ_vals, '-o', color='blue', label='Classical Baseline Success Rate', linewidth=2)
    plt.plot(h_vals, q_succ_vals, '-s', color='purple', label='Quantum Agent Success Rate', linewidth=2)
    plt.title(f"Goal Success Rate (%) across Horizons H=2..6 (N={num_seeds} Seeds)", fontsize=12, fontweight='bold')
    plt.xlabel("Planning Horizon H (Qubits = 2H)", fontsize=10)
    plt.ylabel("Goal Success Rate (%)", fontsize=10)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, "p20_fig2_success_rates_H2_H6.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # Figure 3: Ablation Bar Chart at H=4
    plt.figure(figsize=(8, 5))
    abl_names = list(results["ablation_experiments_H4"].keys())
    abl_means = [results["ablation_experiments_H4"][k]["mean_reward"] for k in abl_names]
    abl_errs = [results["ablation_experiments_H4"][k]["std_reward"] for k in abl_names]

    colors = ['purple', 'crimson', 'orange', 'gray']
    plt.bar(abl_names, abl_means, yerr=abl_errs, color=colors, capsize=5, alpha=0.85)
    plt.axhline(uniform_ref_reward, color='red', linestyle='--', label='Uniform Random Reference Line')
    plt.title(f"Ablation Study at Horizon H=4 (256 Candidate Futures, 8 Qubits, N={num_seeds} Seeds)", fontsize=12, fontweight='bold')
    plt.ylabel("Mean Cumulative Reward", fontsize=10)
    plt.xticks(rotation=15)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.legend(loc='upper right')
    plt.tight_layout()
    fig3_path = os.path.join(fig_dir, "p20_fig3_ablation_H4_study.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # Figure 4: Learning Curves at H=4 (Reward vs Episode)
    plt.figure(figsize=(8, 5))
    plt.plot(c_rc_mean, color='blue', linewidth=2.5, label='Classical Baseline (MA Episode Reward)')
    plt.plot(q_rc_mean, color='purple', linewidth=2.5, label='Quantum Agent (MA Episode Reward)')
    plt.title(f"Learning Curves at Horizon H=4: Cumulative Reward vs. Episode (N={num_seeds} Seeds)", fontsize=12, fontweight='bold')
    plt.xlabel("Training Episode", fontsize=10)
    plt.ylabel("Mean Reward", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    fig4_path = os.path.join(fig_dir, "p20_fig4_learning_reward_H4.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # Figure 5: Learning Curves at H=4 (Success Rate vs Episode)
    plt.figure(figsize=(8, 5))
    plt.plot(np.array(c_sc_mean) * 100, color='blue', linewidth=2.5, label='Classical Baseline Success Rate')
    plt.plot(np.array(q_sc_mean) * 100, color='purple', linewidth=2.5, label='Quantum Agent Success Rate')
    plt.title(f"Learning Curves at Horizon H=4: Success Rate (%) vs. Episode (N={num_seeds} Seeds)", fontsize=12, fontweight='bold')
    plt.xlabel("Training Episode", fontsize=10)
    plt.ylabel("Success Rate (%)", fontsize=10)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    fig5_path = os.path.join(fig_dir, "p20_fig5_learning_success_H4.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    print(f"\nGenerated 5 Publication Figures in {fig_dir}:")
    print(f" - {fig1_path}")
    print(f" - {fig2_path}")
    print(f" - {fig3_path}")
    print(f" - {fig4_path}")
    print(f" - {fig5_path}")

if __name__ == "__main__":
    main()
