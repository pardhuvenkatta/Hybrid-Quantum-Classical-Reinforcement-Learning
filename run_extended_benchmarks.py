import numpy as np
import scipy.stats as stats
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.environment.gridworld import GridWorld, ACTIONS
from src.baselines.classical_agent import ClassicalAgent
from src.baselines.mcts_agent import MCTSAgent
from src.baselines.zeng_action_search import ZengActionSearchAgent
from src.baselines.deep_rl_agent import DeepRLAgent
from src.quantum.quantum_agent import QuantumAgent
from src.quantum.qiskit_hardware_model import QiskitHardwareSimulator

def calculate_cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / s_pooled

def get_ci95(std_val, N=20):
    return 1.96 * (std_val / np.sqrt(N))

def run_extended_protocol():
    start_time = time.time()
    print("=========================================================================")
    print("  EXTENDED BENCHMARK & NOISE ROBUSTNESS PROTOCOL (H=1..6, 20 SEEDS)     ")
    print("=========================================================================")
    
    num_seeds = 20
    horizons = [1, 2, 3, 4, 5, 6]
    grid_size = 6
    shots = 128
    episodes = 50

    results = {
        "metadata": {
            "num_seeds": num_seeds,
            "horizons": horizons,
            "grid_size": grid_size,
            "episodes": episodes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "scaling_experiments": {},
        "mcts_baseline": {},
        "deep_rl_baseline": {},
        "zeng_baseline_H4": {},
        "ablation_experiments_H4": {},
        "noise_robustness": {}
    }

    # 1. Horizon Scaling Experiments H=1..6 (Classical vs QWM across 20 seeds)
    print("\n--- Phase 1: Horizon Scaling Experiments H=1..6 (20 Seeds) ---")
    for H in horizons:
        qubits = 2 * H
        futures = 4**H
        print(f"\nEvaluating Horizon H={H} ({futures} Candidate Futures, {qubits} Qubits)...")

        c_rewards, q_rewards = [], []
        c_succ, q_succ = [], []

        for seed in range(100, 100 + num_seeds):
            # Classical Agent
            np.random.seed(seed)
            env_c = GridWorld(width=grid_size, height=grid_size, start_pos=(0, 0), goal_pos=(5, 5))
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
            c_rewards.append(np.mean(c_ep_r))
            c_succ.append(float(np.mean(c_ep_s)))

            # Quantum Agent
            np.random.seed(seed)
            env_q = GridWorld(width=grid_size, height=grid_size, start_pos=(0, 0), goal_pos=(5, 5))
            q_agent = QuantumAgent(env=env_q, horizon=H, shots=shots)
            q_ep_r, q_ep_s = [], []
            for ep in range(episodes):
                pos = env_q.reset()
                total_r = 0.0
                done = False
                while not done:
                    a = q_agent.select_action(pos)
                    next_p, r, done, info = env_q.step(a)
                    q_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                q_ep_r.append(total_r)
                q_ep_s.append(1.0 if info.get("reached_goal", False) else 0.0)
            q_rewards.append(np.mean(q_ep_r))
            q_succ.append(float(np.mean(q_ep_s)))

        c_r, q_r = np.array(c_rewards), np.array(q_rewards)
        t_stat, p_val_t = stats.ttest_rel(q_r, c_r)
        d = calculate_cohens_d(q_r, c_r)
        p_val_bonf = min(1.0, float(p_val_t) * 6.0)

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

        print(f"H={H} ({qubits}Q) -> Class Reward: {np.mean(c_r):.4f} +/- {get_ci95(np.std(c_r)):.4f}, Quant Reward: {np.mean(q_r):.4f} +/- {get_ci95(np.std(q_r)):.4f}")
        print(f"         -> Class Succ: {np.mean(c_succ)*100:.1f}%, Quant Succ: {np.mean(q_succ)*100:.1f}% | Bonf p: {p_val_bonf:.4e}, d: {d:.4f}")

    # 2. Stronger Classical Baseline: MCTS (H=3..6, 20 Seeds)
    print("\n--- Phase 2: Stronger Classical Baseline - MCTS with UCB1 (20 Seeds) ---")
    for H in [3, 4, 5, 6]:
        futures = 4**H
        print(f"Running MCTS at H={H} (Budget = {futures} simulations)...")
        mcts_rewards, mcts_succ = [], []
        for seed in range(100, 100 + num_seeds):
            np.random.seed(seed)
            env = GridWorld(width=grid_size, height=grid_size, start_pos=(0, 0), goal_pos=(5, 5))
            agent = MCTSAgent(env=env, horizon=H, eval_budget=futures)
            ep_rewards, ep_succ = [], []
            for ep in range(episodes):
                state_pos = env.reset()
                total_r = 0.0
                done = False
                steps = 0
                while not done and steps < 30:
                    a = agent.select_action(state_pos)
                    next_pos, r, done, info = env.step(a)
                    total_r += r
                    state_pos = next_pos
                    steps += 1
                ep_rewards.append(total_r)
                ep_succ.append(1.0 if info.get("reached_goal", False) else 0.0)
            mcts_rewards.append(np.mean(ep_rewards))
            mcts_succ.append(float(np.mean(ep_succ)))

        m_r = np.array(mcts_rewards)
        results["mcts_baseline"][f"H{H}"] = {
            "horizon": H,
            "mean_reward": float(np.mean(m_r)),
            "std_reward": float(np.std(m_r)),
            "ci95": float(get_ci95(np.std(m_r), num_seeds)),
            "success_rate": float(np.mean(mcts_succ))
        }
        print(f"MCTS H={H} -> Mean Reward: {np.mean(m_r):.4f} +/- {get_ci95(np.std(m_r)):.4f} | Success Rate: {np.mean(mcts_succ)*100:.1f}%")

    # 3. Deep RL Baseline (20 Seeds)
    print("\n--- Phase 3: Function-Approximated Deep RL Baseline (20 Seeds) ---")
    drl_rewards, drl_succ = [], []
    for seed in range(100, 100 + num_seeds):
        np.random.seed(seed)
        env = GridWorld(width=grid_size, height=grid_size, start_pos=(0, 0), goal_pos=(5, 5))
        agent = DeepRLAgent(env=env)
        ep_rewards, ep_succ = [], []
        for ep in range(episodes):
            state_pos = env.reset()
            total_r = 0.0
            done = False
            steps = 0
            while not done and steps < 30:
                a = agent.select_action(state_pos)
                next_pos, r, done, info = env.step(a)
                agent.update(state_pos, a, r, next_pos, done)
                total_r += r
                state_pos = next_pos
                steps += 1
            ep_rewards.append(total_r)
            ep_succ.append(1.0 if info.get("reached_goal", False) else 0.0)
        drl_rewards.append(np.mean(ep_rewards))
        drl_succ.append(float(np.mean(ep_succ)))

    d_r = np.array(drl_rewards)
    results["deep_rl_baseline"] = {
        "mean_reward": float(np.mean(d_r)),
        "std_reward": float(np.std(d_r)),
        "ci95": float(get_ci95(np.std(d_r), num_seeds)),
        "success_rate": float(np.mean(drl_succ))
    }
    print(f"Deep RL -> Mean Reward: {np.mean(d_r):.4f} +/- {get_ci95(np.std(d_r)):.4f} | Success Rate: {np.mean(drl_succ)*100:.1f}%")

    # 4. Zeng et al. Grover Action Search Baseline at H=4 (20 Seeds)
    print("\n--- Phase 4: Zeng et al. Grover Action-Search Baseline at H=4 (20 Seeds) ---")
    zeng_rewards, zeng_succ = [], []
    for seed in range(100, 100 + num_seeds):
        np.random.seed(seed)
        env = GridWorld(width=grid_size, height=grid_size, start_pos=(0, 0), goal_pos=(5, 5))
        agent = ZengActionSearchAgent(env=env, horizon=4, shots=shots)
        ep_rewards, ep_succ = [], []
        for ep in range(episodes):
            state_pos = env.reset()
            total_r = 0.0
            done = False
            steps = 0
            while not done and steps < 30:
                a = agent.select_action(state_pos)
                next_pos, r, done, info = env.step(a)
                agent.update(state_pos, a, r, next_pos, done)
                total_r += r
                state_pos = next_pos
                steps += 1
            ep_rewards.append(total_r)
            ep_succ.append(1.0 if info.get("reached_goal", False) else 0.0)
        zeng_rewards.append(np.mean(ep_rewards))
        zeng_succ.append(float(np.mean(ep_succ)))

    z_r = np.array(zeng_rewards)
    results["zeng_baseline_H4"] = {
        "mean_reward": float(np.mean(z_r)),
        "std_reward": float(np.std(z_r)),
        "ci95": float(get_ci95(np.std(z_r), num_seeds)),
        "success_rate": float(np.mean(zeng_succ))
    }
    print(f"Zeng Action Search H=4 -> Mean Reward: {np.mean(z_r):.4f} +/- {get_ci95(np.std(z_r)):.4f} | Success Rate: {np.mean(zeng_succ)*100:.1f}%")

    # 5. Ablation Studies at H=4 (20 Seeds)
    print("\n--- Phase 5: Ablation Studies at H=4 (20 Seeds) ---")
    ablation_configs = {
        "Full_QWM": {"enable_oracle": True, "enable_interference": True},
        "No_Oracle": {"enable_oracle": False, "enable_interference": True},
        "No_Interference": {"enable_oracle": True, "enable_interference": False},
        "No_Oracle_No_Interference": {"enable_oracle": False, "enable_interference": False}
    }
    for name, config in ablation_configs.items():
        print(f"Running Ablation Variant: {name}...")
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
            "ci95": float(get_ci95(np.std(q_r), num_seeds)),
            "success_rate": float(np.mean(q_succ))
        }
        print(f"Variant {name} -> Mean Reward: {np.mean(q_r):.4f} +/- {get_ci95(np.std(q_r)):.4f} | Success Rate: {np.mean(q_succ)*100:.1f}%")

    # 6. NISQ Hardware & Noise Robustness Study (H=3, H=4)
    print("\n--- Phase 6: NISQ Hardware Simulation & Noise Robustness Study (H=3, H=4) ---")
    depol_rates = [0.001, 0.005, 0.01, 0.02]
    readout_rates = [0.01, 0.02, 0.05]

    for H in [3, 4]:
        results["noise_robustness"][f"H{H}"] = {"depolarizing": {}, "readout": {}}
        print(f"Noise Robustness Evaluation at H={H}...")
        
        # Depolarizing Noise Sweep
        for dep in depol_rates:
            dep_rewards = []
            for seed in range(100, 100 + 20):
                np.random.seed(seed)
                env = GridWorld(width=grid_size, height=grid_size, start_pos=(0, 0), goal_pos=(5, 5))
                sim = QiskitHardwareSimulator(env=env, horizon=H, shots=1024, depol_error=dep)
                q_agent = QuantumAgent(env=env, horizon=H, shots=shots)
                ep_rewards = []
                for ep in range(episodes):
                    pos = env.reset()
                    total_r, done = 0.0, False
                    while not done:
                        eval_probs = sim.execute_noisy_simulation(q_agent.world_model, pos)
                        action_probs = np.zeros(env.num_actions)
                        for seq, prob in eval_probs.items():
                            action_probs[seq[0] % env.num_actions] += prob
                        total_p = np.sum(action_probs)
                        if total_p > 0:
                            action_probs /= total_p
                            a = int(np.random.choice(range(env.num_actions), p=action_probs))
                        else:
                            a = q_agent.select_action(pos)
                        next_p, r, done, info = env.step(a)
                        q_agent.update(pos, a, r, next_p, done)
                        total_r += r
                        pos = next_p
                    ep_rewards.append(total_r)
                dep_rewards.append(float(np.mean(ep_rewards)))

            results["noise_robustness"][f"H{H}"]["depolarizing"][str(dep)] = {
                "mean_reward": float(np.mean(dep_rewards)),
                "std_reward": float(np.std(dep_rewards)),
                "ci95": float(get_ci95(np.std(dep_rewards), 20))
            }

        # Readout Error Sweep
        for ro in readout_rates:
            ro_rewards = []
            for seed in range(100, 100 + 20):
                np.random.seed(seed)
                env = GridWorld(width=grid_size, height=grid_size, start_pos=(0, 0), goal_pos=(5, 5))
                sim = QiskitHardwareSimulator(env=env, horizon=H, shots=1024, readout_err=ro)
                q_agent = QuantumAgent(env=env, horizon=H, shots=shots)
                ep_rewards = []
                for ep in range(episodes):
                    pos = env.reset()
                    total_r, done = 0.0, False
                    while not done:
                        eval_probs = sim.execute_noisy_simulation(q_agent.world_model, pos)
                        action_probs = np.zeros(env.num_actions)
                        for seq, prob in eval_probs.items():
                            action_probs[seq[0] % env.num_actions] += prob
                        total_p = np.sum(action_probs)
                        if total_p > 0:
                            action_probs /= total_p
                            a = int(np.random.choice(range(env.num_actions), p=action_probs))
                        else:
                            a = q_agent.select_action(pos)
                        next_p, r, done, info = env.step(a)
                        q_agent.update(pos, a, r, next_p, done)
                        total_r += r
                        pos = next_p
                    ep_rewards.append(total_r)
                ro_rewards.append(float(np.mean(ep_rewards)))

            results["noise_robustness"][f"H{H}"]["readout"][str(ro)] = {
                "mean_reward": float(np.mean(ro_rewards)),
                "std_reward": float(np.std(ro_rewards)),
                "ci95": float(get_ci95(np.std(ro_rewards), 20))
            }

    # Save JSON data
    os.makedirs("artifacts", exist_ok=True)
    json_path = "artifacts/scaling_and_ablation_results_extended.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured experimental results: {json_path}")

    # Generate Figures
    fig_dir = "artifacts/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: Rewards H=1..6
    plt.figure(figsize=(8, 5))
    h_vals = horizons
    c_means = [results["scaling_experiments"][f"H{h}"]["classical_mean_reward"] for h in h_vals]
    q_means = [results["scaling_experiments"][f"H{h}"]["quantum_mean_reward"] for h in h_vals]
    c_errs = [results["scaling_experiments"][f"H{h}"]["classical_ci95"] for h in h_vals]
    q_errs = [results["scaling_experiments"][f"H{h}"]["quantum_ci95"] for h in h_vals]

    plt.errorbar(h_vals, c_means, yerr=c_errs, fmt='-o', color='blue', label='Classical Baseline (Tabular Q)', capsize=4, linewidth=2)
    plt.errorbar(h_vals, q_means, yerr=q_errs, fmt='-s', color='purple', label='Quantum Agent (QWM)', capsize=4, linewidth=2)
    plt.axhline(-0.1, color='red', linestyle='--', label='Uniform Random Path Baseline')
    plt.title("Reward Scaling across Horizons H=1..6 (95% CIs, N=20 Seeds)", fontsize=11, fontweight='bold')
    plt.xlabel("Planning Horizon H (Qubits n = 2H)", fontsize=10)
    plt.ylabel("Mean Cumulative Reward", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ext_fig1_rewards_H2_H6.png"), dpi=300)
    plt.close()

    # Figure 2: Success Rates H=1..6
    plt.figure(figsize=(8, 5))
    c_succ_vals = [results["scaling_experiments"][f"H{h}"]["classical_success_rate"] * 100 for h in h_vals]
    q_succ_vals = [results["scaling_experiments"][f"H{h}"]["quantum_success_rate"] * 100 for h in h_vals]

    plt.plot(h_vals, c_succ_vals, '-o', color='blue', label='Classical Baseline', linewidth=2)
    plt.plot(h_vals, q_succ_vals, '-s', color='purple', label='Quantum Agent (QWM)', linewidth=2)
    plt.title("Goal Success Rate (%) across Horizons H=1..6", fontsize=11, fontweight='bold')
    plt.xlabel("Planning Horizon H", fontsize=10)
    plt.ylabel("Goal Success Rate (%)", fontsize=10)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ext_fig2_success_rates_H2_H6.png"), dpi=300)
    plt.close()

    # Figure 3: Ablation Study
    plt.figure(figsize=(8, 5))
    abl_names = list(results["ablation_experiments_H4"].keys())
    abl_means = [results["ablation_experiments_H4"][k]["mean_reward"] for k in abl_names]
    abl_errs = [results["ablation_experiments_H4"][k]["ci95"] for k in abl_names]
    colors = ['purple', 'crimson', 'orange', 'gray']
    plt.bar(abl_names, abl_means, yerr=abl_errs, color=colors, capsize=5, alpha=0.85)
    plt.axhline(-0.1, color='red', linestyle='--', label='Uniform Random Baseline')
    plt.title("Ablation Study at Horizon H=4 (256 Candidates, N=20 Seeds)", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Cumulative Reward (95% CI)", fontsize=10)
    plt.xticks(rotation=15)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ext_fig3_ablation_H4_study.png"), dpi=300)
    plt.close()

    # Figure 6: Cohen's d vs Horizon H=1..6
    plt.figure(figsize=(8, 5))
    d_vals = [results["scaling_experiments"][f"H{h}"]["cohens_d"] for h in h_vals]
    plt.plot(h_vals, d_vals, '-d', color='darkgreen', linewidth=2.5, markersize=8)
    plt.axhline(0.0, color='gray', linestyle='--', label='No Effect (d = 0)')

    crossing = None
    for i in range(len(h_vals) - 1):
        if d_vals[i] * d_vals[i + 1] < 0:
            crossing = h_vals[i] + (0 - d_vals[i]) / (d_vals[i + 1] - d_vals[i]) * (h_vals[i + 1] - h_vals[i])
            break
    if crossing is not None:
        plt.axvline(crossing, color='orange', linestyle=':',
                    label=f'Crossover (H ≈ {crossing:.2f})')

    plt.title("Cohen's d Effect Size Growth vs. Horizon H=1..6", fontsize=11, fontweight='bold')
    plt.xlabel("Planning Horizon H", fontsize=10)
    plt.ylabel("Cohen's d (Quantum vs Classical)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ext_fig6_cohens_d_vs_horizon.png"), dpi=300)
    plt.close()

    # Figure 7: Noise Robustness (Qiskit NISQ Aer Simulation)
    plt.figure(figsize=(8, 5))
    dep_rates = [float(k) for k in results["noise_robustness"]["H3"]["depolarizing"].keys()]
    dep_h3 = [results["noise_robustness"]["H3"]["depolarizing"][str(k)]["mean_reward"] for k in dep_rates]
    dep_h4 = [results["noise_robustness"]["H4"]["depolarizing"][str(k)]["mean_reward"] for k in dep_rates]
    plt.plot(dep_rates, dep_h3, '-o', color='purple', label='H=3 (6 Qubits)', linewidth=2)
    plt.plot(dep_rates, dep_h4, '-s', color='crimson', label='H=4 (8 Qubits)', linewidth=2)
    plt.axhline(0.1472, color='blue', linestyle='--', label='H=6 Classical Return Threshold')
    plt.title("NISQ Depolarizing Error Robustness (Qiskit Aer Simulator)", fontsize=11, fontweight='bold')
    plt.xlabel("Depolarizing Error Rate per Gate (p)", fontsize=10)
    plt.ylabel("Mean Trajectory Reward", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ext_fig7_noise_robustness.png"), dpi=300)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\nExtended benchmark protocol completed cleanly in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_extended_protocol()
