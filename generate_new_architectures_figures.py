import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_all_new_architecture_figures(
    json_path: str = "artifacts/new_architectures_benchmark_results.json",
    fig_dir: str = "artifacts/figures"
):
    os.makedirs(fig_dir, exist_ok=True)
    if not os.path.exists(json_path):
        print(f"JSON results file {json_path} not found yet.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    primary = data.get("primary_benchmarks", {})
    architectures = data["metadata"]["architectures_evaluated"]
    benchmarks = list(primary.keys())

    # Figure 1: Reward vs. Benchmark (Noise-Free)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(benchmarks))
    width = 0.09

    for i, arch in enumerate(architectures):
        means = [primary[b][arch]["mean_reward"] for b in benchmarks if arch in primary[b]]
        errs = [primary[b][arch]["ci95"] for b in benchmarks if arch in primary[b]]
        plt.bar(x + i * width, means, width=width, yerr=errs, label=arch, capsize=2)

    plt.xlabel("Benchmark Task", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Cumulative Reward (95% CI)", fontsize=11, fontweight='bold')
    plt.title("Figure 1: Mean Return across Seven New Hybrid Architectures", fontsize=12, fontweight='bold')
    plt.xticks(x + width * 4, benchmarks)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_new_rewards.png"), dpi=300)
    plt.close()

    # Figure 2: Reward vs. NISQ Noise Level
    noise_data = data.get("noise_robustness", {})
    plt.figure(figsize=(10, 5))
    noise_conds = list(noise_data.keys())
    for arch in architectures:
        means = [noise_data[n][arch]["mean_reward"] for n in noise_conds if arch in noise_data[n]]
        plt.plot(noise_conds, means, '-o', label=arch, linewidth=2)

    plt.xlabel("NISQ Hardware Noise Condition", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Return (H=4 Maze)", fontsize=11, fontweight='bold')
    plt.title("Figure 2: Reward Degradation under Physical Hardware Noise", fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_new_noise_degradation.png"), dpi=300)
    plt.close()

    # Figure 3: Goal Success Rate (%) vs Benchmark
    plt.figure(figsize=(12, 6))
    for i, arch in enumerate(architectures):
        succs = [primary[b][arch]["success_rate"] * 100 for b in benchmarks if arch in primary[b]]
        plt.bar(x + i * width, succs, width=width, label=arch)

    plt.xlabel("Benchmark Task", fontsize=11, fontweight='bold')
    plt.ylabel("Goal Success Rate (%)", fontsize=11, fontweight='bold')
    plt.title("Figure 3: Goal Success Rate (%) across Architectures", fontsize=12, fontweight='bold')
    plt.xticks(x + width * 4, benchmarks)
    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_new_success_rates.png"), dpi=300)
    plt.close()

    # Figure 4: Parameter Count vs Reward Scatter Plot
    plt.figure(figsize=(9, 5))
    first_b = benchmarks[0] if benchmarks else "Maze-8x8"
    for arch in architectures:
        if arch in primary.get(first_b, {}):
            params = primary[first_b][arch]["parameter_count"]
            rew = primary[first_b][arch]["mean_reward"]
            plt.scatter(params, rew, s=140, label=arch)

    plt.xscale('log')
    plt.xlabel("Parameter Count (log scale)", fontsize=11, fontweight='bold')
    plt.ylabel(f"Mean Reward ({first_b})", fontsize=11, fontweight='bold')
    plt.title("Figure 4: Parameter Count vs Performance Tradeoff", fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_new_parameter_efficiency.png"), dpi=300)
    plt.close()

    # Figure 5: Total QPU Calls
    plt.figure(figsize=(10, 5))
    first_b = benchmarks[0] if benchmarks else "Maze-8x8"
    qpu_calls = [primary[first_b][arch]["total_qpu_calls"] for arch in architectures if arch in primary[first_b]]
    plt.bar(architectures, qpu_calls, color='navy', alpha=0.8)
    plt.xlabel("Architecture", fontsize=11, fontweight='bold')
    plt.ylabel("Total QPU Calls per Seed", fontsize=11, fontweight='bold')
    plt.title("Figure 5: QPU Resource Execution Overhead", fontsize=12, fontweight='bold')
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5_new_qpu_calls.png"), dpi=300)
    plt.close()

    # Figure 6: Cohen's d Effect Size Heatmap
    plt.figure(figsize=(9, 6))
    d_matrix = []
    hybrid_archs = [a for a in architectures if a != "Classical_Baseline"]
    for b in benchmarks:
        row = [primary[b][arch]["stats_vs_classical"]["cohens_d"] for arch in hybrid_archs]
        d_matrix.append(row)

    im = plt.imshow(d_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(im, label="Cohen's d vs Classical")
    plt.xticks(np.arange(len(hybrid_archs)), hybrid_archs, rotation=35, ha='right', fontsize=9)
    plt.yticks(np.arange(len(benchmarks)), benchmarks, fontsize=10, fontweight='bold')
    plt.title("Figure 6: Cohen's d Effect Sizes (Hybrid vs Classical)", fontsize=12, fontweight='bold')
    for i in range(len(benchmarks)):
        for j in range(len(hybrid_archs)):
            plt.text(j, i, f"{d_matrix[i][j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig6_new_cohens_d_heatmap.png"), dpi=300)
    plt.close()

    # Figure 7: Maze Size Scaling (4x4 to 16x16)
    maze_scale = data.get("maze_size_scaling", {})
    plt.figure(figsize=(10, 5))
    sizes = list(maze_scale.keys())
    for arch in architectures:
        vals = [maze_scale[sz][arch] for sz in sizes if arch in maze_scale[sz]]
        plt.plot(sizes, vals, '-o', label=arch, linewidth=2)

    plt.xlabel("Maze Grid Dimension", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Return", fontsize=11, fontweight='bold')
    plt.title("Figure 7: Spatial Maze Scalability (4x4 to 16x16)", fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig7_new_maze_scaling.png"), dpi=300)
    plt.close()

    # Figure 8: Circuit Depth Selection Distribution (Adaptive Complexity)
    plt.figure(figsize=(7, 4.5))
    metrics_q_depth = primary.get("CartPole-v1", {}).get("Q_Depth_Adaptive", {}).get("extra_metrics", {})
    if "depth_distribution" not in metrics_q_depth:
        raise KeyError(f"Missing depth_distribution in extra_metrics for Q_Depth_Adaptive in {json_path}")
    raw_depth = metrics_q_depth["depth_distribution"]
    if isinstance(raw_depth, list):
        keys = list(raw_depth[0].keys())
        depth_dist = {k: float(np.mean([d[k] for d in raw_depth])) for k in keys}
    else:
        depth_dist = raw_depth
    plt.bar(list(depth_dist.keys()), list(depth_dist.values()), color='purple', alpha=0.8)
    plt.ylabel("Allocation Frequency", fontsize=11, fontweight='bold')
    plt.title("Figure 8: Adaptive Circuit Complexity Allocation", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig8_depth_selection_dist.png"), dpi=300)
    plt.close()

    # Figure 9: Quantum Algorithm Selection Distribution
    plt.figure(figsize=(8, 4.5))
    metrics_q_algo = primary.get("CartPole-v1", {}).get("Q_Algorithm_Selection", {}).get("extra_metrics", {})
    if "algorithm_distribution" not in metrics_q_algo:
        raise KeyError(f"Missing algorithm_distribution in extra_metrics for Q_Algorithm_Selection in {json_path}")
    raw_algo = metrics_q_algo["algorithm_distribution"]
    if isinstance(raw_algo, list):
        keys = list(raw_algo[0].keys())
        algo_dist = {k: float(np.mean([d[k] for d in raw_algo])) for k in keys}
    else:
        algo_dist = raw_algo
    plt.bar(list(algo_dist.keys()), list(algo_dist.values()), color='darkgreen', alpha=0.8)
    plt.ylabel("Invocation Frequency", fontsize=11, fontweight='bold')
    plt.title("Figure 9: AI Quantum Algorithm Primitive Selection", fontsize=12, fontweight='bold')
    plt.xticks(rotation=20, ha='right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig9_algo_selection_dist.png"), dpi=300)
    plt.close()

    # Figure 10: Counterfactual Accuracy
    plt.figure(figsize=(7, 4.5))
    cf_acc = []
    for b in benchmarks:
        em = primary[b]["Q_Counterfactual"]["extra_metrics"]
        if "counterfactual_accuracy" not in em:
            raise KeyError(f"Missing counterfactual_accuracy for {b}")
        val = em["counterfactual_accuracy"]
        cf_acc.append(float(np.mean(val)) if isinstance(val, list) else float(val))
    plt.plot(benchmarks, cf_acc, '-s', color='crimson', linewidth=2, markersize=8)
    plt.ylabel("Counterfactual Prediction Accuracy", fontsize=11, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.title("Figure 10: Quantum Counterfactual Value Accuracy", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig10_counterfactual_accuracy.png"), dpi=300)
    plt.close()

    # Figure 11: Multi-Agent Coordination Success Rate
    plt.figure(figsize=(7, 4.5))
    coord_acc = []
    for b in benchmarks:
        em = primary[b]["Q_MultiAgent"]["extra_metrics"]
        if "coordination_success_rate" not in em:
            raise KeyError(f"Missing coordination_success_rate for {b}")
        val = em["coordination_success_rate"]
        coord_acc.append(float(np.mean(val)) if isinstance(val, list) else float(val))
    plt.bar(benchmarks, [c * 100 for c in coord_acc], color='darkorange', alpha=0.8)
    plt.ylabel("Coordination Success Rate (%)", fontsize=11, fontweight='bold')
    plt.ylim(0, 105)
    plt.title("Figure 11: Quantum Entangled Multi-Agent Coordination", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig11_multiagent_coordination.png"), dpi=300)
    plt.close()

    # Figure 12: Causal Intervention Accuracy
    plt.figure(figsize=(7, 4.5))
    causal_acc = []
    for b in benchmarks:
        em = primary[b]["Q_Causal"]["extra_metrics"]
        if "causal_intervention_accuracy" not in em:
            raise KeyError(f"Missing causal_intervention_accuracy for {b}")
        val = em["causal_intervention_accuracy"]
        causal_acc.append(float(np.mean(val)) if isinstance(val, list) else float(val))
    plt.plot(benchmarks, causal_acc, '-d', color='darkblue', linewidth=2, markersize=8)
    plt.ylabel("Intervention Correlation Accuracy", fontsize=11, fontweight='bold')
    plt.ylim(-1.0, 1.05)
    plt.title("Figure 12: Quantum Causal Inference Intervention Accuracy", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig12_causal_intervention_accuracy.png"), dpi=300)
    plt.close()

    print(f"Successfully generated Figures 1-12 in {fig_dir}")

if __name__ == "__main__":
    generate_all_new_architecture_figures()
