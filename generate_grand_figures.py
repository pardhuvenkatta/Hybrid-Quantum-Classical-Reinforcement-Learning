import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_all_grand_figures(json_path: str = "artifacts/grand_hybrid_comparison_results.json", fig_dir: str = "artifacts/figures"):
    os.makedirs(fig_dir, exist_ok=True)
    if not os.path.exists(json_path):
        print(f"JSON results file {json_path} not found yet.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    primary = data.get("primary_benchmarks", {})
    architectures = data["metadata"]["architectures_evaluated"]

    # Figure 1: Reward vs. Benchmark (Noise-Free)
    plt.figure(figsize=(10, 6))
    benchmarks = list(primary.keys())
    x = np.arange(len(benchmarks))
    width = 0.12

    for i, arch in enumerate(architectures):
        means = [primary[b][arch]["mean_reward"] for b in benchmarks if arch in primary[b]]
        errs = [primary[b][arch]["ci95"] for b in benchmarks if arch in primary[b]]
        plt.bar(x + i * width, means, width=width, yerr=errs, label=arch, capsize=3)

    plt.xlabel("Benchmark Task", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Cumulative Reward (95% CI)", fontsize=11, fontweight='bold')
    plt.title("Figure 1: Benchmark Reward across Architectures (Noise-Free)", fontsize=12, fontweight='bold')
    plt.xticks(x + width * 3, benchmarks)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_grand_rewards.png"), dpi=300)
    plt.close()

    # Figure 2: Reward vs. Noise Condition
    noise_data = data.get("noise_robustness", {})
    plt.figure(figsize=(9, 5))
    noise_conds = list(noise_data.keys())
    for arch in architectures:
        means = [noise_data[n][arch]["mean_reward"] for n in noise_conds if arch in noise_data[n]]
        plt.plot(noise_conds, means, '-o', label=arch, linewidth=2)

    plt.xlabel("NISQ Noise Condition", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Return (H=4 Maze)", fontsize=11, fontweight='bold')
    plt.title("Figure 2: Reward Degradation under Hardware Noise", fontsize=12, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_grand_noise_degradation.png"), dpi=300)
    plt.close()

    # Figure 3: Success Rate (%) vs Benchmark
    plt.figure(figsize=(10, 6))
    for i, arch in enumerate(architectures):
        succs = [primary[b][arch]["success_rate"] * 100 for b in benchmarks if arch in primary[b]]
        plt.bar(x + i * width, succs, width=width, label=arch)

    plt.xlabel("Benchmark Task", fontsize=11, fontweight='bold')
    plt.ylabel("Success Rate (%)", fontsize=11, fontweight='bold')
    plt.title("Figure 3: Goal Success Rate (%) across Architectures", fontsize=12, fontweight='bold')
    plt.xticks(x + width * 3, benchmarks)
    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_grand_success_rates.png"), dpi=300)
    plt.close()

    # Figure 4: Parameter Count vs Reward
    plt.figure(figsize=(8, 5))
    first_b = benchmarks[0] if benchmarks else "Maze-8x8"
    for arch in architectures:
        if arch in primary.get(first_b, {}):
            params = primary[first_b][arch]["parameter_count"]
            rew = primary[first_b][arch]["mean_reward"]
            plt.scatter(params, rew, s=120, label=arch)

    plt.xscale('log')
    plt.xlabel("Parameter Count (log scale)", fontsize=11, fontweight='bold')
    plt.ylabel(f"Mean Reward ({first_b})", fontsize=11, fontweight='bold')
    plt.title("Figure 4: Parameter Efficiency vs. Performance", fontsize=12, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_grand_parameter_efficiency.png"), dpi=300)
    plt.close()

    # Figure 5: Total QPU Calls — removed.
    # Reason: run_grand_hybrid_comparison.py does not instrument QPU calls per
    # architecture, so the previous fallback (`50*15`) yielded identical values
    # across all architectures and the figure conveyed no information. Restore
    # this figure if per-architecture QPU instrumentation is added.
    #
    # Note: run_adaptive_comparison.py DOES report measured QPU calls (see
    # artifacts/adaptive_comparison_results.json → horizon_scaling.H*.fixed_q.qpu_calls
    # and .adaptive_q.qpu_calls). That is a separate suite, not the grand comparison.

    # Figure 6: Cohen's d Heatmap (vs. Classical)
    plt.figure(figsize=(8, 5))
    first_b = benchmarks[0] if benchmarks else "Maze-8x8"
    d_vals = [primary[first_b][arch]["stats_vs_classical"]["cohens_d"] for arch in architectures if arch != "Classical_Baseline"]
    non_class_archs = [arch for arch in architectures if arch != "Classical_Baseline"]
    plt.bar(non_class_archs, d_vals, color='darkgreen', alpha=0.85)
    plt.axhline(0.0, color='gray', linestyle='--')
    plt.xlabel("Hybrid Architecture", fontsize=11, fontweight='bold')
    plt.ylabel("Cohen's d (vs. Classical)", fontsize=11, fontweight='bold')
    plt.title("Figure 6: Effect Size relative to Classical Baseline", fontsize=12, fontweight='bold')
    plt.xticks(rotation=20)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig6_grand_cohens_d.png"), dpi=300)
    plt.close()

    # Figure 7: Maze Size Scaling (4x4 to 16x16)
    maze_scaling = data.get("maze_size_scaling", {})
    plt.figure(figsize=(8, 5))
    sizes = list(maze_scaling.keys())
    for arch in architectures:
        means = [maze_scaling[sz][arch] for sz in sizes if arch in maze_scaling[sz]]
        plt.plot(sizes, means, '-s', label=arch, linewidth=2)

    plt.xlabel("Maze Size", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Cumulative Reward", fontsize=11, fontweight='bold')
    plt.title("Figure 7: Spatial Maze Scaling (4x4 to 16x16)", fontsize=12, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig7_grand_maze_scaling.png"), dpi=300)
    plt.close()

    generated_figures = [
        "fig1_grand_rewards.png",
        "fig2_grand_noise_degradation.png",
        "fig3_grand_success_rates.png",
        "fig4_grand_parameter_efficiency.png",
        "fig6_grand_cohens_d.png",
        "fig7_grand_maze_scaling.png",
    ]
    print(f"Successfully generated {len(generated_figures)} figures in {fig_dir}:")
    for fname in generated_figures:
        print(f"  - {fname}")
    print("Note: Figure 5 (QPU calls) was removed — see comment near former Figure 5 block.")

if __name__ == "__main__":
    generate_all_grand_figures()
