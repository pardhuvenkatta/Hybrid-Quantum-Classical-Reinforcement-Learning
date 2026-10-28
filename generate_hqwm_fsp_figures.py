import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_all_hqwm_fsp_figures(
    json_path: str = "artifacts/hqwm_fsp_results.json",
    fig_dir: str = "artifacts/figures"
):
    os.makedirs(fig_dir, exist_ok=True)
    if not os.path.exists(json_path):
        print(f"JSON results file {json_path} not found yet.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    models_data = data.get("models", {})
    models = list(models_data.keys())

    # Figure 1: Success Rate (%) across Models
    plt.figure(figsize=(10, 5))
    succs = [models_data[m]["success_rate"] * 100 for m in models]
    colors = ['gray', 'blue', 'purple', 'red', 'cyan', 'green', 'magenta', 'gold']
    plt.bar(models, succs, color=colors, alpha=0.85)
    plt.ylabel("Success Rate (%)", fontsize=11, fontweight='bold')
    plt.title("Figure 1: Goal Success Rate (%) across All 8 Models", fontsize=12, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_hqwm_success_rates.png"), dpi=300)
    plt.close()

    # Figure 2: Mean Episode Return across Models
    plt.figure(figsize=(10, 5))
    returns = [models_data[m]["mean_return"] for m in models]
    errs = [models_data[m]["ci95_return"] for m in models]
    plt.bar(models, returns, yerr=errs, color=colors, alpha=0.85, capsize=3)
    plt.ylabel("Mean Episode Return (95% CI)", fontsize=11, fontweight='bold')
    plt.title("Figure 2: Episode Return Comparison across 8 Models", fontsize=12, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_hqwm_returns.png"), dpi=300)
    plt.close()

    # Figure 3: Alpha* Evolution vs Training Interval
    plt.figure(figsize=(7, 4.5))
    alpha_ev = data.get("alpha_star_evolution", [])
    if alpha_ev:
        intervals = [a["interval"] for a in alpha_ev]
        alphas = [a.get("mean_alpha_star", a.get("alpha_star", 0.5)) for a in alpha_ev]
        ci95s = [a.get("ci95_alpha_star", 0.0) for a in alpha_ev]
        if max(ci95s) > 0:
            plt.errorbar(intervals, alphas, yerr=ci95s, fmt='-o', color='crimson', linewidth=2.5, markersize=7, capsize=3)
        else:
            plt.plot(intervals, alphas, '-o', color='crimson', linewidth=2.5, markersize=7)
    plt.xlabel("Training Episode Interval", fontsize=11, fontweight='bold')
    plt.ylabel("Validation-Selected Mixing Weight α*", fontsize=11, fontweight='bold')
    plt.title("Figure 3: Hybrid Mixing Weight α* Validation Selection", fontsize=12, fontweight='bold')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_hqwm_alpha_star.png"), dpi=300)
    plt.close()

    # Figure 4: Held-Out NLL Prediction Quality
    plt.figure(figsize=(9, 5))
    mod_labels = [m for m in models if "Model_Free" not in m and "Reference" not in m]
    nlls = [models_data[m].get("mean_test_nll", models_data[m].get("mean_nll", 0.5)) for m in mod_labels]
    plt.bar(mod_labels, nlls, color='navy', alpha=0.8)
    plt.ylabel("Held-Out Test Negative Log-Likelihood (NLL)", fontsize=11, fontweight='bold')
    plt.title("Figure 4: Transition Predictor Held-Out Test NLL Quality", fontsize=12, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_hqwm_heldout_nll.png"), dpi=300)
    plt.close()

    # Figure 5: Planning Cost Breakdown (Wall-Clock Execution Time)
    plt.figure(figsize=(9, 5))
    times = [models_data[m]["wall_clock_time"] for m in models]
    plt.bar(models, times, color='teal', alpha=0.85)
    plt.ylabel("Wall-Clock Execution Time (s)", fontsize=11, fontweight='bold')
    plt.title("Figure 5: Planning Execution Time Breakdown", fontsize=12, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5_hqwm_planning_cost.png"), dpi=300)
    plt.close()

    # Figure 6: Quantum Cost Breakdown (Circuit Executions & Shots)
    plt.figure(figsize=(8, 4.5))
    q_mods = [m for m in models if models_data[m]["circuit_executions"] > 0]
    circuits = [models_data[m]["circuit_executions"] for m in q_mods]
    plt.bar(q_mods, circuits, color='purple', alpha=0.85)
    plt.ylabel("Total QPU Circuit Executions", fontsize=11, fontweight='bold')
    plt.title("Figure 6: Quantum QPU Execution Budget", fontsize=12, fontweight='bold')
    plt.xticks(rotation=20, ha='right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig6_hqwm_quantum_cost.png"), dpi=300)
    plt.close()

    # Figure 7: Robustness to Slip Probability
    plt.figure(figsize=(9, 5))
    slip_data = data.get("robustness_slip_prob", {})
    slips = [float(s.replace("p_", "")) for s in slip_data.keys()]
    for m in ["1_Model_Free_Q_Learning", "2_Classical_World_Model", "4_Proposed_Hybrid_HQWM", "5_Two_Classical_Ensemble"]:
        if "p_0.05" in slip_data and m in slip_data["p_0.05"]:
            vals = []
            for sl in slips:
                entry = slip_data[f"p_{sl}"][m]
                vals.append(entry["mean_reward"] if isinstance(entry, dict) else entry)
            plt.plot(slips, vals, '-o', label=m, linewidth=2)
    plt.xlabel("Environment Slip Probability (Stochasticity)", fontsize=11, fontweight='bold')
    plt.ylabel("Mean Episode Return", fontsize=11, fontweight='bold')
    plt.title("Figure 7: Model Robustness to Environment Stochasticity", fontsize=12, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig7_hqwm_slip_robustness.png"), dpi=300)
    plt.close()

    # Figure 8: Robustness to Finite Measurement Shots
    plt.figure(figsize=(8, 4.5))
    shot_data = data.get("robustness_finite_shots", {})
    shots_keys = list(shot_data.keys())
    shots_vals = [float(k.replace("shots_", "")) for k in shots_keys]
    nll_vals = [shot_data[k].get("mean_nll", shot_data[k].get("nll", 0.5)) for k in shots_keys]
    plt.plot(shots_vals, nll_vals, '-s', color='darkgreen', linewidth=2, markersize=7)
    plt.xscale('log')
    plt.xlabel("QPU Measurement Shots (M, log scale)", fontsize=11, fontweight='bold')
    plt.ylabel("Quantum Model Held-Out NLL", fontsize=11, fontweight='bold')
    plt.title("Figure 8: Finite-shot transition-predictor NLL (model underfits at L=3; not evidence of readout as dominant bottleneck)", fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig8_hqwm_finite_shots.png"), dpi=300)
    plt.close()

    print(f"Successfully generated Figures 1-8 in {fig_dir}")

if __name__ == "__main__":
    generate_all_hqwm_fsp_figures()
