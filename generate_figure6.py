"""
Standalone Figure-6 regenerator (Cohen's d vs. planning horizon H).

Reads `artifacts/canonical_results.json` and writes
`artifacts/figures/fig6_cohens_d_vs_horizon.png`.

NOTE: run_canonical_experiments.py ALSO produces an equivalent figure at
`artifacts/figures/canon_fig6_cohens_d_vs_horizon.png` from the same JSON.
The two files exist because Fig-6 was historically regenerated independently
of the canonical suite. For new work, prefer the `canon_`-prefixed version
emitted by `run_canonical_experiments.py` (both scripts dynamically
read and plot all horizons present in `canonical_results.json`).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    fig_dir = "artifacts/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    json_path = "artifacts/canonical_results.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing data file: {json_path}. Run run_canonical_experiments.py first.")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if "scaling_experiments" not in data:
        raise KeyError(f"Key 'scaling_experiments' missing from {json_path}.")
    
    scaling = data["scaling_experiments"]
    horizons = sorted([int(k.replace("H", "")) for k in scaling.keys()])
    cohens_d = [scaling[f"H{h}"]["cohens_d"] for h in horizons]
    
    print(f"Reading from: {json_path}")
    print(f"Horizons: {horizons}")
    print(f"Cohen's d values: {cohens_d}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(horizons, cohens_d, '-o', color='purple', linewidth=2.5, markersize=8, label="Cohen's d Effect Size")
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='No Advantage Threshold (d = 0)')

    crossing = None
    for i in range(len(horizons) - 1):
        if cohens_d[i] * cohens_d[i + 1] < 0:
            # linear interpolation between horizons[i] and horizons[i+1]
            crossing = horizons[i] + (0 - cohens_d[i]) / (cohens_d[i + 1] - cohens_d[i]) * (horizons[i + 1] - horizons[i])
            break

    if crossing is not None:
        plt.axvline(crossing, color='gray', linestyle=':', linewidth=1.5,
                    label=f'Crossover (H ≈ {crossing:.2f})')

    plt.title("Effect Size Scaling: Cohen's d vs. Planning Horizon H", fontsize=12, fontweight='bold')
    plt.xlabel("Planning Horizon H (Qubits = 2H)", fontsize=10)
    plt.ylabel("Cohen's d Effect Size (Quantum vs. Classical)", fontsize=10)
    plt.xticks(horizons)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig6_cohens_d_vs_horizon.png"), dpi=300)
    plt.close()
    print(f"Saved Figure 6 to {fig_dir}/fig6_cohens_d_vs_horizon.png")

if __name__ == "__main__":
    main()
