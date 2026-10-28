"""
Single-configuration benchmark runner (legacy). Not part of the canonical
20-seed suite — use run_canonical_experiments.py for publication figures.

Default grid size matches the canonical 6x6 GridWorld; pass --grid-size 5 for
compatibility with pre-2026 scripts.
"""

import argparse
import json
import os
import sys
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.visualization import MetricVisualizer

def main():
    parser = argparse.ArgumentParser(description="Quantum-Assisted World Model Research Benchmark")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes per agent")
    parser.add_argument("--horizon", type=int, default=2, help="Trajectory planning horizon H")
    parser.add_argument("--grid-size", type=int, default=6, help="Grid size N x N (canonical = 6)")
    parser.add_argument("--shots", type=int, default=64, help="Quantum measurement shots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory to save figures and JSON metrics")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")

    print("===============================================================")
    print("   QUANTUM-ASSISTED WORLD MODEL RESEARCH BENCHMARK RUNNER   ")
    print("===============================================================")
    print(f"Grid Size:        {args.grid_size} x {args.grid_size}")
    print(f"Planning Horizon: H = {args.horizon} ({4**args.horizon} Candidate Trajectories)")
    print(f"Episodes:         {args.episodes}")
    print(f"Quantum Shots:    {args.shots}")
    print(f"Random Seed:      {args.seed}")
    print("---------------------------------------------------------------")

    runner = BenchmarkRunner(
        grid_width=args.grid_size,
        grid_height=args.grid_size,
        horizon=args.horizon,
        episodes=args.episodes,
        eval_budget=4**args.horizon,
        shots=args.shots,
        seed=args.seed
    )

    results = runner.run_benchmark()

    classical_m = results["classical_metrics"]
    quantum_m = results["quantum_metrics"]
    h_report = results["hypothesis_report"]

    print("\n---------------------- EXPERIMENT RESULTS ----------------------")
    print(f"Classical Agent Mean Reward:   {classical_m.mean_reward:.4f} +/- {classical_m.std_reward:.4f}")
    print(f"Classical Agent Success Rate:  {classical_m.success_rates * 100:.2f}%")
    print(f"Quantum Agent Mean Reward:     {quantum_m.mean_reward:.4f} +/- {quantum_m.std_reward:.4f}")
    print(f"Quantum Agent Success Rate:    {quantum_m.success_rates * 100:.2f}%")
    print(f"Qubits Used:                   {quantum_m.qubit_count}")
    print("---------------------------------------------------------------")
    print(f"T-statistic:                  {h_report['t_statistic']:.4f}")
    print(f"P-value (T-test):             {h_report['p_value_t_test']:.4e}")
    print(f"Mann-Whitney U p-value:       {h_report['p_value_mann_whitney']:.4e}")
    print(f"Hypothesis Decision:          {h_report['conclusion']}")
    print("===============================================================\n")

    # Generate Visualization Plots
    visualizer = MetricVisualizer(output_dir=figures_dir)
    lc_path = visualizer.plot_learning_curves(classical_m, quantum_m)
    sr_path = visualizer.plot_success_rates(classical_m, quantum_m)

    print(f"Saved figure: {lc_path}")
    print(f"Saved figure: {sr_path}")

    # Export JSON metrics
    export_data = {
        "config": vars(args),
        "hypothesis_report": h_report,
        "classical_metrics": {
            "mean_reward": classical_m.mean_reward,
            "std_reward": classical_m.std_reward,
            "success_rate": classical_m.success_rates,
            "rewards": classical_m.rewards.tolist()
        },
        "quantum_metrics": {
            "mean_reward": quantum_m.mean_reward,
            "std_reward": quantum_m.std_reward,
            "success_rate": quantum_m.success_rates,
            "qubit_count": quantum_m.qubit_count,
            "rewards": quantum_m.rewards.tolist()
        }
    }

    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Exported metrics JSON: {json_path}")

if __name__ == "__main__":
    main()
