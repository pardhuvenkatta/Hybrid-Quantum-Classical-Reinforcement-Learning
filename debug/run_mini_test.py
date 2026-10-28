"""
Diagnostic-only script. Uses legacy seeds 42-61 and is NOT part of any
canonical suite or figure pipeline. Do not treat its output as a
publication result.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import numpy as np
from src.evaluation.benchmark import BenchmarkRunner

def main():
    print("--- Running Mini Verification Experiment (H=2, 2 Seeds) ---")
    results = []
    for seed in [42, 43]:
        runner = BenchmarkRunner(
            grid_width=6,
            grid_height=6,
            horizon=2,
            episodes=10,
            eval_budget=16,
            shots=128,
            seed=seed
        )
        res = runner.run_benchmark()
        c_m, q_m = res["classical_metrics"], res["quantum_metrics"]
        entry = {
            "seed": seed,
            "classical_mean_reward": float(c_m.mean_reward),
            "quantum_mean_reward": float(q_m.mean_reward),
            "classical_success_rate": float(c_m.success_rates),
            "quantum_success_rate": float(q_m.success_rates)
        }
        results.append(entry)
        print(f"Seed {seed} -> Classical: {c_m.mean_reward:.4f}, Quantum: {q_m.mean_reward:.4f}")

    out_file = "artifacts/verification_test.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Verification test completed. Output saved to {out_file}")

if __name__ == "__main__":
    print("[diagnostic] this script is not part of the canonical suite")
    main()
