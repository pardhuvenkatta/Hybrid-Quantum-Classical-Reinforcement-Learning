import sys
if "debug" not in __file__:
    raise SystemExit("debug_diff.py is a diagnostic script; do not run from production paths.")

import numpy as np
import copy
from src.environment.gridworld import GridWorld
from src.baselines.classical_agent import ClassicalAgent
from src.quantum.quantum_agent import QuantumAgent
from src.evaluation.benchmark import BenchmarkRunner

def test_config(episodes, horizon, eval_budget, shots, seed=42):
    np.random.seed(seed)
    runner = BenchmarkRunner(
        grid_width=5,
        grid_height=5,
        horizon=horizon,
        episodes=episodes,
        eval_budget=eval_budget,
        shots=shots,
        seed=seed
    )
    res = runner.run_benchmark()
    c_m = res["classical_metrics"]
    q_m = res["quantum_metrics"]
    print(f"[Ep={episodes}, H={horizon}, Budget={eval_budget}, Shots={shots}, Seed={seed}]")
    print(f"  Classical: Mean Reward = {c_m.mean_reward:.4f}, Success = {c_m.success_rates*100:.1f}%")
    print(f"  Quantum:   Mean Reward = {q_m.mean_reward:.4f}, Success = {q_m.success_rates*100:.1f}%")
    return c_m.mean_reward, q_m.mean_reward

if __name__ == "__main__":
    print("=== DEBUG TEST 1: First Run Parameters (episodes=50, H=3, budget=64, shots=128) ===")
    test_config(episodes=50, horizon=3, eval_budget=64, shots=128, seed=42)

    print("\n=== DEBUG TEST 2: Second Run Parameters (episodes=30, H=3, budget=64, shots=128) ===")
    test_config(episodes=30, horizon=3, eval_budget=4**3, shots=128, seed=42)

    print("\n=== DEBUG TEST 3: Multi-seed check with episodes=50, H=3, budget=64, shots=128 ===")
    c_list, q_list = [], []
    for s in range(42, 47):
        c_r, q_r = test_config(episodes=50, horizon=3, eval_budget=64, shots=128, seed=s)
        c_list.append(c_r)
        q_list.append(q_r)
    print(f"\nAverage over 5 seeds (Ep=50, H=3, Budget=64):")
    print(f"  Classical Mean: {np.mean(c_list):.4f}")
    print(f"  Quantum Mean:   {np.mean(q_list):.4f}")
