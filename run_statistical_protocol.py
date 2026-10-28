import numpy as np
import scipy.stats as stats
import json
import os
from src.evaluation.benchmark import BenchmarkRunner

# Module-level defaults. Override via `import run_statistical_protocol as m; m.num_seeds = N; m.main()`
# for smoke tests. Do not shadow these names inside main().
num_seeds = 20
horizon = 3
grid_size = 6
shots = 128

def calculate_cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / s_pooled

def main():
    """
    Protocol runner.

    Uses module-level constants num_seeds/horizon/grid_size/shots.
    Override for smoke tests via `import run_statistical_protocol as m; m.num_seeds = 2; m.main()`.
    """
    if num_seeds < 2:
        raise ValueError(
            f"Paired t-tests require num_seeds >= 2, got {num_seeds}. "
            "Cannot compute p-values or Cohen's d with a single seed. "
            "If you want a smoke test, set num_seeds = 2 (not 1)."
        )

    print(f"=== Running {num_seeds}-Seed Power Analysis & Experimental Protocol Run ===")
    
    classical_rewards = []
    quantum_rewards = []
    classical_successes = []
    quantum_successes = []

    for seed in range(100, 100 + num_seeds):
        print(f"Running seed {seed}...")
        runner = BenchmarkRunner(
            grid_width=grid_size,
            grid_height=grid_size,
            horizon=horizon,
            episodes=30,
            eval_budget=4**horizon,
            shots=shots,
            seed=seed
        )
        res = runner.run_benchmark()
        c_m = res["classical_metrics"]
        q_m = res["quantum_metrics"]

        classical_rewards.append(c_m.mean_reward)
        quantum_rewards.append(q_m.mean_reward)
        classical_successes.append(c_m.success_rates)
        quantum_successes.append(q_m.success_rates)

    c_r = np.array(classical_rewards)
    q_r = np.array(quantum_rewards)

    # Statistical tests across seeds
    t_stat, p_val_t = stats.ttest_rel(q_r, c_r)
    w_stat, p_val_w = stats.wilcoxon(q_r, c_r)
    cohen_d = calculate_cohens_d(q_r, c_r)

    print(f"\n================ STATISTICAL EXPERIMENTAL RESULTS (N={num_seeds} SEEDS) ================")
    print(f"Classical Mean Reward ({num_seeds} seeds):  {np.mean(c_r):.4f} +/- {np.std(c_r):.4f}")
    print(f"Quantum Mean Reward ({num_seeds} seeds):    {np.mean(q_r):.4f} +/- {np.std(q_r):.4f}")
    print(f"Classical Mean Success Rate:       {np.mean(classical_successes)*100:.2f}%")
    print(f"Quantum Mean Success Rate:         {np.mean(quantum_successes)*100:.2f}%")
    print("-------------------------------------------------------------------------------")
    print(f"Paired t-statistic:               {t_stat:.4f}")
    print(f"p-value (Paired t-test):           {p_val_t:.4e}")
    print(f"Wilcoxon signed-rank p-value:     {p_val_w:.4e}")
    print(f"Cohen's d Effect Size:             {cohen_d:.4f}")
    print("===============================================================================\n")

    output_data = {
        "num_seeds": num_seeds,
        "grid_size": grid_size,
        "horizon": horizon,
        "qubits": 2 * horizon,
        "candidate_futures": 4**horizon,
        "classical_rewards_mean": float(np.mean(c_r)),
        "classical_rewards_std": float(np.std(c_r)),
        "quantum_rewards_mean": float(np.mean(q_r)),
        "quantum_rewards_std": float(np.std(q_r)),
        "classical_success_mean": float(np.mean(classical_successes)),
        "quantum_success_mean": float(np.mean(quantum_successes)),
        "paired_t_stat": float(t_stat),
        "p_val_t": float(p_val_t),
        "p_val_wilcoxon": float(p_val_w),
        "cohens_d": float(cohen_d)
    }

    os.makedirs("artifacts", exist_ok=True)
    out_name = f"artifacts/protocol_{num_seeds}seeds_results.json"
    with open(out_name, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved {num_seeds}-seed protocol results to: {out_name}")

if __name__ == "__main__":
    main()
