import os
import numpy as np
from src.evaluation.metrics import ExperimentMetrics

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class MetricVisualizer:
    """
    Generates high-resolution comparative figures and plots for the research proposal.
    """
    def __init__(self, output_dir: str = "artifacts/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_learning_curves(
        self,
        classical_metrics: ExperimentMetrics,
        quantum_metrics: ExperimentMetrics,
        window: int = 5,
        filename: str = "learning_curves.png"
    ) -> str:
        filepath = os.path.join(self.output_dir, filename)
        if not HAS_MATPLOTLIB:
            print(f"[Warning] matplotlib not installed; skipping plot generation for {filename}")
            return filepath

        plt.figure(figsize=(10, 5))
        c_rewards = classical_metrics.rewards
        q_rewards = quantum_metrics.rewards
        
        def moving_avg(arr, w):
            if len(arr) < w:
                return arr
            return np.convolve(arr, np.ones(w)/w, mode='valid')

        c_ma = moving_avg(c_rewards, window)
        q_ma = moving_avg(q_rewards, window)

        plt.plot(c_rewards, alpha=0.3, color='blue', label='Classical Baseline (Raw)')
        plt.plot(range(window-1, len(c_rewards)), c_ma, color='blue', linewidth=2.5, label=f'Classical (MA {window})')

        plt.plot(q_rewards, alpha=0.3, color='purple', label='Quantum Agent (Raw)')
        plt.plot(range(window-1, len(q_rewards)), q_ma, color='purple', linewidth=2.5, label=f'Quantum-Assisted (MA {window})')

        plt.title("Learning Curves: Classical Baseline vs Quantum-Assisted World Model", fontsize=12, fontweight='bold')
        plt.xlabel("Episode", fontsize=10)
        plt.ylabel("Cumulative Reward", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_success_rates(
        self,
        classical_metrics: ExperimentMetrics,
        quantum_metrics: ExperimentMetrics,
        filename: str = "success_rates.png"
    ) -> str:
        filepath = os.path.join(self.output_dir, filename)
        if not HAS_MATPLOTLIB:
            print(f"[Warning] matplotlib not installed; skipping plot generation for {filename}")
            return filepath

        plt.figure(figsize=(8, 5))

        c_succ = [1 if res.success else 0 for res in classical_metrics.episode_results]
        q_succ = [1 if res.success else 0 for res in quantum_metrics.episode_results]

        c_cum_rate = np.cumsum(c_succ) / (np.arange(len(c_succ)) + 1)
        q_cum_rate = np.cumsum(q_succ) / (np.arange(len(q_succ)) + 1)

        plt.plot(c_cum_rate * 100, color='blue', linewidth=2, label=f'Classical Final Rate: {c_cum_rate[-1]*100:.1f}%')
        plt.plot(q_cum_rate * 100, color='purple', linewidth=2, label=f'Quantum Final Rate: {q_cum_rate[-1]*100:.1f}%')

        plt.title("Cumulative Goal Success Rate (%)", fontsize=12, fontweight='bold')
        plt.xlabel("Episode", fontsize=10)
        plt.ylabel("Success Rate (%)", fontsize=10)
        plt.ylim(0, 105)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath
