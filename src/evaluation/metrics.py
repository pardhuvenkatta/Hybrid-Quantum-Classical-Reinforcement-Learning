import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

@dataclass
class EpisodeResult:
    episode_index: int
    total_reward: float
    steps: int
    success: bool
    evaluations_count: int
    shots_count: int

@dataclass
class ExperimentMetrics:
    agent_name: str
    episode_results: List[EpisodeResult] = field(default_factory=list)
    qubit_count: int = 0
    circuit_depth: int = 0

    @property
    def total_episodes(self) -> int:
        return len(self.episode_results)

    @property
    def rewards(self) -> np.ndarray:
        return np.array([res.total_reward for res in self.episode_results])

    @property
    def success_rates(self) -> float:
        if not self.episode_results:
            return 0.0
        return float(np.mean([1.0 if res.success else 0.0 for res in self.episode_results]))

    @property
    def avg_steps(self) -> float:
        if not self.episode_results:
            return 0.0
        return float(np.mean([res.steps for res in self.episode_results]))

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards)) if len(self.rewards) > 0 else 0.0

    @property
    def std_reward(self) -> float:
        return float(np.std(self.rewards)) if len(self.rewards) > 0 else 0.0

class HypothesisTester:
    """
    Evaluates Hypothesis H1 (Quantum World Model Advantage) vs H0 (Null Hypothesis).
    Performs independent Welch's t-test and Mann-Whitney U test on episode rewards.
    """
    @staticmethod
    def compare_agents(
        classical_metrics: ExperimentMetrics,
        quantum_metrics: ExperimentMetrics,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        class_rewards = classical_metrics.rewards
        quant_rewards = quantum_metrics.rewards

        n1, n2 = len(quant_rewards), len(class_rewards)
        m1, m2 = quantum_metrics.mean_reward, classical_metrics.mean_reward
        v1 = np.var(quant_rewards, ddof=1) if n1 > 1 else 0.0
        v2 = np.var(class_rewards, ddof=1) if n2 > 1 else 0.0

        if not HAS_SCIPY:
            raise ImportError("scipy is required for statistical hypothesis testing (Welch's t-test and Mann-Whitney U test).")

        t_stat, p_val_t = stats.ttest_ind(quant_rewards, class_rewards, equal_var=False)
        u_stat, p_val_u = stats.mannwhitneyu(quant_rewards, class_rewards, alternative='greater')

        # H1 accepted if p_val_u < alpha and quantum mean reward > classical mean reward
        h1_accepted = bool((p_val_u < alpha) and (m1 > m2))

        return {
            "classical_mean_reward": m2,
            "classical_success_rate": classical_metrics.success_rates,
            "quantum_mean_reward": m1,
            "quantum_success_rate": quantum_metrics.success_rates,
            "reward_difference": m1 - m2,
            "t_statistic": float(t_stat),
            "p_value_t_test": float(p_val_t),
            "u_statistic": float(u_stat),
            "p_value_mann_whitney": float(p_val_u),
            "alpha": alpha,
            "h1_accepted": h1_accepted,
            "conclusion": (
                "STATISTICALLY SIGNIFICANT QUANTUM ADVANTAGE (H1 Accepted)"
                if h1_accepted
                else "NO STATISTICALLY SIGNIFICANT QUANTUM ADVANTAGE (H0 Retained)"
            )
        }
