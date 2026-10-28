import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class UncertaintyEstimator:
    """
    State Prediction Uncertainty Estimator for Adaptive Quantum Resource Allocation.
    Combines:
    1. Predictive Variance (variance of trajectory returns from recent evaluations)
    2. State Novelty (count-based novelty 1/sqrt(N_visit + 1))
    3. Value Confidence (margin between top-2 action values Q(s, a_1) - Q(s, a_2))
    """
    def __init__(
        self,
        num_states: int,
        num_actions: int = 4,
        alpha: float = 0.2,
        history_capacity: int = 10
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.history_capacity = history_capacity

        self.visit_counts = np.zeros(num_states, dtype=int)
        self.recent_returns = defaultdict(list)
        self.uncertainty_table = np.ones(num_states, dtype=float)  # Initialize high uncertainty (1.0)

    def estimate_uncertainty(
        self,
        state_idx: int,
        q_table: np.ndarray,
        returns_override: Optional[List[float]] = None
    ) -> float:
        """Estimate and return scalar uncertainty in [0, 1] for state_idx."""
        state_idx = max(0, min(self.num_states - 1, state_idx))

        # 1. Predictive Variance component
        history = returns_override if returns_override is not None else self.recent_returns[state_idx]
        if len(history) < 2:
            var_norm = 1.0
        else:
            v_var = float(np.var(history))
            var_norm = 1.0 - float(np.exp(-v_var))

        # 2. State Novelty component
        count = self.visit_counts[state_idx]
        novelty = 1.0 / np.sqrt(count + 1.0)

        # 3. Value Confidence component
        if q_table is not None and state_idx < len(q_table):
            q_vals = q_table[state_idx]
            sorted_q = np.sort(q_vals)
            q_diff = sorted_q[-1] - sorted_q[-2]
            scale = abs(sorted_q[-1]) + 1e-5
            confidence = float(np.clip(q_diff / scale, 0.0, 1.0))
        else:
            confidence = 0.0

        # Weighted combination of uncertainty signals
        raw_u = 0.4 * var_norm + 0.3 * novelty + 0.3 * (1.0 - confidence)
        raw_u = float(np.clip(raw_u, 0.0, 1.0))

        # Exponential moving average update
        prev_u = self.uncertainty_table[state_idx]
        smoothed_u = self.alpha * raw_u + (1.0 - self.alpha) * prev_u
        self.uncertainty_table[state_idx] = smoothed_u

        return float(np.clip(smoothed_u, 0.0, 1.0))

    def update_state_experience(self, state_idx: int, episode_return: float):
        """Update visit count and recent return history for state_idx."""
        state_idx = max(0, min(self.num_states - 1, state_idx))
        self.visit_counts[state_idx] += 1
        history = self.recent_returns[state_idx]
        history.append(episode_return)
        if len(history) > self.history_capacity:
            history.pop(0)
