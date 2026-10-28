import numpy as np
from typing import Tuple, List, Dict, Any
from src.world_models.classical_transition_model import ClassicalTransitionModel
from src.world_models.quantum_transition_model import QuantumTransitionModel

class HybridTransitionModel:
    """
    Hybrid Quantum-Classical Transition World Model.
    Mixture distribution: P_h(s' | s, a) = (1 - alpha) P_c(s' | s, a) + alpha * P_q(s' | s, a)
    Mixing weight alpha* in [0.0, 1.0] selected dynamically via held-out validation NLL minimization.
    """
    def __init__(
        self,
        classical_model: ClassicalTransitionModel,
        quantum_model: QuantumTransitionModel,
        initial_alpha: float = 0.50
    ):
        self.classical_model = classical_model
        self.quantum_model = quantum_model
        self.alpha = initial_alpha
        self.alpha_history: List[float] = [initial_alpha]

    def predict_next_state_distribution(self, state: Any, action: int) -> np.ndarray:
        p_c = self.classical_model.predict_next_state_distribution(state, action)
        p_q = self.quantum_model.predict_next_state_distribution(state, action)
        p_h = (1.0 - self.alpha) * p_c + self.alpha * p_q
        return p_h / np.sum(p_h)

    def update_validation_alpha(self, val_states: List[Any], val_actions: List[int], val_next_states: List[Any]) -> float:
        if len(val_states) == 0:
            return self.alpha

        candidate_alphas = np.linspace(0.0, 1.0, 21)
        best_alpha = self.alpha
        best_nll = float('inf')

        # Evaluate candidate alphas on validation dataset
        p_c_list = [self.classical_model.predict_next_state_distribution(s, a) for s, a in zip(val_states, val_actions)]
        p_q_list = [self.quantum_model.predict_next_state_distribution(s, a) for s, a in zip(val_states, val_actions)]
        targets = [ns[0] * 4 + ns[1] if isinstance(ns, tuple) else ns % 16 for ns in val_next_states]

        for cand in candidate_alphas:
            nlls = []
            for pc, pq, target_idx in zip(p_c_list, p_q_list, targets):
                ph = (1.0 - cand) * pc + cand * pq
                p_target = max(ph[target_idx], 1e-12)
                nlls.append(-np.log(p_target))
            mean_nll = float(np.mean(nlls))
            if mean_nll < best_nll:
                best_nll = mean_nll
                best_alpha = float(cand)

        self.alpha = best_alpha
        self.alpha_history.append(best_alpha)
        return best_alpha

    def compute_nll(self, states: List[Any], actions: List[int], next_states: List[Any]) -> float:
        if len(states) == 0:
            return 0.0
        nlls = []
        for s, a, ns in zip(states, actions, next_states):
            probs = self.predict_next_state_distribution(s, a)
            ns_idx = ns[0] * 4 + ns[1] if isinstance(ns, tuple) else ns % 16
            nlls.append(-np.log(max(probs[ns_idx], 1e-12)))
        return float(np.mean(nlls))
