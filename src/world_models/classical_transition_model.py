import numpy as np
from typing import Tuple, List, Dict, Any

class ClassicalTransitionModel:
    """
    Classical MLP Transition Predictor.
    Input feature vector: x(s, a) = [r/3.0, c/3.0, b0, b1]
    Output: Softmax probability distribution P_c(s' | s, a) over 16 next-states.
    """
    def __init__(self, hidden_dim: int = 32, lr: float = 0.01):
        self.num_inputs = 4
        self.hidden_dim = hidden_dim
        self.num_outputs = 16
        self.lr = lr

        # Weight matrices initialization
        np.random.seed(42)
        self.W1 = np.random.randn(self.num_inputs, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, self.num_outputs) * 0.1
        self.b2 = np.zeros(self.num_outputs)

        self.parameter_count = self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def _encode_input(self, state: Any, action: int) -> np.ndarray:
        if isinstance(state, tuple):
            r, c = state[0], state[1]
        else:
            r, c = state // 4, state % 4
        
        b0 = float((action >> 1) & 1)
        b1 = float(action & 1)
        return np.array([r / 3.0, c / 3.0, b0, b1], dtype=float)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Layer 1 ReLU
        h = np.maximum(0, np.dot(x, self.W1) + self.b1)
        # Layer 2 Softmax
        logits = np.dot(h, self.W2) + self.b2
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits - logits_max)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        return h, probs

    def predict_next_state_distribution(self, state: Any, action: int) -> np.ndarray:
        x = self._encode_input(state, action)
        _, probs = self.forward(x)
        return probs

    def train_on_batch(self, states: List[Any], actions: List[int], next_states: List[Any], steps: int = 5):
        if len(states) == 0:
            return 0.0

        N = len(states)
        X = np.array([self._encode_input(s, a) for s, a in zip(states, actions)])
        Y = np.zeros((N, 16))
        for idx, ns in enumerate(next_states):
            s_idx = ns[0] * 4 + ns[1] if isinstance(ns, tuple) else ns % 16
            Y[idx, s_idx] = 1.0

        total_loss = 0.0
        for _ in range(steps):
            h, probs = self.forward(X)
            eps = 1e-12
            loss = -np.mean(np.sum(Y * np.log(probs + eps), axis=-1))
            total_loss = float(loss)

            # Backpropagation
            d_logits = (probs - Y) / N
            dW2 = np.dot(h.T, d_logits)
            db2 = np.sum(d_logits, axis=0)

            dh = np.dot(d_logits, self.W2.T)
            dh[h <= 0] = 0.0  # ReLU gradient

            dW1 = np.dot(X.T, dh)
            db1 = np.sum(dh, axis=0)

            # Gradient descent step
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

        return total_loss

    def compute_nll(self, states: List[Any], actions: List[int], next_states: List[Any]) -> float:
        if len(states) == 0:
            return 0.0
        nlls = []
        for s, a, ns in zip(states, actions, next_states):
            probs = self.predict_next_state_distribution(s, a)
            ns_idx = ns[0] * 4 + ns[1] if isinstance(ns, tuple) else ns % 16
            p_target = max(probs[ns_idx], 1e-12)
            nlls.append(-np.log(p_target))
        return float(np.mean(nlls))
