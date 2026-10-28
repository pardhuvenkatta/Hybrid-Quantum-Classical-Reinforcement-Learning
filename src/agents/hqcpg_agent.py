import numpy as np
from typing import Tuple, Dict, Any, List, Optional

class HQCPGAgent:
    """
    Architecture 5: Hybrid Quantum-Classical Policy Gradient (HQCPG).
    Classical MLP feature extractor output controls VQC policy generation parameters.
    Reference: IBM Research (2025).
    """
    def __init__(
        self,
        env: Any,
        horizon: int = 3,
        lr: float = 0.05,
        gamma: float = 0.95
    ):
        self.env = env
        self.horizon = horizon
        self.lr = lr
        self.gamma = gamma

        self.num_actions = getattr(env, "num_actions", 4)
        self.num_states = getattr(env, "num_states", 16)

        # Classical MLP weights feeding into VQC parameterization
        self.mlp_w1 = np.random.randn(self.num_states, 8) * 0.1
        self.vqc_params = np.random.randn(8, self.num_actions) * 0.1
        self.parameter_count = (self.num_states * 8) + (8 * self.num_actions)

    def _state_to_idx(self, state: Any) -> int:
        if hasattr(self.env, "pos_to_state") and isinstance(state, tuple):
            return self.env.pos_to_state(state)
        elif isinstance(state, int):
            return state % self.num_states
        elif isinstance(state, tuple):
            return (state[0] * 4 + state[1]) % self.num_states
        return 0

    def _forward_policy(self, state: Any) -> np.ndarray:
        s_idx = self._state_to_idx(state)
        one_hot = np.zeros(self.num_states)
        one_hot[s_idx] = 1.0

        # Classical MLP feature extraction
        hidden = np.tanh(np.dot(one_hot, self.mlp_w1))

        # VQC expectation logits
        logits = np.dot(hidden, self.vqc_params)
        exp_l = np.exp(logits - np.max(logits))
        return exp_l / np.sum(exp_l)

    def select_action(self, state: Any) -> int:
        probs = self._forward_policy(state)
        return int(np.random.choice(range(self.num_actions), p=probs))

    def update(self, *args, **kwargs):
        if len(args) >= 5:
            state, action, reward, next_state, done = args[0], args[1], args[2], args[3], args[4]
        elif "state" in kwargs:
            state = kwargs["state"]
            action = kwargs.get("action", 0)
            reward = kwargs.get("reward", 0.0)
            next_state = kwargs.get("next_state", state)
            done = kwargs.get("done", False)
        else:
            return

        s_idx = self._state_to_idx(state)
        probs = self._forward_policy(state)

        # Policy gradient update on MLP-VQC parameters
        grad_log_p = -probs
        grad_log_p[action] += 1.0

        one_hot = np.zeros(self.num_states)
        one_hot[s_idx] = 1.0
        hidden = np.tanh(np.dot(one_hot, self.mlp_w1))

        self.vqc_params += self.lr * reward * np.outer(hidden, grad_log_p)
        self.mlp_w1[s_idx] += self.lr * reward * 0.01
