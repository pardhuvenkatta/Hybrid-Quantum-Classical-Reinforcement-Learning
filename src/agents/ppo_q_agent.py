import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from src.quantum.quantum_encoding import StateActionEncoder

class PPOQAgent:
    """
    Architecture 1: PPO-Q (Proximal Policy Optimization with Quantum Policies/Values).
    Reference: Jin et al., arXiv:2501.07085 (2025).
    """
    def __init__(
        self,
        env: Any,
        horizon: int = 3,
        lr: float = 0.05,
        gamma: float = 0.95,
        clip_eps: float = 0.2
    ):
        self.env = env
        self.horizon = horizon
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.num_actions = getattr(env, "num_actions", 4)
        self.num_states = getattr(env, "num_states", 16)

        grid_w = getattr(env, "width", 4)
        grid_h = getattr(env, "height", 4)
        self.encoder = StateActionEncoder(grid_width=grid_w, grid_height=grid_h, depth=horizon)
        self.num_qubits = self.encoder.num_qubits

        # Parameter counts
        self.actor_weights = np.random.randn(self.num_qubits * 2) * 0.1
        self.critic_weights = np.random.randn(self.num_qubits * 2) * 0.1
        self.parameter_count = len(self.actor_weights) + len(self.critic_weights)
        self.last_prob = 0.25

    def _vqc_actor_forward(self, state: Any) -> np.ndarray:
        if isinstance(state, tuple):
            r, c = state
            grid_h = getattr(self.env, "height", 4)
            grid_w = getattr(self.env, "width", 4)
            theta_r = np.pi * (r / max(1, grid_h - 1))
            theta_c = np.pi * (c / max(1, grid_w - 1))
        else:
            s_idx = int(state) % self.num_states
            theta_r = np.pi * (s_idx / self.num_states)
            theta_c = np.pi * ((s_idx * 2) % self.num_states / self.num_states)

        logits = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            w0 = self.actor_weights[a % len(self.actor_weights)]
            w1 = self.actor_weights[(a + 1) % len(self.actor_weights)]
            logits[a] = np.cos(theta_r * w0) + np.sin(theta_c * w1)

        exp_l = np.exp(logits - np.max(logits))
        return exp_l / np.sum(exp_l)

    def _vqc_critic_forward(self, state: Any) -> float:
        if isinstance(state, tuple):
            r, c = state
            grid_h = getattr(self.env, "height", 4)
            grid_w = getattr(self.env, "width", 4)
            theta_r = np.pi * (r / max(1, grid_h - 1))
            theta_c = np.pi * (c / max(1, grid_w - 1))
        else:
            s_idx = int(state) % self.num_states
            theta_r = np.pi * (s_idx / self.num_states)
            theta_c = np.pi * ((s_idx * 2) % self.num_states / self.num_states)

        val = np.dot(self.critic_weights[:2], np.array([theta_r, theta_c]))
        return float(val)

    def select_action(self, state: Any) -> int:
        probs = self._vqc_actor_forward(state)
        action = int(np.random.choice(range(self.num_actions), p=probs))
        self.last_prob = probs[action]
        return action

    def update(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], list):
            for item in args[0]:
                if len(item) == 5:
                    self.update(*item)
                elif len(item) >= 6:
                    self.update(item[0], item[1], item[2], item[3], item[5], old_prob=item[4])
            return
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

        old_prob = kwargs.get("old_prob", getattr(self, "last_prob", 0.25))

        v_s = self._vqc_critic_forward(state)
        v_next = 0.0 if done else self._vqc_critic_forward(next_state)
        advantage = reward + self.gamma * v_next - v_s

        probs = self._vqc_actor_forward(state)
        new_prob = probs[action]
        ratio = new_prob / max(1e-6, old_prob)
        surr1 = ratio * advantage
        surr2 = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
        actor_loss = -min(surr1, surr2)

        self.actor_weights -= self.lr * actor_loss * 0.01
        self.critic_weights += self.lr * advantage * 0.01
