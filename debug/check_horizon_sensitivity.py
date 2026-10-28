"""
Diagnostic-only script. Uses legacy seeds 42-61 and is NOT part of any
canonical suite or figure pipeline. Do not treat its output as a
publication result.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from src.environment.gridworld import GridWorld
from src.baselines.classical_agent import ClassicalAgent
from src.quantum.quantum_agent import QuantumAgent

def test_sensitivity():
    print("=== TESTING HORIZON SENSITIVITY FOR CLASSICAL AND QUANTUM AGENTS (H=2..6) ===")
    for H in [2, 3, 4, 5, 6]:
        c_rewards, q_rewards = [], []
        for seed in range(42, 47): # 5 seeds quick test
            np.random.seed(seed)
            env = GridWorld(width=6, height=6, start_pos=(0, 0), goal_pos=(5, 5))
            c_agent = ClassicalAgent(env=env, horizon=H, eval_budget=4**H)
            
            # Classical run
            c_ep_rewards = []
            for ep in range(40):
                pos = env.reset()
                total_r = 0.0
                done = False
                while not done:
                    a = c_agent.select_action(pos)
                    next_p, r, done, info = env.step(a)
                    c_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                c_ep_rewards.append(total_r)
            c_rewards.append(np.mean(c_ep_rewards))

            # Quantum run
            np.random.seed(seed)
            env = GridWorld(width=6, height=6, start_pos=(0, 0), goal_pos=(5, 5))
            q_agent = QuantumAgent(env=env, horizon=H, shots=128)
            q_ep_rewards = []
            for ep in range(40):
                pos = env.reset()
                total_r = 0.0
                done = False
                while not done:
                    a = q_agent.select_action(pos)
                    next_p, r, done, info = env.step(a)
                    q_agent.update(pos, a, r, next_p, done)
                    total_r += r
                    pos = next_p
                q_ep_rewards.append(total_r)
            q_rewards.append(np.mean(q_ep_rewards))

        print(f"H={H} ({2*H}Q, {4**H} futures) -> Classical Mean: {np.mean(c_rewards):.4f} +/- {np.std(c_rewards):.4f} | Quantum Mean: {np.mean(q_rewards):.4f} +/- {np.std(q_rewards):.4f}")

if __name__ == "__main__":
    print("[diagnostic] this script is not part of the canonical suite")
    test_sensitivity()
