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

def evaluate_agents_learning(horizon=3, episodes=50, seeds=10):
    print(f"=== TESTING REWARD PROFILE: ALL EPISODES vs FINAL EVALUATION (H={horizon}, Ep={episodes}, Seeds={seeds}) ===")
    c_all_rewards, q_all_rewards = [], []
    c_final_rewards, q_final_rewards = [], []
    c_final_succ, q_final_succ = [], []

    for seed in range(42, 42 + seeds):
        np.random.seed(seed)
        env = GridWorld(width=5, height=5, start_pos=(0, 0), goal_pos=(4, 4), obstacles=[(1, 1), (1, 2), (2, 3), (3, 1)], traps=[(2, 1)], max_steps=35)
        
        # Classical Agent
        c_agent = ClassicalAgent(env=env, horizon=horizon, eval_budget=4**horizon)
        c_ep_rewards = []
        for ep in range(episodes):
            pos = env.reset()
            ep_r = 0.0
            done = False
            while not done:
                a = c_agent.select_action(pos)
                n_pos, r, done, info = env.step(a)
                c_agent.update(pos, a, r, n_pos, done)
                ep_r += r
                pos = n_pos
            c_ep_rewards.append(ep_r)
        
        c_all_rewards.extend(c_ep_rewards)
        c_final_rewards.append(np.mean(c_ep_rewards[-10:]))
        c_final_succ.append(info.get("reached_goal", False))

        # Quantum Agent
        np.random.seed(seed)
        env = GridWorld(width=5, height=5, start_pos=(0, 0), goal_pos=(4, 4), obstacles=[(1, 1), (1, 2), (2, 3), (3, 1)], traps=[(2, 1)], max_steps=35)
        q_agent = QuantumAgent(env=env, horizon=horizon, shots=128)
        q_ep_rewards = []
        for ep in range(episodes):
            pos = env.reset()
            ep_r = 0.0
            done = False
            while not done:
                a = q_agent.select_action(pos)
                n_pos, r, done, info = env.step(a)
                q_agent.update(pos, a, r, n_pos, done)
                ep_r += r
                pos = n_pos
            q_ep_rewards.append(ep_r)
        
        q_all_rewards.extend(q_ep_rewards)
        q_final_rewards.append(np.mean(q_ep_rewards[-10:]))
        q_final_succ.append(info.get("reached_goal", False))

    print(f"Classical Agent -> Overall Mean: {np.mean(c_all_rewards):.4f} | Final 10-Ep Mean: {np.mean(c_final_rewards):.4f} | Success Rate: {np.mean(c_final_succ)*100:.1f}%")
    print(f"Quantum Agent   -> Overall Mean: {np.mean(q_all_rewards):.4f} | Final 10-Ep Mean: {np.mean(q_final_rewards):.4f} | Success Rate: {np.mean(q_final_succ)*100:.1f}%")

if __name__ == "__main__":
    print("[diagnostic] this script is not part of the canonical suite")
    for H in [2, 3, 4, 5, 6]:
        evaluate_agents_learning(horizon=H, episodes=50, seeds=10)
