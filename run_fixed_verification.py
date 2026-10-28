"""
Fixed-oracle verification benchmark (5x5 GridWorld).

IMPORTANT: This runner intentionally uses a 5x5 grid (`width=5, height=5,
start=(0,0), goal=(4,4)`) — NOT the canonical 6x6 GridWorld used by
run_canonical_experiments.py. This is a small-scale sanity check that swaps the
learned QuantumWorldModel for a `FixedQuantumWorldModel`, so it is isolated from
the canonical suite by design.

Do not compare its numbers directly against artifacts/canonical_results.json.
Use `run_canonical_experiments.py` for publication results.
"""

import numpy as np
from src.environment.gridworld import GridWorld
from src.baselines.classical_agent import ClassicalAgent
from src.quantum.quantum_agent import QuantumAgent
from src.quantum.fixed_oracle import FixedQuantumWorldModel
from src.evaluation.benchmark import BenchmarkRunner

def run_fixed_benchmark(episodes=50, horizon=3, shots=128, seeds=20):
    print(f"\n================ VERIFYING FIXED ORACLE BENCHMARK (H={horizon}, Ep={episodes}, Seeds={seeds}) ================")
    c_rewards, q_rewards = [], []
    c_succ, q_succ = [], []

    for seed in range(100, 100 + seeds):
        np.random.seed(seed)
        env = GridWorld(
            width=5, height=5, start_pos=(0, 0), goal_pos=(4, 4),
            obstacles=[(1, 1), (1, 2), (2, 3), (3, 1)],
            traps=[(2, 1)], max_steps=35
        )

        c_agent = ClassicalAgent(env=env, horizon=horizon, eval_budget=4**horizon)
        q_agent = QuantumAgent(env=env, horizon=horizon, shots=shots)
        q_agent.world_model = FixedQuantumWorldModel(env=env, horizon=horizon)

        c_ep_r, c_ep_s = [], []
        q_ep_r, q_ep_s = [], []

        for ep in range(episodes):
            # Classical
            c_pos = env.reset()
            c_rew, c_done = 0.0, False
            c_info = {}
            while not c_done:
                a = c_agent.select_action(c_pos)
                n_pos, r, c_done, c_info = env.step(a)
                c_agent.update(c_pos, a, r, n_pos, c_done)
                c_rew += r
                c_pos = n_pos
            c_ep_r.append(c_rew)
            c_ep_s.append(1.0 if c_info.get("reached_goal", False) else 0.0)

            # Quantum
            q_pos = env.reset()
            q_rew, q_done = 0.0, False
            q_info = {}
            while not q_done:
                a = q_agent.select_action(q_pos)
                n_pos, r, q_done, q_info = env.step(a)
                q_agent.update(q_pos, a, r, n_pos, q_done)
                q_rew += r
                q_pos = n_pos
            q_ep_r.append(q_rew)
            q_ep_s.append(1.0 if q_info.get("reached_goal", False) else 0.0)

        c_rewards.append(np.mean(c_ep_r))
        q_rewards.append(np.mean(q_ep_r))
        c_succ.append(float(np.mean(c_ep_s)))
        q_succ.append(float(np.mean(q_ep_s)))

    c_r, q_r = np.array(c_rewards), np.array(q_rewards)
    print(f"Classical Agent Mean Reward: {np.mean(c_r):.4f} +/- {np.std(c_r):.4f} | Success Rate: {np.mean(c_succ)*100:.1f}%")
    print(f"Quantum Agent Mean Reward:   {np.mean(q_r):.4f} +/- {np.std(q_r):.4f} | Success Rate: {np.mean(q_succ)*100:.1f}%")

if __name__ == "__main__":
    for H in [2, 3, 4]:
        run_fixed_benchmark(episodes=50, horizon=H, shots=128)
