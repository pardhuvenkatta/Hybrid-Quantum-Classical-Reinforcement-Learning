import os
import sys
import json
import time
import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Tuple

from src.environments.grid_4x4 import Grid4x4Env
from src.agents.tabular_q_agent import TabularQAgent
from src.world_models.classical_transition_model import ClassicalTransitionModel
from src.world_models.quantum_transition_model import QuantumTransitionModel
from src.world_models.hybrid_transition_model import HybridTransitionModel
from src.planners.sampling_planner import SamplingPlanner

# Module-level defaults. Override via
#   `import run_hqwm_fsp_experiments as m; m.num_seeds = 2; m.run_hqwm_fsp_experiments()`
# for smoke tests. Do not shadow these names inside the function body.
num_seeds = 20

N_COMPARISONS = 7  # Compared against Classical World Model

def calculate_cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(n1 + n2 - 2, 1))
    if s_pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / s_pooled

def get_ci95(std_val, N=20):
    return 1.96 * (std_val / np.sqrt(N))

class EnsembleModel:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2
    def predict_next_state_distribution(self, s, a):
        p1 = self.m1.predict_next_state_distribution(s, a)
        p2 = self.m2.predict_next_state_distribution(s, a)
        p = 0.5 * (p1 + p2)
        return p / np.sum(p)
    def compute_nll(self, states, actions, next_states):
        nll1 = self.m1.compute_nll(states, actions, next_states)
        nll2 = self.m2.compute_nll(states, actions, next_states)
        return float(0.5 * (nll1 + nll2))

def run_finite_shots_sweep(num_seeds_val: int = 20, out_path: str = "artifacts/hqwm_fsp_results.json"):
    print("Evaluating Robustness to Finite Measurement Shots on Held-Out Transitions...", flush=True)
    shots_list = [100, 500, 1000, 5000, 10000]

    env_eval = Grid4x4Env(slip_prob=0.10)
    all_transitions = []
    for r in range(4):
        for c in range(4):
            for act in range(4):
                next_p, _, _ = env_eval.simulate_step((r, c), act)
                all_transitions.append(((r, c), act, next_p))

    # Deterministic split — do not shuffle with an unseeded RNG
    # 60% train (38 tuples), 40% eval (26 tuples) held-out
    rng = np.random.default_rng(20241028)
    idx = rng.permutation(len(all_transitions))
    n_train = int(0.6 * len(all_transitions))   # 60% train (38 tuples)
    train_transitions = [all_transitions[i] for i in idx[:n_train]]
    eval_transitions  = [all_transitions[i] for i in idx[n_train:]]

    np.random.seed(42)
    eval_qm = QuantumTransitionModel(num_blocks=3, lr=0.05, eps_smoothing=0.02)

    # Optimize transition model parameters to fit train_transitions only
    from scipy.optimize import minimize
    def loss_fn(flat_params):
        params = flat_params.reshape(eval_qm.params.shape)
        loss = 0.0
        for s, a, ns in train_transitions:
            x = eval_qm._encode_input(s, a)
            psi = eval_qm.forward_statevector(x, params)
            target = ns[0] * 4 + ns[1]
            p = np.abs(psi[target])**2
            p = (1.0 - 0.02) * p + 0.02 / 16.0
            loss -= np.log(p)
        return loss / len(train_transitions)

    init_p = eval_qm.params.copy().flatten()
    res = minimize(loss_fn, init_p, method='COBYLA', options={'maxiter': 250})
    eval_qm.params = res.x.reshape(eval_qm.params.shape)

    robustness_finite_shots = {}
    for sh in shots_list:
        nll_list = []
        for seed in range(100, 100 + num_seeds_val):
            np.random.seed(seed)
            for s_pos, act, next_pos in eval_transitions:
                probs = eval_qm.predict_next_state_distribution(s_pos, act, shots=sh)
                target = next_pos[0] * 4 + next_pos[1] if isinstance(next_pos, tuple) else next_pos % 16
                nll_list.append(-np.log(max(probs[target], 1e-12)))
        robustness_finite_shots[f"shots_{sh}"] = {
            "mean_nll": float(np.mean(nll_list)),
            "std_nll": float(np.std(nll_list))
        }

    return robustness_finite_shots

def run_hqwm_fsp_experiments():
    start_time = time.time()

    print("=" * 73, flush=True)
    print(f"  HYBRID QUANTUM-CLASSICAL WORLD MODEL FOR FUTURE-STATE PLANNING (N={num_seeds})  ", flush=True)
    print(f"  Robustness sub-experiments: {num_seeds} seeds each (full benchmark set)  ", flush=True)
    print("=" * 73, flush=True)

    seeds = list(range(100, 100 + num_seeds))
    episodes = 500
    eval_interval = 10  # Increased training frequency (every 10 episodes)
    train_steps_per_interval = 3  # Equal gradient steps for all models

    model_names = [
        "1_Model_Free_Q_Learning",
        "2_Classical_World_Model",
        "3_Quantum_Only_World_Model",
        "4_Proposed_Hybrid_HQWM",
        "5_Two_Classical_Ensemble",
        "6_Capacity_Matched_Classical",
        "7_Untrained_Quantum_Control",
        "8_Exact_Transition_Reference"
    ]

    results = {
        "metadata": {
            "num_seeds": num_seeds,
            "episodes": episodes,
            "eval_interval": eval_interval,
            "train_steps_per_interval": train_steps_per_interval,
            "n_comparisons": N_COMPARISONS,
            "models_evaluated": model_names,
            "grid_size": "4x4",
            "slip_prob": 0.10,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "models": {},
        "alpha_star_evolution": {},
        "robustness_slip_prob": {},
        "robustness_finite_shots": {}
    }
    results["metadata"]["robustness_seed_count"] = num_seeds
    results["metadata"]["robustness_seed_note"] = (
        "Slip-probability and finite-shot robustness loops use the same "
        f"{num_seeds}-seed set as the main benchmark. Expect end-to-end runtime "
        f"~{(num_seeds / 5):.0f}x longer than the pre-Round-5 5-seed version."
    )

    data_store = {m: {
        "returns": [],
        "success": [],
        "ep_hist": [],
        "nll_test": [],
        "wall_clock": [],
        "model_queries": [],
        "circuit_executions": [],
        "total_shots": [],
        "param_count": 0
    } for m in model_names}

    # Track alpha* across all seeds
    alpha_history_all_seeds: Dict[int, List[float]] = {s: [] for s in seeds}

    for seed in seeds:
        print(f"Executing Seed {seed}/{seeds[-1]}...", flush=True)
        np.random.seed(seed)

        env = Grid4x4Env(slip_prob=0.10)

        # 2.1 Independent Model Instances (No shared objects)
        c_model = ClassicalTransitionModel(hidden_dim=32, lr=0.02)
        q_model = QuantumTransitionModel(num_blocks=3, lr=0.05)

        h_c_model = ClassicalTransitionModel(hidden_dim=32, lr=0.02)
        h_q_model = QuantumTransitionModel(num_blocks=3, lr=0.05)
        h_model = HybridTransitionModel(h_c_model, h_q_model, initial_alpha=0.50)

        c1_ens = ClassicalTransitionModel(hidden_dim=32, lr=0.02)
        c2_ens = ClassicalTransitionModel(hidden_dim=32, lr=0.02)

        ensemble_model = EnsembleModel(c1_ens, c2_ens)

        cap_matched_c = ClassicalTransitionModel(hidden_dim=64, lr=0.02)
        un_c = ClassicalTransitionModel(hidden_dim=32, lr=0.02)
        untrained_q = QuantumTransitionModel(num_blocks=3, lr=0.0)
        un_hybrid = HybridTransitionModel(un_c, untrained_q, initial_alpha=0.50)

        agents = {m: TabularQAgent(num_actions=4, lr=0.1, gamma=0.95, epsilon=0.15) for m in model_names}
        planners = {m: SamplingPlanner(num_actions=4, horizon=3, num_trajectories=16, gamma=0.95) for m in model_names if m != "1_Model_Free_Q_Learning"}

        # 2.2 Separate Replay Buffers per Model
        model_buffers = {m: {"s": [], "a": [], "ns": []} for m in model_names}

        # 2.4 True Validation & Test Sets (Separate holdout sets)
        val_s, val_a, val_ns = [], [], []
        test_s, test_a, test_ns = [], [], []

        # Parameter Counts
        def _param_count(agent, default=0):
            return int(getattr(agent, "parameter_count", default))

        data_store["1_Model_Free_Q_Learning"]["param_count"] = _param_count(
            agents["1_Model_Free_Q_Learning"],
            len(agents["1_Model_Free_Q_Learning"].q_table) if hasattr(agents["1_Model_Free_Q_Learning"], "q_table") else 0
        )
        data_store["2_Classical_World_Model"]["param_count"] = _param_count(c_model, 0)
        data_store["3_Quantum_Only_World_Model"]["param_count"] = _param_count(q_model, 0)
        data_store["4_Proposed_Hybrid_HQWM"]["param_count"] = _param_count(h_c_model, 0) + _param_count(h_q_model, 0) + 1
        data_store["5_Two_Classical_Ensemble"]["param_count"] = _param_count(c1_ens, 0) + _param_count(c2_ens, 0)
        data_store["6_Capacity_Matched_Classical"]["param_count"] = _param_count(cap_matched_c, 0)
        data_store["7_Untrained_Quantum_Control"]["param_count"] = _param_count(un_c, 0) + _param_count(untrained_q, 0)
        data_store["8_Exact_Transition_Reference"]["param_count"] = 0

        seed_ep_returns = {m: [] for m in model_names}
        seed_success = {m: [] for m in model_names}
        seed_times = {m: 0.0 for m in model_names}

        for ep in range(episodes):
            for m in model_names:
                t0 = time.time()
                pos = env.reset(seed=seed * 1000 + ep)
                agent = agents[m]
                planner = planners.get(m, None)

                total_r, done = 0.0, False
                step_cnt = 0

                while not done:
                    step_cnt += 1
                    if m == "1_Model_Free_Q_Learning":
                        action = agent.select_action(pos)
                    elif m == "2_Classical_World_Model":
                        action = planner.select_action(pos, c_model, agent.q_table)
                    elif m == "3_Quantum_Only_World_Model":
                        action = planner.select_action(pos, q_model, agent.q_table)
                    elif m == "4_Proposed_Hybrid_HQWM":
                        action = planner.select_action(pos, h_model, agent.q_table)
                    elif m == "5_Two_Classical_Ensemble":
                        action = planner.select_action(pos, ensemble_model, agent.q_table)
                    elif m == "6_Capacity_Matched_Classical":
                        action = planner.select_action(pos, cap_matched_c, agent.q_table)
                    elif m == "7_Untrained_Quantum_Control":
                        action = planner.select_action(pos, un_hybrid, agent.q_table)
                    elif m == "8_Exact_Transition_Reference":
                        action = planner.select_action(pos, env, agent.q_table)

                    next_p, r, done, info = env.step(action)
                    agent.update(pos, action, r, next_p, done)

                    # Partition transitions: 70% train buffer, 15% val buffer, 15% test buffer
                    roll = np.random.rand()
                    if roll < 0.15:
                        val_s.append(pos)
                        val_a.append(action)
                        val_ns.append(next_p)
                    elif roll < 0.30:
                        test_s.append(pos)
                        test_a.append(action)
                        test_ns.append(next_p)
                    else:
                        model_buffers[m]["s"].append(pos)
                        model_buffers[m]["a"].append(action)
                        model_buffers[m]["ns"].append(next_p)

                    total_r += r
                    pos = next_p

                seed_ep_returns[m].append(total_r)
                seed_success[m].append(1.0 if info.get("reached_goal", False) else 0.0)
                seed_times[m] += time.time() - t0

            # 2.3 Equal Training Budgets (Trained every eval_interval episodes)
            if (ep + 1) % eval_interval == 0:
                steps = train_steps_per_interval

                # Train Model 2
                buf2 = model_buffers["2_Classical_World_Model"]
                if len(buf2["s"]) > 10:
                    c_model.train_on_batch(buf2["s"][-150:], buf2["a"][-150:], buf2["ns"][-150:], steps=steps)

                # Train Model 3
                buf3 = model_buffers["3_Quantum_Only_World_Model"]
                if len(buf3["s"]) > 10:
                    q_model.train_on_batch(buf3["s"][-100:], buf3["a"][-100:], buf3["ns"][-100:], steps=steps)

                # Train Model 4 (Hybrid: train both c and q components independently)
                buf4 = model_buffers["4_Proposed_Hybrid_HQWM"]
                if len(buf4["s"]) > 10:
                    h_c_model.train_on_batch(buf4["s"][-150:], buf4["a"][-150:], buf4["ns"][-150:], steps=steps)
                    h_q_model.train_on_batch(buf4["s"][-100:], buf4["a"][-100:], buf4["ns"][-100:], steps=steps)

                # Train Model 5 (Ensemble)
                buf5 = model_buffers["5_Two_Classical_Ensemble"]
                if len(buf5["s"]) > 10:
                    c1_ens.train_on_batch(buf5["s"][-150:], buf5["a"][-150:], buf5["ns"][-150:], steps=steps)
                    c2_ens.train_on_batch(buf5["s"][-150:], buf5["a"][-150:], buf5["ns"][-150:], steps=steps)

                # Train Model 6 (Capacity Matched)
                buf6 = model_buffers["6_Capacity_Matched_Classical"]
                if len(buf6["s"]) > 10:
                    cap_matched_c.train_on_batch(buf6["s"][-150:], buf6["a"][-150:], buf6["ns"][-150:], steps=steps)

                # Train Model 7 (Only classical component trains, quantum remains untrained)
                buf7 = model_buffers["7_Untrained_Quantum_Control"]
                if len(buf7["s"]) > 10:
                    un_c.train_on_batch(buf7["s"][-150:], buf7["a"][-150:], buf7["ns"][-150:], steps=steps)

                # 2.4 Tune alpha* on held-out VALIDATION set
                if len(val_s) > 5:
                    alpha_star = h_model.update_validation_alpha(val_s[-100:], val_a[-100:], val_ns[-100:])
                    alpha_history_all_seeds[seed].append(float(alpha_star))

        # 2.6 Bookkeeping per Model
        for m in model_names:
            data_store[m]["returns"].append(np.mean(seed_ep_returns[m]))
            # Episode-averaged success rate across all training episodes for this seed
            data_store[m]["success"].append(float(np.mean(seed_success[m])))
            data_store[m]["ep_hist"].append(seed_ep_returns[m])
            data_store[m]["wall_clock"].append(seed_times[m])
            data_store[m]["model_queries"].append(planners[m].total_model_queries if m in planners else 0)

            # Record circuit executions accurately per model instance
            if m == "3_Quantum_Only_World_Model":
                data_store[m]["circuit_executions"].append(q_model.total_circuit_executions)
                data_store[m]["total_shots"].append(q_model.total_shots_executed)
            elif m == "4_Proposed_Hybrid_HQWM":
                data_store[m]["circuit_executions"].append(h_q_model.total_circuit_executions)
                data_store[m]["total_shots"].append(h_q_model.total_shots_executed)
            elif m == "7_Untrained_Quantum_Control":
                data_store[m]["circuit_executions"].append(untrained_q.total_circuit_executions)
                data_store[m]["total_shots"].append(untrained_q.total_shots_executed)
            else:
                data_store[m]["circuit_executions"].append(0)
                data_store[m]["total_shots"].append(0)

            # 2.4 Compute NLL on held-out TEST set
            if len(test_s) > 0:
                if m == "2_Classical_World_Model":
                    nll_val = c_model.compute_nll(test_s, test_a, test_ns)
                elif m == "3_Quantum_Only_World_Model":
                    nll_val = q_model.compute_nll(test_s, test_a, test_ns)
                elif m == "4_Proposed_Hybrid_HQWM":
                    nll_val = h_model.compute_nll(test_s, test_a, test_ns)
                elif m == "5_Two_Classical_Ensemble":
                    nll_val = ensemble_model.compute_nll(test_s, test_a, test_ns)
                elif m == "6_Capacity_Matched_Classical":
                    nll_val = cap_matched_c.compute_nll(test_s, test_a, test_ns)
                elif m == "7_Untrained_Quantum_Control":
                    nll_val = un_hybrid.compute_nll(test_s, test_a, test_ns)
                else:
                    nll_val = 0.5
            else:
                nll_val = 0.5
            data_store[m]["nll_test"].append(nll_val)

    # Compute alpha* evolution statistics across all 20 seeds
    intervals = list(range(eval_interval, episodes + 1, eval_interval))
    alpha_evolution_summary = []
    for idx, interval in enumerate(intervals):
        vals_at_idx = [alpha_history_all_seeds[s][idx] for s in seeds if idx < len(alpha_history_all_seeds[s])]
        if vals_at_idx:
            alpha_evolution_summary.append({
                "interval": interval,
                "mean_alpha_star": float(np.mean(vals_at_idx)),
                "std_alpha_star": float(np.std(vals_at_idx)),
                "ci95_alpha_star": float(get_ci95(np.std(vals_at_idx), len(vals_at_idx)))
            })
    results["alpha_star_evolution"] = alpha_evolution_summary

    classical_arr = np.array(data_store["2_Classical_World_Model"]["returns"])

    for m in model_names:
        arr = np.array(data_store[m]["returns"])
        succ = np.array(data_store[m]["success"])

        t_stat, p_val = stats.ttest_rel(arr, classical_arr) if len(np.unique(arr - classical_arr)) > 1 else (0.0, 1.0)
        d_val = calculate_cohens_d(arr, classical_arr)
        p_bonf = min(1.0, float(p_val) * N_COMPARISONS)

        results["models"][m] = {
            "mean_return": float(np.mean(arr)),
            "std_return": float(np.std(arr)),
            "ci95_return": float(get_ci95(np.std(arr))),
            "success_rate": float(np.mean(succ)),
            "mean_test_nll": float(np.mean(data_store[m]["nll_test"])),
            "std_test_nll": float(np.std(data_store[m]["nll_test"])),
            "wall_clock_time": float(np.mean(data_store[m]["wall_clock"])),
            "model_queries": float(np.mean(data_store[m]["model_queries"])),
            "circuit_executions": float(np.mean(data_store[m]["circuit_executions"])),
            "total_shots": float(np.mean(data_store[m]["total_shots"])),
            "parameter_count": data_store[m]["param_count"],
            "stats_vs_classical": {
                "t_stat": float(t_stat),
                "p_val": float(p_val),
                "p_val_bonferroni": float(p_bonf),
                "cohens_d": float(d_val)
            }
        }
        print(f"Model: {m:32s} | Return: {np.mean(arr):6.2f} ± {np.std(arr):4.2f} | Succ: {np.mean(succ)*100:5.1f}% | d: {d_val:+5.2f} (p_bonf={p_bonf:.4e})", flush=True)

    # 2.5 Fix Robustness to Slip Probability: Train agents first, then evaluate
    print("\nEvaluating Robustness to Slip Probability (Trained Agents)...", flush=True)
    slips = [0.05, 0.10, 0.20, 0.30, 0.40]
    for sl in slips:
        results["robustness_slip_prob"][f"p_{sl}"] = {}
        for m in ["1_Model_Free_Q_Learning", "2_Classical_World_Model", "4_Proposed_Hybrid_HQWM", "5_Two_Classical_Ensemble"]:
            env_rob = Grid4x4Env(slip_prob=sl)
            r_list = []
            for seed in range(100, 100 + num_seeds):
                np.random.seed(seed)
                agent_rob = TabularQAgent(num_actions=4, lr=0.1, gamma=0.95, epsilon=0.15)
                planner = SamplingPlanner(num_actions=4, horizon=3, num_trajectories=16, gamma=0.95)

                if m == "2_Classical_World_Model":
                    rob_model = ClassicalTransitionModel(hidden_dim=32, lr=0.02)
                elif m == "4_Proposed_Hybrid_HQWM":
                    h_c = ClassicalTransitionModel(hidden_dim=32, lr=0.02)
                    h_q = QuantumTransitionModel(num_blocks=3, lr=0.05)
                    rob_model = HybridTransitionModel(h_c, h_q, initial_alpha=0.50)
                elif m == "5_Two_Classical_Ensemble":
                    c1_m = ClassicalTransitionModel(hidden_dim=32, lr=0.02)
                    c2_m = ClassicalTransitionModel(hidden_dim=32, lr=0.02)
                    rob_model = EnsembleModel(c1_m, c2_m)
                else:
                    rob_model = None

                tr_s, tr_a, tr_ns = [], [], []

                # Training phase
                for _ in range(500):
                    p = env_rob.reset()
                    d = False
                    while not d:
                        a = agent_rob.select_action(p)
                        np_p, r, d, _ = env_rob.step(a)
                        agent_rob.update(p, a, r, np_p, d)
                        tr_s.append(p)
                        tr_a.append(a)
                        tr_ns.append(np_p)
                        p = np_p

                # Train transition models on collected transitions
                if m == "2_Classical_World_Model":
                    if len(tr_s) > 10:
                        rob_model.train_on_batch(tr_s[-200:], tr_a[-200:], tr_ns[-200:], steps=20)
                elif m == "4_Proposed_Hybrid_HQWM":
                    if len(tr_s) > 10:
                        h_c.train_on_batch(tr_s[-200:], tr_a[-200:], tr_ns[-200:], steps=20)
                        h_q.train_on_batch(tr_s[-100:], tr_a[-100:], tr_ns[-100:], steps=20)
                elif m == "5_Two_Classical_Ensemble":
                    if len(tr_s) > 10:
                        c1_m.train_on_batch(tr_s[-200:], tr_a[-200:], tr_ns[-200:], steps=20)
                        c2_m.train_on_batch(tr_s[-200:], tr_a[-200:], tr_ns[-200:], steps=20)

                # Evaluation phase
                for _ in range(50):
                    p = env_rob.reset()
                    tot_r, d = 0.0, False
                    while not d:
                        if m == "1_Model_Free_Q_Learning" or rob_model is None:
                            a = agent_rob.select_action(p)
                        else:
                            a = planner.select_action(p, rob_model, agent_rob.q_table)
                        np_p, r, d, _ = env_rob.step(a)
                        tot_r += r
                        p = np_p
                    r_list.append(tot_r)
            results["robustness_slip_prob"][f"p_{sl}"][m] = {
                "mean_reward": float(np.mean(r_list)),
                "std_reward": float(np.std(r_list))
            }

    # 2.5 Finite-Shot Robustness: Real held-out transition-predictor NLL
    out_path = "artifacts/hqwm_fsp_results.json"
    results["robustness_finite_shots"] = run_finite_shots_sweep(num_seeds_val=num_seeds, out_path=out_path)

    os.makedirs("artifacts", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nHQWM-FSP Benchmark completed in {elapsed:.2f} seconds.", flush=True)
    print(f"Results saved to: {out_path}", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--finite-shots-only", action="store_true",
                        help="Run only the finite-shots robustness sweep")
    args = parser.parse_args()
    if args.finite_shots_only:
        out_path = "artifacts/hqwm_fsp_results.json"
        existing = {}
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                existing = json.load(f)
        existing["robustness_finite_shots"] = run_finite_shots_sweep(num_seeds_val=num_seeds)
        os.makedirs("artifacts", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Updated {out_path} (robustness_finite_shots only; other keys preserved)")
    else:
        run_hqwm_fsp_experiments()
