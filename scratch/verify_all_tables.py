import json

print("=== TABLE 1 VERIFICATION ===")
canon = json.load(open("artifacts/canonical_results.json"))
for h_str in ["2", "3", "4", "5", "6"]:
    h_data = canon["scaling_experiments"][f"H{h_str}"]
    cm, c_ci = h_data["classical_mean_reward"], h_data["classical_ci95"]
    qm, q_ci = h_data["quantum_mean_reward"], h_data["quantum_ci95"]
    cs = h_data["classical_success_rate"] * 100
    qs = h_data["quantum_success_rate"] * 100
    cd = h_data["cohens_d"]
    bp = h_data["p_val_bonferroni"]
    print(f"H={h_str}: Class={cm:+.4f} +/- {c_ci:.4f} ({cs:.1f}%), Quant={qm:+.4f} +/- {q_ci:.4f} ({qs:.1f}%), d={cd:+.4f}, p={bp:.2e}")

print("\n=== TABLE 2 VERIFICATION ===")
grand = json.load(open("artifacts/grand_hybrid_comparison_results.json"))
for env in ["CartPole-v1", "Acrobot-v1", "FrozenLake-v1", "Maze-8x8"]:
    print(env, ":")
    for m in ["Classical_Baseline", "PPO_Q", "Q_SpiRL", "QGRNN", "QBRL", "HQCPG", "Adaptive_Q_Ours"]:
        d = grand["primary_benchmarks"][env][m]
        print(f"  {m:20s}: {d['mean_reward']:+.2f} +/- {d['ci95']:.2f}")

print("\n=== TABLE 3 VERIFICATION ===")
new_arch = json.load(open("artifacts/new_architectures_benchmark_results.json"))
for m in new_arch["metadata"]["architectures_evaluated"]:
    cp = new_arch["primary_benchmarks"]["CartPole-v1"][m]
    fl = new_arch["primary_benchmarks"]["FrozenLake-v1"][m]
    mz = new_arch["primary_benchmarks"]["Maze-8x8"][m]
    print(f"  {m:28s}: CP={cp['mean_reward']:+.2f} +/- {cp['ci95']:.2f}, FL={fl['mean_reward']:+.4f} +/- {fl['ci95']:.4f}, MZ={mz['mean_reward']:+.2f} +/- {mz['ci95']:.2f}")

print("\n=== TABLE 4 VERIFICATION ===")
oracle = json.load(open("artifacts/oracle_comparison_results.json"))
for h_str in ["2", "3", "4", "5", "6"]:
    o = oracle[f"H{h_str}"]
    c = o["classical"]
    eq = o["quantum_exact"]
    cq = o["quantum_continuous"]
    st = o["stats"]["exact_vs_continuous"]
    print(f"H={h_str}: Class={c['mean']:+.4f} +/- {c['ci95']:.4f}, Exact={eq['mean']:+.4f} +/- {eq['ci95']:.4f} ({eq['success']*100:.1f}%), Cont={cq['mean']:+.4f} +/- {cq['ci95']:.4f} ({cq['success']*100:.1f}%), t={st['t_stat']:+.2f}, p={st['p_val_bonferroni']:.2e}, d={st['cohens_d']:+.2f}")

print("\n=== TABLE 5 VERIFICATION ===")
for v in ["Full_QWM", "No_Oracle", "No_Interference", "No_Oracle_No_Interference"]:
    abl = canon["ablation_experiments_H4"][v]
    print(f"{v:28s}: Mean={abl['mean_reward']:+.4f} +/- {abl['ci95']:.4f}, Succ={abl['success_rate']*100:.1f}%")
