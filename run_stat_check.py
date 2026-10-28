import os
import json
import numpy as np

def main():
    json_path = 'artifacts/canonical_results.json'
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"CRITICAL ERROR: '{json_path}' missing. Run 'run_canonical_experiments.py' first.")

    with open(json_path) as f:
        data = json.load(f)

    if 'scaling_experiments' not in data:
        raise KeyError(f"CRITICAL ERROR: 'scaling_experiments' key missing in '{json_path}'.")

    scaling = data['scaling_experiments']
    num_horizons = len(scaling)

    print(f"--- Statistical Analysis for {json_path} ---")
    for k, v in scaling.items():
        if 'quantum_mean_reward' not in v or 'classical_mean_reward' not in v:
            raise KeyError(f"Missing reward metrics in key '{k}'")
        
        q_r = v['quantum_mean_reward']
        c_r = v['classical_mean_reward']
        d = v.get('cohens_d', 0.0)
        
        if 'p_val_bonferroni' in v:
            p_bonf = v['p_val_bonferroni']
        elif 'p_val_t' in v:
            p_bonf = min(1.0, float(v['p_val_t']) * num_horizons)
        elif 'p_val_uncorrected' in v:
            p_bonf = min(1.0, float(v['p_val_uncorrected']) * num_horizons)
        else:
            raise KeyError(f"No valid p-value key found in scaling entry '{k}'!")

        print(f"{k}: Quantum={q_r:.4f}, Classical={c_r:.4f}, Cohen_d={d:.4f}, Bonferroni_p={p_bonf:.4e}")

if __name__ == '__main__':
    main()
