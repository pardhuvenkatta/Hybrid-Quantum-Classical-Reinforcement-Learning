# Debug Diagnostics Directory

This directory contains standalone diagnostic scripts used for historical debugging and parameter sensitivity checks:
- `debug_diff.py`: Historical parameter comparison diagnostic.
- `check_horizon_sensitivity.py`: Diagnostic for testing horizon sensitivity using legacy seeds.
- `verify_learning_and_convergence.py`: Diagnostic comparing all-episode vs. final-evaluation metrics using legacy seeds.
- `run_mini_test.py`: Fast mini-verification check (2 seeds) for local smoke testing.

Files in this directory are strictly excluded from all canonical benchmarks, experimental suites, and publication figure pipelines. Do not treat outputs as canonical publication results.
