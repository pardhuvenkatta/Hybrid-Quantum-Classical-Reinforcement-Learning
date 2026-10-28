import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Dict, Any, Tuple

def _run_seed_task(task_fn: Callable, seed: int, kwargs: Dict[str, Any]) -> Tuple[int, Any]:
    # Ensure per-worker determinism
    np.random.seed(seed)
    result = task_fn(seed=seed, **kwargs)
    return seed, result

def run_seeds_parallel(task_fn: Callable,
                       seeds: List[int],
                       task_kwargs: Dict[str, Any],
                       max_workers: int = None,
                       progress: bool = True) -> Dict[int, Any]:
    """
    Run task_fn(seed=seed, **task_kwargs) for each seed in parallel.
    Returns a dict { seed: result }.
    """
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    if max_workers <= 1:
        results: Dict[int, Any] = {}
        for s in sorted(seeds):
            _, res = _run_seed_task(task_fn, s, task_kwargs)
            results[s] = res
            if progress:
                print(f"  [serial] seed {s} done ({len(results)}/{len(seeds)})", flush=True)
        return results

    results: Dict[int, Any] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_seed_task, task_fn, s, task_kwargs): s for s in seeds}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                seed, res = fut.result()
                results[seed] = res
                if progress:
                    print(f"  [parallel] seed {seed} done ({len(results)}/{len(seeds)})", flush=True)
            except Exception as e:
                print(f"  [parallel] seed {s} FAILED: {e}", flush=True)
                raise
    return results

def run_agents_parallel_for_seed(agent_task_fns: Dict[str, Callable],
                                 seed: int,
                                 max_workers: int = None) -> Dict[str, Any]:
    """
    Run multiple agent task functions in parallel for a single seed.
    Each task fn takes seed as its first positional argument.
    Returns { agent_name: result }.
    """
    if max_workers is None:
        max_workers = len(agent_task_fns)

    out: Dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {name: ex.submit(fn, seed) for name, fn in agent_task_fns.items()}
        for name, fut in futures.items():
            out[name] = fut.result()
    return out
