"""Paired effects with task-level resampling across repeated executions."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def paired_effect(rows: list[dict], base: str, target: str, field: str = "passed",
                  cluster_field: str = "task_id", bootstrap_samples: int = 5000, seed: int = 0) -> dict:
    pairs = {}
    for row in rows:
        if row['condition'] not in {base, target}: continue
        key = (str(row['task_id']), str(row.get('repeat', 0)))
        bucket = pairs.setdefault(key, {})
        if row['condition'] in bucket:
            raise ValueError(f"Duplicate task/repeat/condition: {key}/{row['condition']}")
        bucket[row['condition']] = row
    effects, clusters, per_repeat = {}, {}, {}
    for (task, repeat), pair in pairs.items():
        if base not in pair or target not in pair:
            raise ValueError(f"Unpaired observations: {task}/{repeat}")
        b, t = bool(pair[base][field]), bool(pair[target][field])
        effects.setdefault(task, []).append(int(t) - int(b))
        clusters[task] = str(pair[base].get(cluster_field, task))
        r = per_repeat.setdefault(repeat, {'n': 0, 'gains': 0, 'losses': 0, 'base_passed': 0, 'target_passed': 0})
        r['n'] += 1; r['gains'] += (not b and t); r['losses'] += (b and not t)
        r['base_passed'] += b; r['target_passed'] += t
    if not effects: raise ValueError("No paired tasks.")
    repeats_per_task = {len(v) for v in effects.values()}
    if len(repeats_per_task) != 1:
        raise ValueError("Tasks have different numbers of repeated observations.")
    task_means = {k: float(np.mean(v)) for k, v in effects.items()}
    by_cluster = {}
    for task, value in task_means.items(): by_cluster.setdefault(clusters[task], []).append(value)
    group_values = [by_cluster[key] for key in sorted(by_cluster)]
    sums = np.array([sum(v) for v in group_values]); counts = np.array([len(v) for v in group_values])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(group_values), size=(bootstrap_samples, len(group_values)))
    distribution = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    for r in per_repeat.values():
        n = r['gains'] + r['losses']
        r['exact_mcnemar_p'] = (float(binomtest(r['gains'], n, 0.5).pvalue) if n else 1.0) if cluster_field == 'task_id' else None
        r['paired_difference'] = (r['gains'] - r['losses']) / r['n']
    return {'base': base, 'target': target, 'num_tasks': len(effects), 'num_clusters': len(group_values),
            'repeats_per_task': next(iter(repeats_per_task)), 'cluster_field': cluster_field,
            'paired_difference': float(np.mean(list(task_means.values()))),
            'ci95': np.quantile(distribution, [.025, .975]).tolist(), 'per_repeat': per_repeat}
