#!/usr/bin/env python
"""Evaluate frozen protocol tasks and retain task/repeat pairing."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tddexp.execution import execute
from tddexp.io import read_jsonl, write_jsonl, append_jsonl
from tddexp.lcb_execution import grade


def lcb_input_key(value, call_based):
    if call_based:
        try: return repr([json.loads(line) for line in value.strip().splitlines()])
        except ValueError: pass
    return value.strip()


def evaluate_one(item):
    row, task, repeat, timeout = item
    timeout = task.get('evaluation_timeout_seconds', timeout)
    result = {k: row[k] for k in ['task_id', 'condition']}
    result.update(repeat=row.get('repeat', repeat), source_dataset=task['source_dataset'],
                  experiment=task.get('experiment', 'semantic' if task['source_dataset']=='semantic_pairs' else 'main'),
                  source_task_id=task.get('source_task_id', task['task_id']),
                  family_id=task.get('family_id', task['task_id']),
                  model_path=row.get('model_path'), finish_reason=row.get('finish_reason'),
                  generated_token_count=row.get('generated_token_count'))
    code = row['candidate_code']
    if task['source_dataset'] == 'semantic_pairs':
        tests = task['heldout_a'] + task['heldout_b'] + task['heldout_common']
        output = execute(code, tests=tests, timeout=timeout)
        passed = output['passed']
        na, nb = len(task['heldout_a']), len(task['heldout_b'])
        a, b, common = passed[:na], passed[na:na+nb], passed[na+nb:]
        result.update(rule_a_passed=len(a)==na and all(a) and all(common),
                      rule_b_passed=len(b)==nb and all(b) and all(common),
                      rule_a_input_accuracy=sum(a)/na, rule_b_input_accuracy=sum(b)/nb,
                      common_passed=all(common), execution_status=output['status'])
        intended = 'b' if row['condition'] in {'tests_b', 'explicit_b'} else 'a'
        result['passed'] = result['rule_' + intended + '_passed']
    else:
        # Independent processes keep the full suite from mutating held-out evaluation state.
        official = execute(code, tests=[task['official_test']], timeout=timeout)
        heldout = execute(code, tests=[task['heldout_test']], timeout=timeout)
        result.update(official_passed=bool(official['passed'] and all(official['passed'])),
                      heldout_passed=bool(heldout['passed'] and all(heldout['passed'])),
                      execution_status=heldout['status'], heldout_counts=task.get('heldout_counts'))
        result['passed'] = result['heldout_passed']
    return result


def evaluate_lcb(rows, tasks, repeat, root, timeout, workers, on_result=None):
    sys.path.insert(0, str(Path(root).resolve()))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    official = {r['question_id']: r for r in read_jsonl('data/livecodebench_release_v6_minus_v5.jsonl')}
    keys = {'question_title', 'question_content', 'platform', 'question_id', 'contest_id', 'contest_date',
            'starter_code', 'difficulty', 'public_test_cases', 'private_test_cases', 'metadata'}
    problems = [CodeGenerationProblem(**{k:v for k,v in official[tasks[r['task_id']]['source_task_id']].items() if k in keys}) for r in rows]
    samples = [p.get_evaluation_sample() for p in problems]
    # Run the same official checker with only original private tests.
    private_samples = []
    for problem, sample in zip(problems, samples):
        raw = json.loads(sample['input_output'])
        public_n = len(problem.public_test_cases)
        call_based = bool(raw.get('fn_name'))
        public_inputs = {lcb_input_key(x, call_based) for x in raw['inputs'][:public_n]}
        keep = [i for i in range(public_n, len(raw['inputs'])) if lcb_input_key(raw['inputs'][i], call_based) not in public_inputs]
        for key in ['inputs', 'outputs']:
            raw[key] = [raw[key][i] for i in keep]
        if not raw['inputs']: raise ValueError('No private LiveCodeBench tests.')
        private_samples.append({**sample, 'input_output': json.dumps(raw)})
    def evaluate_pair(index):
        row = rows[index]
        official_result = grade(samples[index], row['candidate_code'], root, int(timeout))
        heldout_result = grade(private_samples[index], row['candidate_code'], root, int(timeout))
        return dict(task_id=row['task_id'], source_task_id=tasks[row['task_id']]['source_task_id'],
                    source_dataset='livecodebench', experiment=tasks[row['task_id']].get('experiment', 'main'),
                    condition=row['condition'], repeat=row.get('repeat', repeat), model_path=row.get('model_path'),
                    finish_reason=row.get('finish_reason'), generated_token_count=row.get('generated_token_count'),
                    official_passed=official_result['passed'], heldout_passed=heldout_result['passed'],
                    passed=heldout_result['passed'], memory_limit_mb=4096,
                    official_execution_status=official_result['status'], heldout_execution_status=heldout_result['status'])
    results = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_pair, i): i for i in range(len(rows))}
        for future in as_completed(futures):
            result = future.result()
            results[futures[future]] = result
            if on_result is not None:
                on_result(result)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True); p.add_argument('--generations', required=True)
    p.add_argument('--output', required=True); p.add_argument('--repeat', type=int, default=0)
    p.add_argument('--workers', type=int, default=8); p.add_argument('--timeout', type=float, default=15)
    p.add_argument('--lcb-root', default='.tmp/LiveCodeBench')
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--checkpoint')
    a = p.parse_args()
    tasks = {r['task_id']:r for r in read_jsonl(a.dataset)}
    rows = read_jsonl(a.generations)
    repeats = sorted({r.get('repeat', a.repeat) for r in rows})
    expected = {(t['task_id'], c, repeat) for t in tasks.values() for c in t['condition_prompts'] for repeat in repeats}
    actual = [(r['task_id'], r['condition'], r.get('repeat',a.repeat)) for r in rows]
    if len(actual) != len(set(actual)) or not set(actual).issubset(expected) or (not a.allow_partial and set(actual) != expected):
        raise SystemExit('Generation file does not contain exactly one row per frozen task and condition.')
    def key(row): return row['task_id'], row['condition'], row.get('repeat', a.repeat)
    cached = read_jsonl(a.checkpoint) if a.checkpoint and Path(a.checkpoint).exists() else []
    cache = {key(r):r for r in cached if r.get('evaluation_protocol') == 3
             or (r.get('evaluation_protocol') == 2 and r['result']['source_dataset'] != 'livecodebench')}
    completed = {key(r):cache[key(r)]['result'] for r in rows if key(r) in cache and cache[key(r)]['candidate_code'] == r['candidate_code']
                 and cache[key(r)].get('timeout_seconds', a.timeout) == tasks[r['task_id']].get('evaluation_timeout_seconds', a.timeout)}
    results = list(completed.values())
    codes = {key(r):r['candidate_code'] for r in rows}
    rows = [r for r in rows if key(r) not in completed]
    def save(result):
        results.append(result)
        if a.checkpoint:
            append_jsonl(a.checkpoint,dict(task_id=result['task_id'],condition=result['condition'],repeat=result['repeat'],
                        evaluation_protocol=3,candidate_code=codes[key(result)],result=result,
                        timeout_seconds=tasks[result['task_id']].get('evaluation_timeout_seconds',a.timeout)))
        if len(results) % 64 == 0:
            print(json.dumps({'evaluated':len(results)}),flush=True)
    function_rows = [r for r in rows if tasks[r['task_id']]['source_dataset'] != 'livecodebench']
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(evaluate_one,(r,tasks[r['task_id']],a.repeat,a.timeout)) for r in function_rows]
        for future in as_completed(futures): save(future.result())
    lcb = [r for r in rows if tasks[r['task_id']]['source_dataset'] == 'livecodebench']
    if lcb:
        evaluate_lcb(lcb, tasks, a.repeat, a.lcb_root, a.timeout, a.workers, on_result=save)
    write_jsonl(a.output, results)
    print(json.dumps({'evaluated':len(results), 'output':a.output}))


if __name__ == '__main__': main()
