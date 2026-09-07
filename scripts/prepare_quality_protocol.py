#!/usr/bin/env python
"""Select equal-size suites by union kill rate and validate a separate error pool."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tddexp.controlled import choose_test_suites, filter_evalplus_test, input_key, literal_call_inputs, prompt_with_tests
from tddexp.execution import execute
from tddexp.io import read_jsonl, write_jsonl
from tddexp.mutations import mutations
from tddexp.prompts import stable_seed


def prepare(item):
    task, candidates, old_codes, validation_codes = item
    task_id = task['task_id']; seed = stable_seed(task_id)
    code = task['reference_code']; entry = task['entry_point']
    tests, excluded_inputs = [], set()
    for item in candidates.get('valid_tests', []):
        test = item['test']
        try: key = input_key(literal_call_inputs(test, entry))
        except (ValueError, SyntaxError): continue
        if key in excluded_inputs: continue
        tests.append(test); excluded_inputs.add(key)
        if len(tests) >= 12: break
    if len(tests) < 5: return None, {'task_id':task_id,'reason':'fewer_than_five_candidate_inputs'}
    reference = execute(code, tests=tests, timeout=8)
    if reference['status'] != 'ok': return None, {'task_id':task_id,'reason':'reference_execution'}
    tests = [t for t, ok in zip(tests, reference['passed']) if ok]
    if len(tests) < 5: return None, {'task_id':task_id,'reason':'reference_validated_candidates'}
    development = task['condition_tests']['nl_tests']
    sequence_parameters = set()
    try:
        function = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == entry)
        arguments = [literal_call_inputs(t, entry) for t in development]
        for i, parameter in enumerate(function.args.args):
            if all(i < len(args) and isinstance(args[i], (list, tuple, str)) for args in arguments):
                sequence_parameters.add(parameter.arg)
    except (ValueError, StopIteration, SyntaxError): pass
    selection = [{'code':c, 'source':'historical_qwen25', 'operator':'generated', 'pool':'selection'} for c in old_codes]
    selection += mutations(code, 'selection', seed=seed, limit=96, sequence_parameters=sequence_parameters)
    validation = [{'code':c, 'source':'deepseek_main', 'operator':'generated', 'pool':'validation'} for c in validation_codes]
    validation += mutations(code, 'validation', seed=seed+1, limit=96, sequence_parameters=sequence_parameters)
    seen = {ast.unparse(ast.parse(code))}; pools = {'selection':[], 'validation':[]}
    for name, proposed in [('selection',selection), ('validation',validation)]:
        for candidate in proposed:
            try: key = ast.unparse(ast.parse(candidate['code']))
            except SyntaxError: continue
            if key in seen: continue
            seen.add(key)
            result = execute(candidate['code'], tests=development+tests, timeout=2.5)
            # Development examples certify a functional error independently of
            # the candidate suite used for quality ranking.
            if result['status'] != 'ok' or all(result['passed'][:len(development)]): continue
            candidate['id'] = f'{name}_{len(pools[name])}'
            candidate['candidate_test_passed'] = result['passed'][len(development):]
            pools[name].append(candidate)
            if len(pools[name]) == 8: break
    if min(map(len,pools.values())) < 3:
        return None, {'task_id':task_id,'reason':'insufficient_independent_errors',
                      'pool_sizes':{k:len(v) for k,v in pools.items()}}
    killed = [{p['id'] for p in pools['selection'] if not p['candidate_test_passed'][i]} for i in range(len(tests))]
    selection = choose_test_suites(tests, killed, seed=seed)
    def coverage(indices, pool):
        return sum(any(not p['candidate_test_passed'][i] for i in indices) for p in pool) / len(pool)
    metrics = {s:{'selection_kill_rate':coverage(selection[s],pools['selection']),
                 'validation_kill_rate':coverage(selection[s],pools['validation']),
                 'characters':sum(len(tests[i]) for i in selection[s])} for s in ['high','low','random']}
    if metrics['high']['selection_kill_rate'] <= metrics['low']['selection_kill_rate']:
        return None, {'task_id':task_id,'reason':'no_selection_quality_contrast'}
    excluded_inputs = {input_key(literal_call_inputs(t,entry)) for t in tests+development}
    try:
        heldout, counts = filter_evalplus_test(task['official_test'], excluded_inputs)
    except (ValueError, SyntaxError):
        return None, {'task_id':task_id,'reason':'heldout_input_filter'}
    result = dict(task)
    result.update(experiment='quality', heldout_test=heldout, heldout_counts=counts, candidate_tests=tests,
                  quality_metrics=metrics, wrong_program_pools=pools,
                  quality_selection={'count':3, 'objective':'union_kill_rate', 'length_tolerance':.15,
                                     'selection_indices':{s:selection[s] for s in ['high','low','random']},
                                     'seed':seed, 'validation_used_for_selection':False})
    result['condition_tests'] = {'nl_only':[], **{'quality_'+s:[tests[i] for i in selection[s]] for s in ['random','low','high']}}
    result['condition_prompts'] = {c:prompt_with_tests(task['prompt'], ts) for c,ts in result['condition_tests'].items()}
    return result, None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--main-tasks',default='data/controlled/main/tasks.jsonl')
    p.add_argument('--output-dir',default='data/controlled/quality')
    p.add_argument('--target-tasks',type=int,default=180);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--extra-selection-generations',nargs='*',default=[])
    p.add_argument('--extra-validation-generations',nargs='*',default=[])
    p.add_argument('--retry-excluded',action='store_true')
    a=p.parse_args(); out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    tasks=[r for r in read_jsonl(a.main_tasks) if r['source_dataset']!='livecodebench']
    rng=random.Random(271828)
    groups={s:[r for r in tasks if r['source_dataset']==s] for s in ['humanevalplus','mbppplus']}
    for group in groups.values():rng.shuffle(group)
    ordered=[]
    while any(groups.values()):
        for group in groups.values():
            if group:ordered.append(group.pop())
    candidates={}; weak={}; validation_codes={}
    for source,run in [('humanevalplus','humanevalplus_qwen_structured'),('mbppplus','mbppplus_qwen_full')]:
        for r in read_jsonl(f'data/synthetic_public_tests/{source}/synthetic_test_candidates.jsonl'):
            candidates[source+'::'+r['task_id']]=r
        for r in read_jsonl(f'outputs/{run}/eval_results.jsonl'):
            weak.setdefault(source+'::'+r['task_id'],[]).append(r['candidate_code'])
    for paths, destination in [(a.extra_selection_generations, weak), (a.extra_validation_generations, validation_codes)]:
        for path in paths:
            for row in read_jsonl(path):
                destination.setdefault(row['task_id'],[]).append(row['candidate_code'])
    completed_path=out/'prepared.jsonl'; excluded_path=out/'excluded.jsonl'
    records=read_jsonl(completed_path) if completed_path.exists() else []
    errors=read_jsonl(excluded_path) if excluded_path.exists() and not a.retry_excluded else []
    done={r['task_id'] for r in records+errors}
    ordered=[r for r in ordered if r['task_id'] not in done]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for start in range(0,len(ordered),a.workers):
            if len(records)>=a.target_tasks:break
            batch=ordered[start:start+a.workers]
            results=pool.map(prepare, [(t,candidates.get(t['task_id'],{}),weak.get(t['task_id'],[]),validation_codes.get(t['task_id'],[])) for t in batch])
            for row,error in results:
                if row is not None:records.append(row)
                else:errors.append(error)
            write_jsonl(completed_path,records);write_jsonl(excluded_path,errors)
            print(json.dumps({'prepared':len(records),'excluded':len(errors)}),flush=True)
    records=records[:a.target_tasks]
    for record in records: record['experiment'] = 'quality'
    write_jsonl(out/'tasks.jsonl',records)
    summary={'target_tasks':a.target_tasks,'eligible_tasks':len(records),
             'by_dataset':{s:sum(r['source_dataset']==s for r in records) for s in groups},
             'independent_validation_mean_kill_rate':{s:sum(r['quality_metrics'][s]['validation_kill_rate'] for r in records)/len(records) if records else None for s in ['high','low','random']}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
