from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'outputs/semantic_visible'
RESULTS = PROJECT/'outputs/controlled/semantic'
MODELS = ('qwen25', 'qwen35', 'qwen36', 'qwen38', 'deepseek')
TIMEOUT = 15.0


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def check(row):
    program = row['candidate_code'].rstrip() + '\n\n' + '\n'.join(row['prompt_tests']) + '\n'
    try:
        p = subprocess.run(
            [sys.executable, '-I', '-c', program],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT,
            check=False,
        )
        return {**{k: row[k] for k in ('task_id', 'family_id', 'condition', 'repeat')},
                'visible_passed': p.returncode == 0,
                'visible_status': 'pass' if p.returncode == 0 else 'fail',
                'visible_error': p.stderr.decode('utf-8', errors='replace')[-1200:] if p.returncode else ''}
    except subprocess.TimeoutExpired as exc:
        return {**{k: row[k] for k in ('task_id', 'family_id', 'condition', 'repeat')},
                'visible_passed': False, 'visible_status': 'timeout',
                'visible_error': str(exc)}


def main():
    tasks = {r['task_id']: r for r in read_jsonl(PROJECT/'data/controlled/semantic/tasks.jsonl')}
    evaluations = {}
    generations = []
    for model in MODELS:
        eval_rows = read_jsonl(RESULTS/model/'eval_results.jsonl')
        model_evaluations = {}
        for row in eval_rows:
            key = (row['task_id'], row['condition'], row['repeat'])
            if key in model_evaluations:
                raise ValueError(f'duplicate eval key: {model} {key}')
            model_evaluations[key] = row
        evaluations[model] = model_evaluations
        gen_rows = read_jsonl(RESULTS/model/'generations.jsonl')
        if len(gen_rows) != 2160:
            raise ValueError(f'{model}: expected 2160 generations, got {len(gen_rows)}')
        for row in gen_rows:
            task = tasks[row['task_id']]
            row['family_id'] = task['family_id']
            if row['prompt_tests'] != task['condition_tests'][row['condition']]:
                raise ValueError(f"shown tests mismatch task protocol: {model} {row['task_id']} {row['condition']}")
            if row['prompt'] != task['condition_prompts'][row['condition']]:
                raise ValueError(f"prompt mismatch task protocol: {model} {row['task_id']} {row['condition']}")
            if row['condition'] in ('tests_a', 'tests_b'):
                generations.append((model, row))

    print(f'Validated exact prompt/test definitions for {5*2160} generations; evaluating {len(generations)} A/B test-conditioned programs.', flush=True)
    outcomes = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(check, row): (model, row) for model, row in generations}
        for i, fut in enumerate(as_completed(futures), 1):
            model, source = futures[fut]
            result = fut.result()
            result['model'] = model
            ev = evaluations[model][(source['task_id'], source['condition'], source['repeat'])]
            for k in ('rule_a_passed', 'rule_b_passed', 'common_passed', 'execution_status', 'passed'):
                result[k] = ev[k]
            result['target_hidden_passed'] = ev['rule_a_passed'] if source['condition'] == 'tests_a' else ev['rule_b_passed']
            result['other_hidden_passed'] = ev['rule_b_passed'] if source['condition'] == 'tests_a' else ev['rule_a_passed']
            result['pair_target'] = 'A' if source['condition'] == 'tests_a' else 'B'
            outcomes.append(result)
            if i % 300 == 0:
                print(f'checked {i}/{len(generations)}', flush=True)
    outcomes.sort(key=lambda r: (r['model'], r['task_id'], r['condition'], r['repeat']))
    with (ROOT/'visible_hidden_records.jsonl').open('w', encoding='utf-8') as f:
        for row in outcomes:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    # One observation is a task-run. Run repetitions stay nested in each instance.
    by_model = {}
    for model in MODELS:
        rows = [r for r in outcomes if r['model'] == model]
        by_direction = {}
        for cond in ('tests_a', 'tests_b'):
            rr = [r for r in rows if r['condition'] == cond]
            n = len(rr)
            vis = sum(r['visible_passed'] for r in rr)
            hid = sum(r['target_hidden_passed'] for r in rr)
            both = sum(r['visible_passed'] and r['target_hidden_passed'] for r in rr)
            vonly = sum(r['visible_passed'] and not r['target_hidden_passed'] for r in rr)
            honly = sum(not r['visible_passed'] and r['target_hidden_passed'] for r in rr)
            neither = n-both-vonly-honly
            by_direction[cond] = {
                'n_task_runs': n,
                'visible_pass': vis, 'visible_pass_pct': 100*vis/n,
                'hidden_target_pass': hid, 'hidden_target_pass_pct': 100*hid/n,
                'visible_and_hidden_pass': both, 'visible_and_hidden_pct': 100*both/n,
                'visible_only': vonly, 'hidden_only': honly, 'neither': neither,
                'other_rule_hidden_pass': sum(r['other_hidden_passed'] for r in rr),
                'visible_pass_then_other_rule': sum(r['visible_passed'] and r['other_hidden_passed'] for r in rr),
                'execution_error': sum(r['visible_status']=='fail' or r['execution_status']!='ok' for r in rr),
                'timeout': sum(r['visible_status']=='timeout' or 'timeout' in r['execution_status'] for r in rr),
            }
        # Family-level task-run averages for paired transitions.
        families = sorted({r['family_id'] for r in rows})
        famsum = {}
        for fam in families:
            fr = [r for r in rows if r['family_id']==fam]
            famsum[fam] = {}
            for cond in ('tests_a','tests_b'):
                cr=[r for r in fr if r['condition']==cond]
                famsum[fam][cond] = {
                    'n':len(cr),
                    'visible_pass_rate':sum(r['visible_passed'] for r in cr)/len(cr),
                    'hidden_target_rate':sum(r['target_hidden_passed'] for r in cr)/len(cr),
                    'both_rate':sum(r['visible_passed'] and r['target_hidden_passed'] for r in cr)/len(cr),
                    'visible_only_rate':sum(r['visible_passed'] and not r['target_hidden_passed'] for r in cr)/len(cr),
                    'hidden_only_rate':sum(not r['visible_passed'] and r['target_hidden_passed'] for r in cr)/len(cr),
                }
        by_model[model] = {'directions':by_direction, 'families':famsum}

    # Behavioral transition matrices relate the NL-only default to A/B test outcomes.
    # Each cell counts paired task-runs, not independent instances.
    paired = {}
    for model in MODELS:
        eval_rows = read_jsonl(RESULTS/model/'eval_results.jsonl')
        index={(r['task_id'],r['condition'],r['repeat']):r for r in eval_rows}
        matrices={}
        for cond in ('tests_a','tests_b'):
            matrix={d:{a:0 for a in ('A','B','both','neither')} for d in ('A','B','both','neither')}
            for task in tasks.values():
                for repeat in range(3):
                    nl=index[(task['task_id'],'nl_only',repeat)]
                    test=index[(task['task_id'],cond,repeat)]
                    def cls(r):
                        a,b=bool(r['rule_a_passed']),bool(r['rule_b_passed'])
                        return 'both' if a and b else 'A' if a else 'B' if b else 'neither'
                    matrix[cls(nl)][cls(test)]+=1
            matrices[cond]=matrix
        paired[model]=matrices
    (ROOT/'visible_hidden_analysis.json').write_text(json.dumps({
        'description':'Visible assertions executed from each saved generation prompt using the task project Python interpreter, isolated subprocesses, matching the original semantic evaluator 15s timeout; no completions regenerated.',
        'source_checks':{'models':list(MODELS),'tasks':len(tasks),'generations_per_model':2160,'exact_prompt_and_visible_test_match':True,'evaluated_A_B_records':len(outcomes),'task_repeats_nested':True,'families':len({x['family_id'] for x in tasks.values()})},
        'by_model':by_model,'nl_default_to_test_behavior_transitions':paired,
        'metric_note':'Counts are task-run records; family bootstrap should be used for inferential intervals. Visible pass requires all three displayed assertions.'
    },indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'evaluated':len(outcomes),'counts':{m:{c:{k:v for k,v in d.items() if k in ('visible_pass','visible_pass_pct','hidden_target_pass','hidden_target_pass_pct','visible_and_hidden_pass','visible_and_hidden_pct','visible_only','hidden_only','neither','execution_error','timeout')} for c,d in by_model[m]['directions'].items()} for m in MODELS}},indent=2),flush=True)


if __name__ == '__main__':
    main()
