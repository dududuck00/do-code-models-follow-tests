#!/usr/bin/env python
"""Render manuscript tables from complete task-level evaluation records."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tddexp.io import read_jsonl
from analyze_controlled import analyze

MODELS={'qwen25':'Qwen2.5-7B','deepseek':'DeepSeek-6.7B','qwen35':'Qwen3.5-9B','qwen38':'Qwen3.8-27B','qwen36':'Qwen3.6-27B'}
DATASETS={'humanevalplus':'HumanEval+','mbppplus':'MBPP+','livecodebench':'LCB v6-new'}
CONDITIONS={'main':['nl_only','nl_tests','wrong_tests','inputs_only'],
            'quality':['nl_only','quality_random','quality_low','quality_high'],
            'semantic':['nl_only','inputs_only','tests_a','tests_b','explicit_a','explicit_b']}

def percent(x):return f'{100*x:.1f}'
def signed_percent(x):
    value=round(100*x,1)
    return f'{value if value else 0.0:+.1f}'
def interval(r):return f"{signed_percent(r['paired_difference'])} [{signed_percent(r['ci95'][0])}, {signed_percent(r['ci95'][1])}]"
def table(columns, headers, rows, caption, label):
    return '\n'.join([r'\begin{table*}[t]',r'\centering',r'\small',r'\begin{tabular}{'+columns+'}',r'\toprule',
                      ' & '.join(headers)+r' \\',r'\midrule',*[' & '.join(map(str,row))+r' \\' for row in rows],
                      r'\bottomrule',r'\end{tabular}',r'\caption{'+caption+'}',r'\label{tab:'+label+'}',r'\end{table*}',''])

def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    out=ROOT/'aaai/AuthorKit27/AuthorKit27/Tables';out.mkdir(exist_ok=True)
    config=yaml.safe_load((ROOT/'configs/controlled.yaml').read_text())
    records={}; missing=[]
    for exp,spec in config['experiments'].items():
        models=spec['models']
        tasks=read_jsonl(ROOT/f'data/controlled/{exp}/tasks.jsonl')
        repeats=range(spec['repeats'])
        expected={(t['task_id'],c,r) for t in tasks for c in CONDITIONS[exp] for r in repeats}
        for model in models:
            path=ROOT/f'outputs/controlled/{exp}/{model}/eval_results.jsonl'
            if not path.exists():missing.append(str(path.relative_to(ROOT)));continue
            rows=read_jsonl(path)
            actual=[(r['task_id'],r['condition'],r['repeat']) for r in rows]
            if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError(f'Incomplete or duplicated evaluation: {path}')
            records[exp,model]=rows
    if missing and not a.partial:raise SystemExit('Missing evaluations: '+', '.join(missing))
    summary={}; main_rows=[]; official_rows=[]; effects=[]; repeats=[]; quality=[]; semantic=[]; semantic_repeats=[]
    for (exp,model),rows in records.items():
        summary[f'{exp}/{model}']=analyze(rows)
        if exp=='main':
            for dataset,label in DATASETS.items():
                group=[r for r in rows if r['source_dataset']==dataset];result=analyze(group)
                summary[f'{exp}/{model}/{dataset}']=result
                n=len({r['task_id'] for r in group})
                main_rows.append([MODELS[model],label,n]+[percent(result['conditions'][c]['passed']) for c in CONDITIONS[exp]])
                official_rows.append([MODELS[model],label,n]+[percent(result['conditions'][c]['official_passed']) for c in CONDITIONS[exp]])
                effects.append([MODELS[model],label]+[interval(r) for r in result['contrasts']])
                for effect in result['contrasts']:
                    base={'nl_only':'NL','inputs_only':'Inputs','wrong_tests':'Wrong'}[effect['base']]
                    repeats.append([MODELS[model],label,'Correct $-$ '+base]+[
                        f"{100*effect['per_repeat'][str(i)]['paired_difference']:+.1f}" for i in range(3)])
        elif exp=='quality':
            result=summary[f'{exp}/{model}'];contrasts={r['base']:r for r in result['contrasts']}
            quality.append([MODELS[model]]+[percent(result['conditions'][c]['passed']) for c in CONDITIONS[exp]]+
                           [interval(contrasts[c]) for c in ['quality_low','quality_random']])
        else:
            result=summary[f'{exp}/{model}'];c=result['conditions'];switch=result['paired_rule_switch']
            semantic.append([MODELS[model],percent(c['nl_only']['rule_a_passed']),percent(c['nl_only']['rule_b_passed']),
                             percent(c['tests_a']['rule_a_passed']),percent(c['tests_b']['rule_b_passed']),
                             percent(c['explicit_a']['rule_a_passed']),percent(c['explicit_b']['rule_b_passed']),
                             f"{percent(switch['paired_difference'])} [{percent(switch['ci95'][0])}, {percent(switch['ci95'][1])}]"])
            for repeat in sorted({r['repeat'] for r in rows}):
                run=analyze([r for r in rows if r['repeat']==repeat])
                semantic_repeats.append([MODELS[model],repeat+1]+[
                    percent(run['conditions'][condition]['rule_'+side+'_passed'])
                    for condition,side in [('tests_a','a'),('tests_b','b'),('explicit_a','a'),('explicit_b','b')]]+
                    [percent(run['paired_rule_switch']['paired_difference'])])
    qs=json.loads((ROOT/'data/controlled/quality/summary.json').read_text())
    values={'QualityTasks':qs['eligible_tasks'],'QualityHE':qs['by_dataset']['humanevalplus'],'QualityMB':qs['by_dataset']['mbppplus'],
            **{f'Quality{s.title()}Kill':percent(v) for s,v in qs['independent_validation_mean_kill_rate'].items()}}
    (out/'values.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in values.items())+'\n')
    for name,columns,headers,rows,caption,label in [
      ('main','llrrrrr',['Model','Dataset','$n$','NL-only','Correct I/O','Wrong I/O','Inputs-only'],main_rows,
       'Held-out pass rates (\\%), averaged over three runs. Exposed inputs are removed from the evaluation suite. Each task contributes one mean correctness value.','main'),
      ('main_effects','lllll',['Model','Dataset','Correct $-$ NL','Correct $-$ Inputs','Correct $-$ Wrong'],effects,
       'Paired differences in percentage points with task-bootstrap 95\\% intervals. Repeated runs are averaged within tasks before resampling.','main-effects'),
      ('official','llrrrrr',['Model','Dataset','$n$','NL-only','Correct I/O','Wrong I/O','Inputs-only'],official_rows,
       'Complete official-suite pass rates (\\%), averaged over three runs. These include public inputs and are secondary to held-out performance.','official'),
      ('repeats','lllrrr',['Model','Dataset','Contrast','Run 1','Run 2','Run 3'],repeats,
       'Per-run paired held-out differences in percentage points. Task-level gains, losses, and exact McNemar tests are retained in the accompanying machine-readable summaries.','repeats'),
      ('quality','lrrrrll',['Model','NL-only','Random','Low','High','High $-$ Low','High $-$ Random'],quality,
       'Held-out pass rates (\\%) on 180 tasks with three tests per suite. Differences are percentage points with paired task-bootstrap 95\\% intervals. The entire candidate and development input pools are excluded from final evaluation.','quality'),
      ('semantic_repeats','lrrrrrr',['Model','Run','Tests A','Tests B','Explicit A','Explicit B','Paired switch'],semantic_repeats,
       'Per-run semantic adherence and paired switching (\\%). Each run contains the same 120 instances from 20 families.','semantic-repeats'),
      ('semantic','lrrrrrrl',['Model','NL: A','NL: B','Tests A','Tests B','Explicit A','Explicit B','Paired switch [95\\% CI]'],semantic,
       'Rule adherence and paired switching (\\%) averaged over three runs of 120 paired instances from 20 specification families. Tests A/B and Explicit A/B are scored against their assigned rule. NL-only columns report the default rule distribution. Switching requires both test-conditioned programs to follow their respective rules. Paired outcomes are averaged within instances before resampling the 20 families.','semantic')]:
        (out/(name+'.tex')).write_text(table(columns,headers,rows,caption,label))
    dest=ROOT/'outputs/analysis/controlled_results.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps({'complete':not missing,'missing_evaluations':missing,'quality_dataset':qs,'results':summary},indent=2)+'\n')
    print(json.dumps({'complete':not missing,'missing_evaluations':missing,'tables':str(out)}))

if __name__=='__main__':main()
