#!/usr/bin/env python
"""Report rule adherence by authored family and the paired model difference."""
import json,sys
from itertools import combinations
import yaml
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tddexp.io import read_jsonl
from tddexp.statistics import paired_effect

def summarize_models(model_rows):
    family=defaultdict(lambda:defaultdict(dict)); difference=[]
    for model,rows in model_rows.items():
        pairs=defaultdict(dict)
        for row in rows:
            key=(row['task_id'],row.get('repeat',0))
            if row['condition'] in pairs[key]:
                raise ValueError(f'Duplicate semantic observation: {model}/{key}/{row["condition"]}')
            pairs[key][row['condition']]=row
        counts=defaultdict(lambda:defaultdict(list))
        for (task,repeat),pair in pairs.items():
            required={'nl_only','inputs_only','tests_a','tests_b','explicit_a','explicit_b'}
            if set(pair)!=required:raise ValueError(f'Incomplete semantic conditions: {model}/{task}/{repeat}')
            group=pair['tests_a']['family_id']
            values={'tests_a':pair['tests_a']['rule_a_passed'],'tests_b':pair['tests_b']['rule_b_passed'],
                    'switch':pair['tests_a']['rule_a_passed'] and pair['tests_b']['rule_b_passed'],
                    'explicit_both':pair['explicit_a']['rule_a_passed'] and pair['explicit_b']['rule_b_passed']}
            for metric,value in values.items():counts[group][(task,metric)].append(int(value))
            difference.append(dict(task_id=task,repeat=repeat,family_id=group,condition=model,passed=values['switch']))
        for group,values in counts.items():
            for (task,metric),runs in values.items():
                family[group][model][metric]=family[group][model].get(metric,0)+sum(runs)/len(runs)
    comparisons=[paired_effect(difference,a,b,cluster_field='family_id') for a,b in combinations(model_rows,2)]
    return {'family_metric':'successful instances averaged within task over repeats',
            'families':dict(family),'model_comparisons':comparisons,
            'paired_model_switch_difference':next(r for r in comparisons if r['base']=='qwen25' and r['target']=='qwen36')}


def main():
    config=yaml.safe_load((ROOT/'configs/controlled.yaml').read_text())
    models=config['experiments']['semantic']['models']
    names={'qwen25':'Qwen2.5-7B','qwen36':'Qwen3.6-27B','deepseek':'DeepSeek-6.7B','qwen35':'Qwen3.5-9B','qwen38':'Qwen3.8-27B'}
    rows={model:read_jsonl(ROOT/f'outputs/controlled/semantic/{model}/eval_results.jsonl') for model in models}
    result=summarize_models(rows)
    family=result['families']
    out=ROOT/'outputs/analysis/semantic_families.json';out.write_text(json.dumps(result,indent=2)+'\n')
    table=[r'\begin{table*}[t]',r'\centering',r'\small',r'\begin{tabular}{l'+('rr'*len(models))+'}',r'\toprule',
           ' & '+' & '.join(r'\multicolumn{2}{c}{'+names[m]+'}' for m in models)+r' \\',
           'Family & '+' & '.join(['Switch & Explicit']*len(models))+r' \\',r'\midrule']
    for name,values in family.items():
        table.append(' & '.join([name.replace('_',' ')]+[f"{values[m][k]:.1f}" for m in models for k in ['switch','explicit_both']])+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\caption{Mean successful instances per run within each specification family, out of six. Each task is averaged over three runs. Switch and Explicit require successful A and B programs for the same instance and run under tests and explicit rules, respectively. Rule-specific family scores accompany the results archive.}',r'\label{tab:semantic-families}',r'\end{table*}']
    (ROOT/'aaai/AuthorKit27/AuthorKit27/Tables/semantic_families.tex').write_text('\n'.join(table)+'\n')
    tasks = {r['family_id']:r for r in read_jsonl(ROOT/'data/controlled/semantic/tasks.jsonl')}
    escapes = {'_':r'\_', '&':r'\&', '%':r'\%', '#':r'\#'}
    tex = lambda value: ''.join(escapes.get(c,c) for c in value)
    rules = [r'\begin{table*}[t]',r'\centering',r'\small',
             r'\begin{tabular}{p{.16\textwidth}p{.38\textwidth}p{.38\textwidth}}',
             r'\toprule',r'Family & Rule A & Rule B \\',r'\midrule']
    for name,task in tasks.items():
        rules.append(' & '.join([tex(name.replace('_',' ')),tex(task['rule_a']),tex(task['rule_b'])])+r' \\')
    rules += [r'\bottomrule',r'\end{tabular}',
              r'\caption{Alternative rules used in the semantic intervention. Each pair shares an underspecified description, function signature, and three visible inputs. The explicit conditions add the corresponding rule to the description.}',
              r'\label{tab:semantic-rules}',r'\end{table*}']
    (ROOT/'aaai/AuthorKit27/AuthorKit27/Tables/semantic_rules.tex').write_text('\n'.join(rules)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
