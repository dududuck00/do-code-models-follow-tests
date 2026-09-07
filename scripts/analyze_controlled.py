#!/usr/bin/env python
"""Summarize held-out performance and paired rule switching."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tddexp.io import read_jsonl
from tddexp.statistics import paired_effect


def analyze(rows):
    conditions=sorted({r['condition'] for r in rows})
    summary={}
    for condition in conditions:
        group=[r for r in rows if r['condition']==condition]
        fields=[f for f in ['passed','official_passed','heldout_passed','rule_a_passed','rule_b_passed'] if f in group[0]]
        summary[condition]={'observations':len(group),'tasks':len({r['task_id'] for r in group}),
                            **{f:sum(r[f] for r in group)/len(group) for f in fields}}
    output={'conditions':summary,'contrasts':[]}
    if 'tests_a' in conditions:
        pairs=defaultdict(dict)
        for row in rows:pairs[(row['task_id'],row.get('repeat',0))][row['condition']]=row
        switches=[]
        for (task,repeat),pair in pairs.items():
            for c in ['tests_a','tests_b']:
                if c not in pair:raise ValueError('Incomplete semantic pair.')
            success=pair['tests_a']['rule_a_passed'] and pair['tests_b']['rule_b_passed']
            for condition,passed in [('zero',False),('switch',success)]:
                switches.append(dict(task_id=task,repeat=repeat,condition=condition,passed=passed,family_id=pair['tests_a']['family_id']))
        output['paired_rule_switch']=paired_effect(switches,'zero','switch',cluster_field='family_id')
        for side in ['a','b']:
            for base in ['nl_only','inputs_only','explicit_'+side]:
                output['contrasts'].append(paired_effect(rows,base,'tests_'+side,field='rule_'+side+'_passed',cluster_field='family_id'))
    else:
        contrasts=([('nl_only','nl_tests'),('inputs_only','nl_tests'),('wrong_tests','nl_tests')]
                   if 'nl_tests' in conditions else [('quality_random','quality_high'),('quality_low','quality_high'),('nl_only','quality_high')])
        output['contrasts']=[paired_effect(rows,b,t) for b,t in contrasts]
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--eval-results',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    groups=defaultdict(list)
    for path in a.eval_results:
        for row in read_jsonl(path):groups[(row.get('model_path','unknown'),row.get('experiment','main'),row['source_dataset'])].append(row)
    report=[dict(model_path=model,experiment=experiment,dataset=dataset,**analyze(rows)) for (model,experiment,dataset),rows in groups.items()]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    for r in report:
        print(json.dumps({'model':Path(r['model_path']).name,'dataset':r['dataset'],'conditions':r['conditions'],
                           'paired_rule_switch':r.get('paired_rule_switch')}))


if __name__=='__main__':main()
