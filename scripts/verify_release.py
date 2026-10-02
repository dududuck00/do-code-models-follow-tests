#!/usr/bin/env python3
"""CPU-only checks and point-estimate reconstruction from saved release records."""
from pathlib import Path
from collections import defaultdict
import gzip,json,math,statistics

ROOT=Path(__file__).resolve().parents[1]
MODELS=('qwen25','qwen35','qwen36','qwen38','deepseek')
EXPECTED={'main':8520,'unrelated':2130,'semantic':2160,'quality':720}
def rows(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]
def same(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)
def key(r):return r['task_id'],r['condition'],r['repeat']
def main():
    expected=json.loads((ROOT/'paper/Results/controlled_results.json').read_text())['results']
    records={};summary={};total=0;comparisons=0
    for experiment,n in EXPECTED.items():
        tasks=rows(ROOT/f'data/controlled/{experiment}/tasks.jsonl')
        taskids={t['task_id'] for t in tasks}
        for model in MODELS:
            base=ROOT/f'outputs/controlled/{experiment}/{model}'
            es=rows(base/'eval_results.jsonl'); gs=rows(base/'generations.jsonl.gz')
            assert len(es)==len(gs)==n
            assert len({key(r) for r in es})==len({key(r) for r in gs})==n
            assert {key(r) for r in es}=={key(r) for r in gs}
            assert {r['task_id'] for r in es}==taskids
            records[experiment,model]=es;total+=n
            groups=defaultdict(list)
            for r in es:groups[r['condition']].append(r)
            points={}
            for condition,rr in groups.items():
                points[condition]={field:statistics.mean(float(r[field]) for r in rr)
                    for field in ['passed','official_passed','heldout_passed','rule_a_passed','rule_b_passed'] if field in rr[0]}
                if experiment!='unrelated':
                    for field,value in points[condition].items():
                        same(value,expected[f'{experiment}/{model}']['conditions'][condition][field]);comparisons+=1
            if experiment=='main':
                for dataset in ('humanevalplus','mbppplus','livecodebench'):
                    for condition in groups:
                        rr=[r for r in groups[condition] if r['source_dataset']==dataset]
                        for field in points[condition]:
                            same(statistics.mean(float(r[field]) for r in rr),expected[f'main/{model}/{dataset}']['conditions'][condition][field]);comparisons+=1
            if experiment=='semantic':
                ix={key(r):r for r in es}
                switch=statistics.mean(float(ix[t,'tests_a',run]['rule_a_passed'] and ix[t,'tests_b',run]['rule_b_passed']) for t in taskids for run in range(3))
                same(switch,expected[f'semantic/{model}']['paired_rule_switch']['paired_difference'])
                points['paired_switch']=switch;comparisons+=1
            summary[f'{experiment}/{model}']={'records':n,'point_estimates':points}
    assert total==67650
    other=json.loads((ROOT/'outputs/analysis/unrelated_results.json').read_text())['results']
    for model in MODELS:
        combined=records['main',model]+records['unrelated',model]
        for dataset in ('humanevalplus','mbppplus','livecodebench'):
            for condition,value in other[f'{model}/{dataset}']['rates'].items():
                rr=[r for r in combined if r['source_dataset']==dataset and r['condition']==condition]
                same(statistics.mean(float(r['passed']) for r in rr),value);comparisons+=1
    base=ROOT/'outputs/semantic_representation'
    probe=json.loads((base/'family_probe.json').read_text())
    folds=json.loads((base/'family_folds.json').read_text())['outer_fold_by_family']
    predictions=rows(base/'family_probe_predictions.jsonl')
    assert len(predictions)==600 and len(folds)==20
    assert all(list(folds.values()).count(i)==4 for i in range(5))
    for model in MODELS:
        rr=[r for r in predictions if r['model']==model]
        assert len(rr)==120
        for r in rr:
            assert r['fold']==folds[r['family_id']]
            train=[q['paired_switch'] for q in rr if q['fold']!=r['fold']]
            same(r['mean'],statistics.mean(train));same(r['median'],statistics.median(train))
        for kind in ('mean','median','output_text','representation'):
            same(statistics.mean(abs(r[kind]-r['paired_switch']) for r in rr),probe['models'][model]['metrics'][kind]['mae'])
            same(statistics.mean((r[kind]-r['paired_switch'])**2 for r in rr),probe['models'][model]['metrics'][kind]['mse'])
            comparisons+=2
    visible=rows(ROOT/'outputs/semantic_visible/visible_hidden_records.jsonl')
    assert len(visible)==3600
    hidden_fail=[r for r in visible if not r['target_hidden_passed']]
    exposed_fail=sum(not r['visible_passed'] for r in hidden_fail)
    result={'complete':True,'generations':total,'evaluations':total,'numerical_summary_checks':comparisons,
      'held_out_probe_predictions':600,'visible_replay_records':len(visible),
      'hidden_failures':len(hidden_fail),'hidden_failures_also_failing_visible':exposed_fail,
      'percent_hidden_failures_exposed':100*exposed_fail/len(hidden_fail),
      'models':summary,'scope':'Recomputed from saved outcomes and predictions; no generated code executed and no activation tensors loaded.'}
    out=ROOT/'.tmp/release_verification.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2))
if __name__=='__main__':main()
