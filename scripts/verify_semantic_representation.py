"""Verify state coverage, original outcome links, family folds, and held-out metrics."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/semantic_representation'
MODELS={'qwen25':(29,3584),'qwen36':(65,5120),'qwen35':(33,4096),'qwen38':(65,5120),'deepseek':(33,4096)}
def rows(path):
    return [json.loads(line) for line in path.open()]
def key(r):
    return r['task_id'],r['condition'],r['repeat']
def main():
    labels=rows(BASE/'behavior_labels.jsonl');pairs=rows(BASE/'paired_behavior.jsonl')
    predictions=rows(BASE/'family_probe_predictions.jsonl')
    folds=json.loads((BASE/'family_folds.json').read_text())['outer_fold_by_family']
    probe=json.loads((BASE/'family_probe.json').read_text())
    diagnostics=json.loads((BASE/'label_summary.json').read_text())
    assert len(labels)==720*len(MODELS) and len(pairs)==120*len(MODELS) and len(predictions)==120*len(MODELS)
    assert len(folds)==20 and all(list(folds.values()).count(i)==4 for i in range(5))
    result={'complete':True,'unique_prompt_states':0,'behavior_records':len(labels),'task_model_pairs':len(pairs),'models':{}}
    for model,shape in MODELS.items():
        original=ROOT/f'outputs/controlled/semantic/{model}'
        gens={key(r):r for r in rows(original/'generations.jsonl') if r['condition'] in ('tests_a','tests_b')}
        evals={key(r):r for r in rows(original/'eval_results.jsonl') if r['condition'] in ('tests_a','tests_b')}
        states=rows(BASE/model/'replay/generations.jsonl')
        assert len(gens)==len(evals)==720 and len(states)==240
        assert len({(r['task_id'],r['condition']) for r in states})==240
        for r in states:
            assert r['prompt_tokens_verified'] and r['state_prompt_token_hash']==r['prompt_token_hash']
            assert not Path(r['state_path']).is_absolute()
            with np.load(ROOT/r['state_path']) as state:
                h=state['prompt_end_hidden'];assert h.shape==shape and np.isfinite(h).all()
            for repeat in range(3):
                g=gens[r['task_id'],r['condition'],repeat]
                assert g['prompt_token_hash']==r['prompt_token_hash'] and g['prompt']==r['prompt']
        ml={key(r):r for r in labels if r['model']==model};assert len(ml)==720
        for k,r in ml.items():
            e=evals[k]
            for field in ('rule_a_passed','rule_b_passed','common_passed','passed'):
                assert r[field]==e[field]
            target='rule_a_passed' if r['condition']=='tests_a' else 'rule_b_passed'
            alt='rule_b_passed' if r['condition']=='tests_a' else 'rule_a_passed'
            outcome='target_rule' if e[target] else 'alternative_rule' if e[alt] else 'neither_rule'
            assert r['outcome']==outcome
        mp={r['task_id']:r for r in pairs if r['model']==model};assert len(mp)==120
        for tid,r in mp.items():
            run_values=[]
            for repeat in range(3):
                a,b=[ml[tid,c,repeat]['outcome'] for c in ('tests_a','tests_b')]
                expected=int(a==b=='target_rule');run_values.append(expected)
                assert r['runs'][repeat]['paired_switch']==expected
            assert np.isclose(r['paired_switch'],np.mean(run_values))
        pred=[r for r in predictions if r['model']==model]
        assert len(pred)==120 and {r['task_id'] for r in pred}==set(mp)
        for r in pred:
            assert r['fold']==folds[r['family_id']]
            assert r['paired_switch']==mp[r['task_id']]['paired_switch']
            train=[v['paired_switch'] for v in mp.values() if folds[v['family_id']]!=r['fold']]
            assert len(train)==96
            assert np.isclose(r['median'],np.median(train)) and np.isclose(r['mean'],np.mean(train))
        y=np.array([r['paired_switch'] for r in pred])
        for name in ('mean','median','output_text','representation'):
            x=np.array([r[name] for r in pred]);assert np.isfinite(x).all() and ((x>=0)&(x<=1)).all()
            metrics=probe['models'][model]['metrics'][name]
            assert np.isclose(metrics['mae'],np.mean(abs(x-y)))
            assert np.isclose(metrics['mse'],np.mean((x-y)**2))
        assert diagnostics['models'][model]['diagnostic_label_changes']==0
        result['models'][model]={'states':240,'state_shape':list(shape),'original_behavior_links':720,'held_out_predictions':120,'family_disjoint_folds':5,'metrics_recomputed':True}
        result['unique_prompt_states']+=len(states)
    (BASE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
