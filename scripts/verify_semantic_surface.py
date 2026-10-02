"""Check completed surface records against the validated prompts and exported metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
    base=ROOT/'outputs/semantic_surface';analysis=json.loads((base/'analysis.json').read_text());assert analysis['complete']
    data=pd.read_csv(base/'pair_layer_metrics.csv');report={'complete':True,'models':{},'new_code_generations':0}
    for m,result in analysis['models'].items():
        prompts={(r['task_id'],r['condition']):r for r in rows(base/m/'prompts.jsonl')};states=rows(base/m/'replay/generations.jsonl')
        assert len(prompts)==len(states)==480 and len({(r['task_id'],r['condition']) for r in states})==480
        for r in states:
            assert r['prompt']==prompts[r['task_id'],r['condition']]['prompt']
            assert r['reference_equivalence_verified'] and (ROOT/r['state_path']).is_file()
        d=data[data.model==m];assert d.task_id.nunique()==120 and d.family_id.nunique()==20
        assert np.allclose(d.different_rule-d.same_rule,d.difference)
        for v,stats in result['variants'].items():
            for metric,values in stats.items():
                z=values['final_layer'];final=d[(d.variant==v)&(d.metric==metric)&(d.layer==z['layer'])]
                assert len(final)==120
                assert np.isclose(final.difference.mean(),z['difference'])
                assert np.isclose(final.same_rule.mean(),z['same_rule_mean'])
                assert np.isclose(final.different_rule.mean(),z['different_rule_mean'])
        report['models'][m]={'new_states':480,'reused_states':240,'tasks':120,'families':20,'variant_pairs_per_task':2,'final_metrics_recomputed':True}
    (base/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
