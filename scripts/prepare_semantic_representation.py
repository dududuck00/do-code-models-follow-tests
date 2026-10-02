"""Freeze unique semantic prompts and their three-run behavior linkage."""
import json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def main():
    cfg=yaml.safe_load((ROOT/'configs/semantic_representation.yaml').read_text())
    model_paths=yaml.safe_load((ROOT/'configs/controlled.yaml').read_text())['models']
    tasks={r['task_id']:r for r in map(json.loads,(ROOT/cfg['dataset']).open())}
    inventory={}
    for model in cfg['models']:
        source=ROOT/f'outputs/controlled/semantic/{model}/generations.jsonl'
        groups={}
        for row in map(json.loads,source.open()):
            if row['condition'] not in cfg['conditions']:continue
            groups.setdefault((row['task_id'],row['condition']),[]).append(row)
        frozen=[]
        for key,rows in sorted(groups.items()):
            assert sorted(r['repeat'] for r in rows)==[0,1,2]
            assert len({r['prompt_token_hash'] for r in rows})==1
            assert len({r['prompt'] for r in rows})==1
            r=dict(min(rows,key=lambda r:r['repeat']));t=tasks[key[0]]
            assert r['prompt']==t['condition_prompts'][key[1]]
            r.update(family_id=t['family_id'],state_unit='unique_prompt',behavior_repeats=[0,1,2],generation_source=str(source.relative_to(ROOT)),generation_model_path=r['model_path'],model_path=model_paths[model])
            frozen.append(r)
        assert len(frozen)==240
        out=ROOT/f'outputs/semantic_representation/{model}';out.mkdir(parents=True,exist_ok=True)
        (out/'unique_prompts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in frozen))
        inventory[model]={'unique_prompts':240,'behavior_observations':720,'model_path':model_paths[model],'generation_model_paths':sorted({r['generation_model_path'] for r in frozen})}
    (ROOT/'outputs/semantic_representation/prompt_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(inventory,indent=2))
if __name__=='__main__':main()
