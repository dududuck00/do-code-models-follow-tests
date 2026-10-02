"""Freeze historical high3 prompts for the Qwen3.6 completion run."""
from pathlib import Path
import json
from collections import defaultdict, Counter
root=Path(__file__).resolve().parents[1]
source=root/'outputs/livecodebench_v6_new_qwen_synthetic_high3/generations.jsonl'
groups=defaultdict(dict)
for line in source.open():
    r=json.loads(line);assert r['condition'] not in groups[r['task_id']]
    groups[r['task_id']][r['condition']]=r
conditions=['nl_only','nl_tests','shuffled_tests','irrelevant_tests']
target=root/'data/controlled/historical_lcb_high3/tasks.jsonl'
target.parent.mkdir(parents=True,exist_ok=True)
with target.open('w') as f:
    for tid,rows in sorted(groups.items()):
        assert set(rows)==set(conditions)
        r=rows['nl_tests']
        out=dict(protocol_version='semantic-tests-1',experiment='historical_lcb_high3',source_dataset='livecodebench',task_id=tid,prompt=rows['nl_only']['prompt'],entry_point=r['entry_point'],reference_code='',condition_prompts={c:rows[c]['prompt'] for c in conditions},condition_tests={c:rows[c]['prompt_tests'] for c in conditions})
        f.write(json.dumps(out,ensure_ascii=False)+'\n')
assert len(groups)==175
print(json.dumps({'tasks':len(groups),'prompts':len(groups)*4,'source':str(source.relative_to(root)),'suite_sizes':dict(Counter(len(r['nl_tests']['prompt_tests']) for r in groups.values()))}))
