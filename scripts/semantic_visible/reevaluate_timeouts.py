import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_visible as av

av.TIMEOUT=15.0
root=av.ROOT
saved=[json.loads(x) for x in (root/'visible_hidden_records.jsonl').read_text(encoding='utf-8-sig').splitlines() if x]
timed={(r['model'],r['task_id'],r['condition'],r['repeat']) for r in saved if r['visible_status']=='timeout'}
if len(timed)!=11: raise ValueError(f'Expected 11 provisional timeouts, got {len(timed)}')
tasks={t['task_id']:t for t in av.read_jsonl(av.PROJECT/'data/controlled/semantic/tasks.jsonl')}
pending=[]
for model,taskid,condition,repeat in timed:
    p=av.RESULTS/model/'generations.jsonl'
    found=None
    for line in p.read_text(encoding='utf-8-sig').splitlines():
        row=json.loads(line)
        if (row['task_id'],row['condition'],row['repeat'])==(taskid,condition,repeat):
            row['family_id']=tasks[taskid]['family_id']
            found=(model,row);break
    if found is None: raise ValueError(f'Generation not found: {model} {taskid} {condition} {repeat}')
    pending.append((model,found[1]))
updated={}
with ThreadPoolExecutor(max_workers=8) as pool:
    futures={pool.submit(av.check,row):(model,row) for model,row in pending}
    for fut in futures:
        model,row=futures[fut]
        res=fut.result()
        res['model']=model
        ev=next(r for r in av.read_jsonl(av.RESULTS/model/'eval_results.jsonl') if (r['task_id'],r['condition'],r['repeat'])==(row['task_id'],row['condition'],row['repeat']))
        for k in ('rule_a_passed','rule_b_passed','common_passed','execution_status','passed'):
            res[k]=ev[k]
        res['target_hidden_passed']=ev['rule_a_passed'] if row['condition']=='tests_a' else ev['rule_b_passed']
        res['other_hidden_passed']=ev['rule_b_passed'] if row['condition']=='tests_a' else ev['rule_a_passed']
        res['pair_target']='A' if row['condition']=='tests_a' else 'B'
        updated[(model,row['task_id'],row['condition'],row['repeat'])]=res
for r in saved:
    k=(r['model'],r['task_id'],r['condition'],r['repeat'])
    if k in updated:
        for field in ('visible_passed','visible_status','visible_error'):
            r[field]=updated[k][field]
with (root/'visible_hidden_records.jsonl').open('w',encoding='utf-8') as f:
    for r in saved: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps({'rechecked_at_original_15s_timeout':len(updated),'pass_count':sum(x['visible_passed'] for x in updated.values()),'still_timeout':sum(x['visible_status']=='timeout' for x in updated.values()),'outcomes':[{k:v[k] for k in ('task_id','condition','repeat','visible_status')} for v in updated.values()]},indent=2))
