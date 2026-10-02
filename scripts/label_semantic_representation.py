"""Link frozen behavior to prompt states, with execution diagnostics for neither-rule outcomes."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tddexp.execution import execute
MODELS=['qwen25','qwen36','qwen35','qwen38','deepseek']
def main():
    tasks={r['task_id']:r for r in map(json.loads,(ROOT/'data/controlled/semantic/tasks.jsonl').open())}
    observations=[];jobs={}
    for model in MODELS:
        gen={(r['task_id'],r['condition'],r['repeat']):r for r in map(json.loads,(ROOT/f'outputs/controlled/semantic/{model}/generations.jsonl').open())}
        for r in map(json.loads,(ROOT/f'outputs/controlled/semantic/{model}/eval_results.jsonl').open()):
            if r['condition'] not in ['tests_a','tests_b']:continue
            target='a' if r['condition']=='tests_a' else 'b';other='b' if target=='a' else 'a';t=tasks[r['task_id']]
            out=dict(r,model=model,target_rule=target,common_test_count=len(t['heldout_common']))
            assert not (r['rule_a_passed'] and r['rule_b_passed'])
            out['outcome']='target_rule' if r[f'rule_{target}_passed'] else ('alternative_rule' if r[f'rule_{other}_passed'] else 'neither_rule')
            if out['outcome']=='neither_rule':
                code=gen[(r['task_id'],r['condition'],r['repeat'])]['candidate_code']
                key=(r['task_id'],code)
                jobs.setdefault(key,[]).append(out)
            observations.append(out)
    def diagnose(key):
        tid,code=key;t=tasks[tid];na=len(t['heldout_a']);nb=len(t['heldout_b'])
        result=execute(code,tests=t['heldout_a']+t['heldout_b']+t['heldout_common'],timeout=15)
        errors=Counter(e.split(':',1)[0] for e in result.get('errors',[]) if e)
        runtime={k:v for k,v in errors.items() if k!='AssertionError'}
        flags=result['passed'];complete=len(flags)==na+nb+len(t['heldout_common'])
        a=complete and all(flags[:na]) and all(flags[na+nb:]);b=complete and all(flags[na:na+nb]) and all(flags[na+nb:])
        if result['status']!='ok':kind=result['status']
        elif runtime:kind='per_test_runtime_error'
        elif t['heldout_common'] and not all(flags[na+nb:]):kind='common_case_mismatch'
        else:kind='neither_rule_output_mismatch'
        return key,dict(status=result['status'],kind=kind,error_types=dict(errors),rule_a_passed=a,rule_b_passed=b,common_cases_available=bool(t['heldout_common']),replay_label_matches=(not a and not b),error=result.get('error'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(diagnose,k) for k in jobs]):
            key,result=future.result()
            for row in jobs[key]:row['diagnostic_replay']=result
    pairs=[]
    by={(r['model'],r['task_id'],r['condition'],r['repeat']):r for r in observations}
    for model in MODELS:
        for tid,t in sorted(tasks.items()):
            runs=[]
            for repeat in range(3):
                a=by[model,tid,'tests_a',repeat];b=by[model,tid,'tests_b',repeat]
                runs.append(dict(repeat=repeat,paired_switch=int(a['rule_a_passed'] and b['rule_b_passed']),target_adoption=(int(a['rule_a_passed'])+int(b['rule_b_passed']))/2,alternative_adoption=(int(a['rule_b_passed'])+int(b['rule_a_passed']))/2,neither_fraction=(int(a['outcome']=='neither_rule')+int(b['outcome']=='neither_rule'))/2))
            pairs.append(dict(model=model,task_id=tid,family_id=t['family_id'],common_test_count=len(t['heldout_common']),**{key:sum(r[key] for r in runs)/3 for key in ['paired_switch','target_adoption','alternative_adoption','neither_fraction']},runs=runs))
    out=ROOT/'outputs/semantic_representation';out.mkdir(exist_ok=True)
    for name,rows in [('behavior_labels.jsonl',observations),('paired_behavior.jsonl',pairs)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    summary={'observations':len(observations),'pairs':len(pairs),'unique_diagnostic_programs':len(jobs),'models':{}}
    for m in MODELS:
        rs=[r for r in observations if r['model']==m]
        summary['models'][m]=dict(outcomes=dict(Counter(r['outcome'] for r in rs)),diagnostics=dict(Counter(r['diagnostic_replay']['kind'] for r in rs if 'diagnostic_replay' in r)),diagnostic_label_changes=sum(not r['diagnostic_replay']['replay_label_matches'] for r in rs if 'diagnostic_replay' in r))
    (out/'label_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
