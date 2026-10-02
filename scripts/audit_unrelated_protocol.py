#!/usr/bin/env python
"""Validate donor assertions and summarize their behavior on recipient references."""
import json,sys
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tddexp.io import read_jsonl
from tddexp.execution import execute

def main():
 original={r['task_id']:r for r in read_jsonl(ROOT/'data/controlled/main/tasks.jsonl')}
 tasks=read_jsonl(ROOT/'data/controlled/unrelated/tasks.jsonl');mapping=read_jsonl(ROOT/'data/controlled/unrelated/donor_mapping.jsonl')
 extras={r['task_id']:r for r in read_jsonl(ROOT/'data/controlled/unrelated/supplemental_donors.jsonl')}
 grouped=defaultdict(set)
 for r in mapping:
  if isinstance(r['source_test'],str):grouped[r['source_task_id']].add(r['source_test'])
 def check_source(item):
  tid,tests=item;code=original[tid]['reference_code'] if tid in original else extras[tid]['reference_code']
  result=execute(code,tests=sorted(tests),timeout=15)
  if not all(result['passed']) or len(result['passed'])!=len(tests):raise ValueError((tid,result))
  return len(tests)
 with ThreadPoolExecutor(max_workers=8) as pool:validated=sum(pool.map(check_source,grouped.items()))
 def check_target(t):
  if t['source_dataset']=='livecodebench':return None
  src=original[t['task_id']]
  assert t['official_test']==src['official_test'] and t['heldout_test']==src['heldout_test'] and t['prompt']==src['prompt']
  assert len(t['condition_tests']['unrelated_tests'])==len(src['condition_tests']['nl_tests'])
  result=execute(t['reference_code'],tests=t['condition_tests']['unrelated_tests'],timeout=15)
  return dict(task_id=t['task_id'],passed=result['passed'],errors=result.get('errors',[]),status=result['status'])
 with ThreadPoolExecutor(max_workers=8) as pool:target=[r for r in pool.map(check_target,tasks) if r]
 report={'source_function_assertions_validated':validated,'source_function_tasks':len(grouped),'function_targets':len(target),'recipient_suites_all_correct':sum(all(r['passed']) for r in target),'recipient_assertions_correct':sum(sum(r['passed']) for r in target),'recipient_error_types':dict(Counter(e.split(':')[0] for r in target for e in r['errors'] if e)),'recipient_results':target}
 report['recipient_suites_with_runtime_error_or_timeout']=sum(any(e and not e.startswith('AssertionError:') for e in r['errors']) or r['status']!='ok' for r in target)
 report['recipient_timeouts']=sum(r['status']=='timeout' for r in target)
 out=ROOT/'outputs/analysis';out.mkdir(exist_ok=True,parents=True);(out/'unrelated_control_audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='recipient_results'}))
if __name__=='__main__':main()
