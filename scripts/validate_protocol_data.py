#!/usr/bin/env python
"""Check reference programs against the frozen evaluators and visible suites."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tddexp.execution import execute
from tddexp.io import read_jsonl,write_jsonl


def check(task, timeout=15):
    timeout=task.get('evaluation_timeout_seconds',timeout)
    if task['source_dataset']=='livecodebench':
        return {'task_id':task['task_id'],'status':'official_public_oracle','reference_available':False}
    code=task['reference_code']
    suites={'official':task['official_test'],'heldout':task['heldout_test']}
    result={'task_id':task['task_id'],'reference_available':True}
    for name,suite in suites.items():
        output=execute(code,tests=[suite],timeout=timeout)
        result[name+'_passed']=bool(output['passed'] and all(output['passed']))
        if not result[name+'_passed']:result[name+'_error']=output
    result['status']='passed' if all(result[n+'_passed'] for n in suites) else 'failed'
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='data/controlled/main/tasks.jsonl')
    p.add_argument('--output',default='outputs/analysis/control_audit/reference_validation.jsonl')
    p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(check,read_jsonl(a.dataset)))
    write_jsonl(a.output,rows)
    print(json.dumps({'tasks':len(rows),'failed':sum(r['status']=='failed' for r in rows),
                      'reference_validated':sum(r['status']=='passed' for r in rows)}))


if __name__=='__main__':main()
