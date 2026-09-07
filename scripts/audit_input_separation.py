#!/usr/bin/env python
"""Audit candidate/development input overlap with the final evaluation inputs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tddexp.controlled import filter_evalplus_test,input_key,literal_call_inputs
from tddexp.io import read_jsonl

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lcb-root', default='.tmp/LiveCodeBench')
    args=parser.parse_args()
    report={}
    main_tasks={r['task_id']:r for r in read_jsonl(ROOT/'data/controlled/main/tasks.jsonl')}
    for experiment in ['main','quality']:
        tasks=read_jsonl(ROOT/f'data/controlled/{experiment}/tasks.jsonl')
        details=[]
        for t in tasks:
            if t['source_dataset']=='livecodebench':continue
            if experiment=='main':tests=t['condition_tests']['nl_tests']
            else:
                tests=t['candidate_tests']
                original=main_tasks[t['task_id']]
                tests=tests+original['condition_tests']['nl_tests']
            excluded={input_key(literal_call_inputs(s,t['entry_point'])) for s in tests}
            _,counts=filter_evalplus_test(t['heldout_test'],excluded)
            if counts['removed']:raise ValueError(f'Residual evaluation overlap: {t["task_id"]}')
            details.append(dict(task_id=t['task_id'],candidate_and_development_inputs=len(excluded),
                                removed_from_official=t['heldout_counts']['removed'],heldout_inputs=counts['remaining']))
        report[experiment]={'tasks':len(details),'residual_overlap':0,
                            'removed_inputs':sum(r['removed_from_official'] for r in details),'details':details}
    out=ROOT/'outputs/analysis/control_audit/input_separation.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items() if x!='details'} for k,v in report.items()}))
    sys.path.insert(0, str(Path(args.lcb_root).resolve()))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from evaluate_controlled import lcb_input_key
    keys={'question_title','question_content','platform','question_id','contest_id','contest_date',
          'starter_code','difficulty','public_test_cases','private_test_cases','metadata'}
    details=[]
    for row in read_jsonl(ROOT/'data/livecodebench_release_v6_minus_v5.jsonl'):
        problem=CodeGenerationProblem(**{k:v for k,v in row.items() if k in keys})
        raw=json.loads(problem.get_evaluation_sample()['input_output'])
        count=len(problem.public_test_cases)
        public=raw['inputs'][:count]
        private=raw['inputs'][count:]
        stripped={x.strip() for x in public}
        normalized={lcb_input_key(x,bool(raw.get('fn_name'))) for x in public}
        exact=sum(x.strip() in stripped for x in private)
        total=sum(lcb_input_key(x,bool(raw.get('fn_name'))) in normalized for x in private)
        details.append(dict(task_id=row['question_id'],private_tests=len(private),
                            public_duplicate_inputs=exact,additional_json_duplicates=total-exact))
    (out.parent/'lcb_input_separation.json').write_text(json.dumps(details,indent=2)+'\n')
    print(json.dumps({'lcb_tasks':len(details),'private_tests':sum(r['private_tests'] for r in details),
                      'excluded_duplicates':sum(r['public_duplicate_inputs']+r['additional_json_duplicates'] for r in details)}))
if __name__=='__main__':main()
