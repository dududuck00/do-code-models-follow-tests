#!/usr/bin/env python
"""Audit the prompts actually used in the archived experiments."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tddexp.controlled import saved_prompt_tests,same_test_set,normalized_test
from tddexp.data import load_tasks
from tddexp.execution import execute
from tddexp.io import read_jsonl,write_jsonl

RUNS={'mbppplus':'mbppplus_qwen_full','humanevalplus':'humanevalplus_qwen_structured',
      'lcb_7b':'livecodebench_v6_minus_v5_qwen_full_fixed',
      'lcb_27b':'livecodebench_v6_new_qwen36_27b_nothink',
      'lcb_27b_high5':'livecodebench_v6_new_qwen36_27b_synth_high5_nothink'}


def io_pair(test):
    if test.startswith('input: ') and '\nexpected output: ' in test:
        return tuple(test.split('\nexpected output: ',1))
    try:
        node=ast.parse(test).body[0].test
        if isinstance(node,ast.Compare) and len(node.ops)==1 and isinstance(node.ops[0],ast.Eq):
            return ast.dump(node.left),ast.dump(node.comparators[0])
    except (SyntaxError,AttributeError,IndexError): pass
    return None


def interface(tests):
    calls=set()
    for test in tests:
        try: tree=ast.parse(test)
        except SyntaxError: continue
        calls.update((n.func.id,len(n.args)) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name))
    return calls


def reference_check(item):
    detail,code,left,right=item
    for name,tests in [('original',left),('shuffled',right)]:
        result=execute(code,tests=tests,timeout=8)
        detail[name+'_reference_passed']=bool(tests) and result['status']=='ok' and all(result['passed'])
    return detail


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='outputs/analysis/control_audit');a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    report={}; details=[]; reference_jobs=[]
    references={s:{t.task_id:t.imports+'\n'+t.canonical_code for t in load_tasks(ROOT/'data'/f'{s}.jsonl',s)} for s in ['mbppplus','humanevalplus']}
    for label,run in RUNS.items():
        by={}
        for row in read_jsonl(ROOT/'outputs'/run/'generations.jsonl'):
            by.setdefault(row['task_id'],{})[row['condition']]=row
        counts=Counter()
        for task,conditions in by.items():
            base,target=conditions['nl_tests'],conditions['shuffled_tests']
            left,right=saved_prompt_tests(base),saved_prompt_tests(target)
            same=same_test_set(left,right)
            counts['tasks']+=1;counts['unchanged_test_set']+=same
            counts['identical_prompt']+=base['prompt']==target['prompt']
            original_pairs=dict(p for t in left if (p:=io_pair(t)) is not None)
            shuffled_pairs=[p for t in right if (p:=io_pair(t)) is not None]
            changed=sum(original_pairs[x]!=y for x,y in shuffled_pairs if x in original_pairs)
            paired=sum(x in original_pairs for x,y in shuffled_pairs)
            counts['parsed_paired_outputs']+=paired;counts['changed_outputs']+=changed
            irrelevant=saved_prompt_tests(conditions['irrelevant_tests'])
            mismatch=bool(interface(left)) and interface(left)!=interface(irrelevant)
            counts['irrelevant_call_interface_changed']+=mismatch
            detail={'run':label,'task_id':task,'test_count':len(left),'unchanged_test_set':same,
                    'identical_prompt':base['prompt']==target['prompt'],'paired_outputs':paired,
                    'changed_outputs':changed,'irrelevant_call_interface_changed':mismatch}
            details.append(detail)
            if label in references:reference_jobs.append((detail,references[label][task],left,right))
        report[label]=dict(counts)
    high5=read_jsonl('data/synthetic_public_tests/livecodebench_v6_new/livecodebench_v6_new_synth_high5.jsonl')
    report['high5_coverage']={'actual_test_counts':dict(Counter(len(json.loads(r['public_test_cases'])) for r in high5)),
                              'fallback_tasks':sum(r['synthetic_public_tests_metadata']['fallback_public_used'] for r in high5)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(reference_check,reference_jobs))
    for label in references:
        group=[r for r in details if r['run']==label]
        report[label]['original_reference_correct_suites']=sum(r['original_reference_passed'] for r in group)
        report[label]['shuffled_reference_correct_suites']=sum(r['shuffled_reference_passed'] for r in group)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    write_jsonl(out/'task_audit.jsonl',details)
    print(json.dumps(report))


if __name__=='__main__':main()
