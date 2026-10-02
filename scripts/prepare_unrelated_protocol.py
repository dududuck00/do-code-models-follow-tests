#!/usr/bin/env python
"""Freeze other-task tests with matched interfaces and unchanged main evaluators."""
import ast,copy,json,math,random,re,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tddexp.controlled import target_call,input_key,prompt_with_tests
from tddexp.io import read_jsonl,write_jsonl

def type_shape(x):
 if isinstance(x,(list,tuple)):
  return (type(x).__name__,tuple(sorted({repr(type_shape(v)) for v in x})))
 if isinstance(x,dict):return ('dict',tuple(sorted({repr(type_shape(k))+':'+repr(type_shape(v)) for k,v in x.items()})))
 return type(x).__name__

def function_case(task,text):
 c=target_call(text,task['entry_point']);args=tuple(ast.literal_eval(a) for a in c.args)
 node=ast.parse(text).body[0].test
 return dict(args=args,output=ast.literal_eval(node.comparators[0]),text=text,mode='function',source_id=task['task_id'],source_entry=task['entry_point'])

def function_keys(code):
 keys=set()
 class Constants(ast.NodeTransformer):
  def visit_Name(self,n):
   return ast.copy_location(ast.Constant(float(n.id)),n) if n.id in {'inf','nan'} else n
 for node in ast.walk(ast.parse(code)):
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='inputs' for t in node.targets):
   keys.update(repr(tuple(x)) for x in ast.literal_eval(Constants().visit(copy.deepcopy(node.value))))
 return keys

def render(case,target):
 if case['mode']=='function':
  tree=ast.parse(case['text'])
  class Rename(ast.NodeTransformer):
   def visit_Name(self,n):
    return ast.copy_location(ast.Name(id=target['entry_point'],ctx=n.ctx),n) if n.id==case['source_entry'] else n
  return ast.unparse(ast.fix_missing_locations(Rename().visit(tree)))
 return 'input: '+case['input']+'\nexpected output: '+case['raw_output']

def case_key(c):
 if '_key' not in c:
  c['_key']=repr(tuple(c['args'])) if c['mode']=='function' else (json.dumps(c['args'],sort_keys=True,separators=(',',':')) if c['mode']=='functional' else c['input'].strip())
 return c['_key']
def profile(c):
 if c['mode']=='stdin':
  return tuple(tuple('number' if re.fullmatch(r'-?\d+(?:\.\d+)?',x) else 'text' for x in line.split()) for line in c['input'].splitlines())
 return tuple(type_shape(a) for a in c['args'])
def token_set(r):return set(re.findall('[a-z]{3,}',r['prompt'].lower()))

def supplements():
 rng=random.Random(20260908);specs=[]
 code='def donor(values, n, lo, hi):\n    return sum(lo <= x <= hi for x in values[:n])\n'
 cases=[]
 for i in range(30):
  xs=[rng.randint(-25,25) for _ in range(rng.randint(5,12))];lo,hi=sorted(rng.sample(range(-10,20),2));args=(xs,len(xs),lo,hi)
  scope={};exec(code,scope);value=scope['donor'](*args)
  cases.append(dict(args=args,output=value,text=f'assert donor({", ".join(map(repr,args))}) == {value!r}',mode='function',source_id='supplement::count_prefix_in_range',source_entry='donor'))
 specs.append(dict(task_id='supplement::count_prefix_in_range',description='Count prefix elements whose values lie between two inclusive numeric bounds.',reference_code=code,cases=cases))
 code='def donor(a, b, xs, ys):\n    return a * sum(xs) + b * sum(ys)\n'
 cases=[]
 for i in range(30):
  a=rng.randint(15,80);b=rng.randint(1,4);xs=sorted(rng.sample(range(a-1),rng.randint(3,8)));ys=[x+1 for x in xs];args=(a,b,xs,ys)
  scope={};exec(code,scope);value=scope['donor'](*args)
  cases.append(dict(args=args,output=value,input='\n'.join(json.dumps(x) for x in args),raw_output=json.dumps(value),mode='functional',source_id='supplement::weighted_list_sums'))
 specs.append(dict(task_id='supplement::weighted_list_sums',description='Return a weighted sum of two integer lists, using a separate integer weight for each list.',reference_code=code,cases=cases))
 return specs

def main():
 tasks=read_jsonl(ROOT/'data/controlled/main/tasks.jsonl');byid={r['task_id']:r for r in tasks};pool=[];own={};excluded={}
 for task in tasks:
  if task['source_dataset']=='livecodebench':continue
  own[task['task_id']]=[function_case(task,t) for t in task['condition_tests']['nl_tests']]
  pool.extend(own[task['task_id']]);excluded[task['task_id']]=function_keys(task['official_test'])|{case_key(c) for c in own[task['task_id']]}
 print('Function input exclusions ready.',flush=True)
 sys.path.insert(0,str(ROOT/'.tmp/LiveCodeBench'))
 from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
 for row in read_jsonl(ROOT/'data/livecodebench_release_v6_minus_v5.jsonl'):
  tid='livecodebench::'+row['question_id'];public=json.loads(row['public_test_cases']);mode=public[0]['testtype'];cases=[]
  for t in public:
   args=tuple(json.loads(x) for x in t['input'].strip().splitlines()) if mode=='functional' else ()
   try:output=json.loads(t['output'])
   except ValueError:output=t['output']
   cases.append(dict(args=args,output=output,input=t['input'],raw_output=t['output'],mode=mode,source_id=tid))
  own[tid]=cases;pool.extend(cases)
  fields={'question_title','question_content','platform','question_id','contest_id','contest_date','starter_code','difficulty','public_test_cases','private_test_cases','metadata'}
  problem=CodeGenerationProblem(**{k:v for k,v in row.items() if k in fields});sample=json.loads(problem.get_evaluation_sample()['input_output'])
  excluded[tid]={json.dumps([json.loads(y) for y in x.strip().splitlines()],sort_keys=True,separators=(',',':')) if mode=='functional' else x.strip() for x in sample['inputs']}
 print('LCB input exclusions ready.',flush=True)
 extra=supplements()
 for s in extra:pool.extend(s['cases'])
 out=ROOT/'data/controlled/unrelated';out.mkdir(parents=True,exist_ok=True)
 write_jsonl(out/'supplemental_donors.jsonl',extra)
 mapping=[];new=[];failures=[]
 for task_index,task in enumerate(tasks):
  if task_index%100==0:print(f'Matching task {task_index}/{len(tasks)}',flush=True)
  tid=task['task_id'];selected=[];used=set();references=own[tid];tokens=token_set(task)
  for pos,original in enumerate(references):
   candidates=[]
   for c in pool:
    if c['source_id']==tid or c['mode']!=original['mode']:continue
    if c['mode']!='stdin' and len(c['args'])!=len(original['args']):continue
    if case_key(c) in excluded[tid] or case_key(c) in used:continue
    donor=byid.get(c['source_id'])
    if donor and donor['prompt']==task['prompt']:continue
    if donor and donor.get('reference_code')==task.get('reference_code') and task.get('reference_code'):continue
    text=render(c,task);target_text=task['condition_tests']['nl_tests'][pos]
    same_profile=profile(c)==profile(original);same_output=type_shape(c['output'])==type_shape(original['output'])
    length=abs(math.log(max(1,len(text))/max(1,len(target_text))))
    dt=token_set(donor) if donor else set();overlap=len(tokens&dt)/max(1,len(tokens|dt))
    score=(not same_profile,not same_output,length+overlap,c['source_id'],text)
    candidates.append((score,c,text,same_profile,same_output))
   if not candidates:
    failures.append((tid,pos,len(original['args']),original['mode']));break
   _,c,text,ip,op=min(candidates,key=lambda x:x[0]);selected.append(text);used.add(case_key(c))
   mapping.append(dict(task_id=tid,test_index=pos,source_task_id=c['source_id'],source_test=c.get('text',{'input':c.get('input'),'output':c.get('raw_output')}),displayed_test=text,input_profile_matched=ip,output_profile_matched=op,length_ratio=len(text)/len(task['condition_tests']['nl_tests'][pos]),evaluation_input_overlap=False))
  if len(selected)!=len(references):continue
  r=copy.deepcopy(task);r['experiment']='unrelated';r['condition_tests']={'unrelated_tests':selected};r['condition_prompts']={'unrelated_tests':prompt_with_tests(task['prompt'],selected)};new.append(r)
 if failures:
  print(json.dumps({'unmatched':failures}));raise SystemExit(1)
 write_jsonl(out/'tasks.jsonl',new);write_jsonl(out/'donor_mapping.jsonl',mapping)
 summary=dict(tasks=len(new),conditions=['unrelated_tests'],repeats=3,models=5,expected_generations=len(new)*15,tests=len(mapping),input_profile_matched=sum(x['input_profile_matched'] for x in mapping),output_profile_matched=sum(x['output_profile_matched'] for x in mapping),supplemental_targets=sorted({x['task_id'] for x in mapping if x['source_task_id'].startswith('supplement::')}),evaluation_input_overlap=0,source='Other benchmark tasks plus two authored four-argument donor tasks; original main evaluators retained.')
 ratios=sorted(x['length_ratio'] for x in mapping)
 summary['length_ratio_median']=(ratios[(len(ratios)-1)//2]+ratios[len(ratios)//2])/2
 summary['length_within_15_percent']=sum(.85<=x<=1.15 for x in ratios)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
