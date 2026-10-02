#!/usr/bin/env python
"""Compare the other-task condition against the completed matched main runs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tddexp.io import read_jsonl
from tddexp.statistics import paired_effect
MODELS={'qwen25':'Qwen2.5-Coder-7B','qwen36':'Qwen3.6-27B','deepseek':'DeepSeek-Coder-6.7B','qwen35':'Qwen3.5-9B','qwen38':'Qwen3.8-27B'}
DATASETS={'humanevalplus':'HumanEval+','mbppplus':'MBPP+','livecodebench':'LCB v6-new'}
CONDITIONS=['nl_only','nl_tests','wrong_tests','inputs_only','unrelated_tests']
def effect(r):
 def f(x):return f'{round(x*100,1) or 0:+.1f}'
 return f"{f(r['paired_difference'])} [{f(r['ci95'][0])}, {f(r['ci95'][1])}]"
def main():
 tasks=read_jsonl(ROOT/'data/controlled/unrelated/tasks.jsonl');expected={(r['task_id'],i) for r in tasks for i in range(3)};results={};tables=[];runs=[]
 for model,label in MODELS.items():
  added=read_jsonl(ROOT/f'outputs/controlled/unrelated/{model}/eval_results.jsonl')
  assert len(added)==len(expected) and {(r['task_id'],r['repeat']) for r in added}==expected and all(r['condition']=='unrelated_tests' for r in added)
  original=read_jsonl(ROOT/f'outputs/controlled/main/{model}/eval_results.jsonl');assert len(original)==4*len(expected)
  rows=original+added
  for dataset,name in DATASETS.items():
   group=[r for r in rows if r['source_dataset']==dataset];rates={c:sum(r['passed'] for r in group if r['condition']==c)/sum(r['condition']==c for r in group) for c in CONDITIONS}
   contrasts=[paired_effect(group,'unrelated_tests','nl_tests'),paired_effect(group,'nl_only','unrelated_tests')]
   results[f'{model}/{dataset}']={'tasks':len({r['task_id'] for r in group}),'rates':rates,'contrasts':contrasts}
   tables.append([label,name]+[f'{rates[c]*100:.1f}' for c in CONDITIONS]+[effect(r) for r in contrasts])
   for contrast in contrasts:runs.append([label,name,'正确 − 无关' if contrast['target']=='nl_tests' else '无关 − NL']+[f"{r['paired_difference']*100:+.1f}" for _,r in sorted(contrast['per_repeat'].items())])
 summary=json.loads((ROOT/'data/controlled/unrelated/summary.json').read_text())
 audit=json.loads((ROOT/'outputs/analysis/unrelated_control_audit.json').read_text())
 out=ROOT/'outputs/analysis';out.mkdir(parents=True,exist_ok=True)
 report={'complete':True,'added_generations':len(expected)*len(MODELS),'added_evaluations':len(expected)*len(MODELS),'baseline_generations_reused':len(expected)*len(MODELS)*4,'dataset':summary,'control_audit':{k:v for k,v in audit.items() if k!='recipient_results'},'results':results}
 (out/'unrelated_results.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# 来自其他任务的测试：补充实验','',f"共 {len(tasks)} 题、五个模型、每条件三次运行，新增 {len(expected)*len(MODELS):,} 条生成及对应评测。原四条件直接使用已完成的主实验记录。",'',
 '来源测试对来源任务本身正确。函数测试统一为当前函数名；调用参数数量保持一致，并优先匹配输入输出类型与文本长度。LCB 保持调用方式一致。两个唯一四参数接口使用独立编写的来源任务，其参考实现与测试一并保留。来源测试输入与当前任务的可见及评测输入不重叠，隐藏评测集与原主实验完全相同。','',
 f"共 {summary['tests']} 条展示测试，输入结构类型完全匹配 {summary['input_profile_matched']} 条，输出结构类型完全匹配 {summary['output_profile_matched']} 条。逐条来源、长度比和匹配情况见 `data/controlled/unrelated/donor_mapping.jsonl`。",'',
 f"来源与原测试的长度比中位数为 {summary['length_ratio_median']:.1f}，{summary['length_within_15_percent']}/{summary['tests']} 条测试的长度差在 ±15% 内。在 535 道函数题的目标参考实现上，{audit['recipient_suites_all_correct']} 组来源测试恰好全部正确，{audit['recipient_suites_with_runtime_error_or_timeout']} 组出现运行错误或超时。逐条表现保留在 `unrelated_control_audit.json`。",'',
 '下表为隐藏评测正确率（%）；差值为百分点及按任务配对 bootstrap 的 95% 区间。三次差值先在题内平均，再重采样任务。','',
 '| 模型 | 数据集 | NL-only | 正确 I/O | 错误 I/O | 仅输入 | 来自其他任务 | 正确 − 无关 | 无关 − NL |','|---|---|---:|---:|---:|---:|---:|---|---|']
 lines+=['| '+' | '.join(row)+' |' for row in tables]
 lines+=['','## 每次运行的配对差值','','| 模型 | 数据集 | 对比 | 第一次 | 第二次 | 第三次 |','|---|---|---|---:|---:|---:|']
 lines+=['| '+' | '.join(row)+' |' for row in runs]
 lines+=['','这些对比测量来自其他任务的示例所产生的效果，以及任务相关正确示例相对它们的优势。来源测试在当前任务中可能成为错误约束，因此结合原有错误输出和仅输入条件解释。','']
 (out/'unrelated_report.md').write_text('\n'.join(lines))
 tex=[r'\begin{table*}[t]',r'\centering',r'\small',r'\begin{tabular}{llrrrll}',r'\toprule',r'Model & Dataset & NL-only & Correct & Other-task & Correct $-$ Other & Other $-$ NL \\',r'\midrule']
 for row in tables:tex.append(' & '.join([row[0].replace('-Coder',''),row[1],row[2],row[3],row[6],row[7],row[8]])+r' \\')
 tex += [r'\bottomrule',r'\end{tabular}',r'\caption{Other-task test control. Held-out pass rates (\%) average three runs on the same tasks and evaluators as the main experiment. Differences are percentage points with paired task-bootstrap 95\% intervals.}',r'\label{tab:unrelated}',r'\end{table*}','']
 (ROOT/'aaai/AuthorKit27/AuthorKit27/Tables/unrelated.tex').write_text('\n'.join(tex))
 print(json.dumps({'complete':True,'added_generations':report['added_generations']}))
if __name__=='__main__':main()
