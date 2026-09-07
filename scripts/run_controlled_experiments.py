#!/usr/bin/env python
"""Run/resume frozen experiments, evaluate complete outputs, and summarize."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT=Path(__file__).resolve().parents[1]


def run(command, log, env):
    print('Running: '+' '.join(map(str,command)),flush=True)
    with log.open('a') as handle:
        subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/controlled.yaml')
    p.add_argument('--model',required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--experiments',nargs='+',default=['main','semantic','quality'])
    p.add_argument('--gpu-memory-utilization',type=float)
    p.add_argument('--tensor-parallel-size',type=int)
    p.add_argument('--generation-python',default='.venv-vllm/bin/python')
    p.add_argument('--evaluation-python',default='.venv/bin/python')
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--generate-only',action='store_true')
    a=p.parse_args();config=yaml.safe_load(Path(a.config).read_text())
    if a.model not in config['models']:p.error(f'Unknown model: {a.model}')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,
             TMPDIR=str(ROOT/'.tmp'),TMP=str(ROOT/'.tmp'),TEMP=str(ROOT/'.tmp'))
    outputs=[]
    for experiment in a.experiments:
        spec=config['experiments'][experiment]
        if a.model not in spec['models']:continue
        execution=config.get('execution_overrides',{}).get(experiment,{}).get(a.model,{})
        memory_utilization=a.gpu_memory_utilization if a.gpu_memory_utilization is not None else execution.get('gpu_memory_utilization',0.85 if a.model in {'qwen36','qwen38'} else 0.65)
        tensor_parallel_size=a.tensor_parallel_size if a.tensor_parallel_size is not None else execution.get('tensor_parallel_size',1)
        dataset=Path(spec['dataset'])
        if not dataset.exists():raise SystemExit(f'Dataset preparation is incomplete: {dataset}')
        if experiment == 'quality':
            summary_path = dataset.with_name('summary.json')
            while json.loads(summary_path.read_text())['eligible_tasks'] < 180:
                print('Waiting for the requested 180-task quality dataset.',flush=True)
                time.sleep(30)
        out=ROOT/'outputs'/'controlled'/experiment/a.model;out.mkdir(parents=True,exist_ok=True)
        metadata={'experiment':experiment,'model':a.model,'model_path':config['models'][a.model],
                  'config':config['generation'],'repeats':spec['repeats'],'dataset':str(dataset)}
        if not a.evaluate_only:
            metadata['execution']={'gpu':a.gpu,'gpu_memory_utilization':memory_utilization,'tensor_parallel_size':tensor_parallel_size}
        elif (out/'protocol.json').exists():
            metadata=json.loads((out/'protocol.json').read_text())
        (out/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
        if not a.evaluate_only:
            command=[a.generation_python,'-B','scripts/run_generation_vllm.py','--dataset',dataset,
                     '--model-path',config['models'][a.model],'--conditions','auto','--repeats',str(spec['repeats']),
                     '--thinking','disabled','--max-new-tokens',str(config['generation']['max_new_tokens']),
                     '--max-model-len',str(config['generation']['max_model_len']),
                     '--batch-size',str(config['generation']['batch_size']),
                     '--max-num-seqs',str(config['generation']['max_num_seqs']),
                     '--tensor-parallel-size',str(tensor_parallel_size),'--max-num-batched-tokens','8192','--gpu-memory-utilization',str(memory_utilization),
                     '--seed','0','--order-seed','0','--resume','--output-dir',out]
            if a.model in config.get('tokenizers', {}):
                command.extend(['--tokenizer-path', config['tokenizers'][a.model]])
            run(command,out/'generation.log',env)
        if a.generate_only: continue
        command=[a.evaluation_python,'-B','scripts/evaluate_controlled.py','--dataset',dataset,
                 '--generations',out/'generations.jsonl','--output',out/'eval_results.jsonl',
                 '--checkpoint',out/'evaluation_checkpoint.jsonl',
                 '--lcb-root',config['evaluation']['lcb_root'],'--workers',str(config['evaluation']['workers']),
                 '--timeout',str(config['evaluation']['timeout_seconds'])]
        run(command,out/'evaluation.log',env)
        run([a.evaluation_python,'-B','scripts/analyze_controlled.py','--eval-results',out/'eval_results.jsonl',
             '--output',out/'summary.json'],out/'analysis.log',env)
        outputs.append(str(out/'summary.json'))
    print(json.dumps({'completed_summaries':outputs}),flush=True)


if __name__=='__main__':main()
