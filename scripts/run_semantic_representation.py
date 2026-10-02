"""Replay the semantic cohorts with independent model processes."""
import argparse,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--queues',nargs='+',default=['3:qwen25','4:qwen35','0:qwen36,qwen38,deepseek']);args=p.parse_args()
    cfg=yaml.safe_load((ROOT/'configs/controlled.yaml').read_text())
    tmp=ROOT/'.tmp/semantic_representation';tmp.mkdir(parents=True,exist_ok=True)
    def run(spec):
        gpu,models=spec.split(':')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,TMPDIR=str(ROOT/'.tmp'),TMP=str(ROOT/'.tmp'),TEMP=str(ROOT/'.tmp'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        for model in models.split(','):
            out=ROOT/f'outputs/semantic_representation/{model}'
            cmd=[sys.executable,'-B','scripts/collect_generation_states.py','--generations',str(out/'unique_prompts.jsonl'),'--model-path',cfg['models'][model],'--output-dir',str(out/'replay'),'--device','cuda:0','--dtype','bfloat16','--prompt-format','chat','--thinking','disabled','--resume']
            if model in cfg.get('tokenizers', {}):
                cmd.extend(['--tokenizer-path', cfg['tokenizers'][model]])
            print(json.dumps({'starting':model,'gpu':gpu}),flush=True)
            with (tmp/f'{model}.log').open('a') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            records=[json.loads(line) for line in (out/'replay/generations.jsonl').open()]
            for row in records:
                state_path=Path(row['state_path'])
                row['state_path']=str(state_path.relative_to(ROOT) if state_path.is_absolute() else state_path)
            (out/'replay/generations.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records))
            print(json.dumps({'completed':model}),flush=True)
    with ThreadPoolExecutor(max_workers=len(args.queues)) as pool:list(pool.map(run,args.queues))
if __name__=='__main__':main()
