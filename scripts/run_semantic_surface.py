"""Collect surface-control states using the existing prompt-state collector."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import yaml
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queues',nargs='+',default=['0:qwen25,deepseek,qwen35,qwen36,qwen38']);args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/'configs/controlled.yaml').read_text())
    tmp=ROOT/'.tmp/semantic_surface';tmp.mkdir(parents=True,exist_ok=True)
    def run(spec):
        gpu,models=spec.split(':')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,TMPDIR=str(ROOT/'.tmp'),TMP=str(ROOT/'.tmp'),TEMP=str(ROOT/'.tmp'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        for m in models.split(','):
            out=ROOT/f'outputs/semantic_surface/{m}'
            cmd=[sys.executable,'-B','scripts/collect_generation_states.py','--generations',str(out/'prompts.jsonl'),'--model-path',cfg['models'][m],'--output-dir',str(out/'replay'),'--device','cuda:0','--dtype','bfloat16','--prompt-format','chat','--thinking','disabled','--resume']
            if m in cfg.get('tokenizers',{}):cmd+=['--tokenizer-path',cfg['tokenizers'][m]]
            print('Starting',m,flush=True)
            with (tmp/f'{m}.log').open('a') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            p=out/'replay/generations.jsonl';rs=[json.loads(s) for s in p.read_text().splitlines()]
            for r in rs:
                state=Path(r['state_path']);r['state_path']=str(state.relative_to(ROOT) if state.is_absolute() else state)
            p.write_text(''.join(json.dumps(r)+'\n' for r in rs));print('Completed',m,len(rs),flush=True)
    with ThreadPoolExecutor(max_workers=len(args.queues)) as pool:list(pool.map(run,args.queues))
if __name__=='__main__':main()
