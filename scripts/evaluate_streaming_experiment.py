#!/usr/bin/env python
"""Evaluate saved generation ranges inside the caller's execution sandbox."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from merge_controlled_generations import merge
from tddexp.io import write_jsonl
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--parts',nargs='+',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--timeout',type=float,default=15)
    p.add_argument('--lcb-root',default='.tmp/LiveCodeBench')
    p.add_argument('--output-dir',required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    snapshot=ROOT/'.tmp'/f'{out.parent.name}_{out.name}_generation_snapshot.jsonl'
    last=0
    while True:
        rows,expected=merge(a.parts,a.dataset,a.repeats)
        if len(rows)<expected and len(rows)-last<256:
            time.sleep(30);continue
        write_jsonl(snapshot,rows)
        complete=len(rows)==expected
        destination=out/('eval_results.jsonl' if complete else 'eval_results.partial.jsonl')
        command=[sys.executable,'-B','scripts/evaluate_controlled.py','--dataset',a.dataset,'--generations',str(snapshot),
                 '--output',str(destination),'--checkpoint',str(out/'evaluation_checkpoint.jsonl'),'--workers',str(a.workers),'--timeout',str(a.timeout),'--lcb-root',a.lcb_root]
        if not complete:command.append('--allow-partial')
        print(json.dumps({'generation_rows':len(rows),'expected':expected,'complete':complete}),flush=True)
        subprocess.run(command,check=True,cwd=ROOT)
        last=len(rows)
        if complete:
            if len(a.parts)!=1 or Path(a.parts[0]).resolve()!=(out/'generations.jsonl').resolve():
                write_jsonl(out/'generations.jsonl',rows)
            subprocess.run([sys.executable,'-B','scripts/analyze_controlled.py','--eval-results',str(destination),'--output',str(out/'summary.json')],check=True,cwd=ROOT)
            return
if __name__=='__main__':main()
