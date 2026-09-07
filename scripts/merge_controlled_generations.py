#!/usr/bin/env python
"""Merge disjoint generation ranges, optionally while their writers are active."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tddexp.io import read_jsonl,write_jsonl

def merge(parts,dataset,repeats):
    rows=[];seen=set();models=set()
    for path in parts:
        if not Path(path).exists():continue
        data=Path(path).read_bytes()
        complete=data.rsplit(b'\n',1)[0] if b'\n' in data else b''
        for line in complete.splitlines():
            if not line.strip():continue
            row=json.loads(line);key=row['task_id'],row['condition'],row.get('repeat',0)
            if key in seen:raise ValueError(f'Duplicate generation: {key}')
            seen.add(key);models.add(row['model_path']);rows.append(row)
    tasks=read_jsonl(dataset)
    expected={(t['task_id'],c,r) for t in tasks for c in t['condition_prompts'] for r in range(repeats)}
    if not seen.issubset(expected) or len(models)>1:raise ValueError('Unexpected task/condition/repeat or mixed checkpoints.')
    return rows,len(expected)

def main():
    p=argparse.ArgumentParser();p.add_argument('--parts',nargs='+',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    rows,expected=merge(a.parts,a.dataset,a.repeats)
    if len(rows)!=expected and not a.allow_partial:raise SystemExit(f'Incomplete: {len(rows)}/{expected}')
    write_jsonl(a.output,rows);print(json.dumps({'rows':len(rows),'expected':expected,'complete':len(rows)==expected}))
if __name__=='__main__':main()
