"""Run or resume the historical Qwen3.6 high3 generation/evaluation extension."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['generate','evaluate'],required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
    os.chdir(ROOT)
    for k in ['TMPDIR','TMP','TEMP']:os.environ[k]=str(ROOT/'.tmp')
    c=yaml.safe_load((ROOT/'configs/historical_lcb_high3.yaml').read_text());out=ROOT/c['output_dir'];out.mkdir(parents=True,exist_ok=True)
    gen=out/'generations.jsonl'
    if a.stage=='generate':
        cmd=[str(ROOT/'.venv-vllm/bin/python'),'-B','scripts/run_generation_vllm.py','--dataset',c['dataset'],'--model-path',c['model_path'],'--conditions',*c['conditions'],'--output-dir',str(out),'--resume']
        for k,v in c['generation'].items():
            if isinstance(v,bool):
                if v:cmd.append('--'+k.replace('_','-'))
            else:cmd.extend(['--'+k.replace('_','-'),str(v)])
        os.environ['CUDA_VISIBLE_DEVICES']=c['execution']['gpu']
        subprocess.run(cmd,check=True);return
    expected={(json.loads(line)['task_id'],condition) for line in (ROOT/c['dataset']).open() for condition in c['conditions']}
    while True:
        rows=[json.loads(line) for line in gen.open() if line.strip()] if gen.exists() else []
        keys={(r['task_id'],r['condition']) for r in rows}
        if keys==expected and len(rows)==len(expected):break
        if not a.wait:raise SystemExit(f'Incomplete generations: {len(rows)}/{len(expected)}')
        time.sleep(30)
    frozen={r['task_id']:r for r in map(json.loads,(ROOT/c['dataset']).open())}
    for r in rows:
        task=frozen[r['task_id']];assert r['prompt']==task['condition_prompts'][r['condition']];assert r['prompt_tests']==task['condition_tests'][r['condition']]
        assert r['thinking']=='disabled' and r['max_new_tokens']==8192 and r['temperature']==0
    sys.path.insert(0,str(ROOT/'src'))
    from tddexp.historical_code_utils import assemble_candidate
    for r in rows:
        r['candidate_code']=assemble_candidate(r['prompt'],r['completion'],r['entry_point'],prompt_is_code_prefix=False)
        r['candidate_extraction']='historical_20260725'
    gen=out/'generations_historical.jsonl'
    gen.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    evaldir=out/'eval';e=c['evaluation']
    for condition in c['conditions']:
        summary=evaldir/f'{condition}_eval_summary.json'
        if summary.exists() and json.loads(summary.read_text())['num_tasks']==175:continue
        cmd=[sys.executable,'-B',e['script'],'--generations',str(gen),'--condition',condition,'--output-dir',str(evaldir),'--pipe-workers']
        for k in ['dataset','livecodebench_root','timeout','num_process_evaluate']:cmd.extend(['--'+k.replace('_','-'),str(e[k])])
        subprocess.run(cmd,check=True)
    merged=[]
    for condition in c['conditions']:
        merged.extend(map(json.loads,(evaldir/f'{condition}_eval_all.jsonl').open()))
    assert {(r['task_id'],r['condition']) for r in merged}==expected and len(merged)==700
    (out/'eval_results_combined.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
    (out/'protocol.json').write_text(json.dumps(dict(c,complete=True,generations=700,evaluations=700),ensure_ascii=False,indent=2)+'\n')
    subprocess.run([sys.executable,'-B','scripts/analyze_behavior_flips.py','--run','qwen36_synthetic_high3='+str(evaldir),'--output-dir',str(out/'behavior')],check=True)
if __name__=='__main__':main()
