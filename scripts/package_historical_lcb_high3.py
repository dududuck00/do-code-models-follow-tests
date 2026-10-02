"""Package the high3 extension with frozen inputs, raw outputs and six-row statistics."""
from pathlib import Path
import io,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
def main():
    result=json.loads((ROOT/'outputs/analysis/historical_lcb_results.json').read_text());assert result['complete']
    run=ROOT/'outputs/livecodebench_v6_new_qwen36_27b_synth_high3_nothink'
    protocol=json.loads((run/'protocol.json').read_text());assert protocol['complete'] and protocol['evaluations']==700
    paths=set()
    def add(p):
        p=ROOT/p
        if p.is_file():paths.add(p)
        else:
            assert p.is_dir(),p
            paths.update(f for f in p.rglob('*') if f.is_file() and not any(x in {'.git','__pycache__'} for x in f.parts) and f.suffix not in {'.pyc','.log'})
    for p in ['src','tests/test_lcb_execution.py','configs/historical_lcb_high3.yaml','data/controlled/historical_lcb_high3','data/livecodebench_release_v6_minus_v5.jsonl.gz','.tmp/LiveCodeBench','outputs/analysis/historical_lcb_results.json','outputs/analysis/historical_lcb_report.md','README.md','requirements.txt','requirements-vllm.txt','aaai/AuthorKit27/AuthorKit27/paper.pdf','artifacts/tdd_paper_latex.tar.xz']:
        add(p)
    for p in ['prepare_historical_lcb_high3.py','run_historical_lcb_high3.py','analyze_historical_lcb.py','finalize_historical_lcb_high3.py','package_historical_lcb_high3.py','run_generation_vllm.py','evaluate_lcb_subset.py','analyze_behavior_flips.py']:
        add('scripts/'+p)
    for p in ['generations.jsonl','generations_historical.jsonl','eval_results_combined.jsonl','protocol.json','verification.json','behavior']:
        add(str((run/p).relative_to(ROOT)))
    for row in result['runs']:add(row['output_dir']+'/eval')
    records={str(p.relative_to(ROOT)):p.stat().st_size for p in sorted(paths)}
    text='''# Qwen3.6-27B Synthetic high3 supplement

Contains 700 new generations and evaluations, frozen high3 prompts, the six-row historical evaluation records, the current paper and its LaTeX source archive. Earlier controlled experiments remain in tdd_results_20260908.tar.xz.

Reproduction instructions are in README.md. Decompress the official dataset with `gzip -dk data/livecodebench_release_v6_minus_v5.jsonl.gz`. Configure the local model path and GPU in configs/historical_lcb_high3.yaml. The frozen prompts are sufficient for rerunning; preparing them again requires the older Qwen2.5 generation archive.

Read outputs/analysis/historical_lcb_report.md first. generations.jsonl retains the current extractor's intermediate candidates; generations_historical.jsonl contains the archived-extractor candidates actually evaluated. Model weights, runtime environments, GPU caches and logs are excluded.
'''
    target=ROOT/'artifacts/tdd_qwen36_high3_20260908.tar.xz'
    with tarfile.open(target,'w:xz',preset=3) as tar:
        for name,data in [('ARCHIVE_README.md',text.encode()),('MANIFEST.json',(json.dumps(records,indent=2)+'\n').encode())]:
            member=tarfile.TarInfo('tdd_qwen36_high3/'+name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
        for p in sorted(paths):tar.add(p,arcname='tdd_qwen36_high3/'+str(p.relative_to(ROOT)),recursive=False)
    with tarfile.open(target,'r:xz') as tar:
        actual={m.name.removeprefix('tdd_qwen36_high3/'):m.size for m in tar.getmembers() if m.name not in ['tdd_qwen36_high3/MANIFEST.json','tdd_qwen36_high3/ARCHIVE_README.md']};assert actual==records
        for name in ['generations.jsonl','generations_historical.jsonl','eval_results_combined.jsonl']:
            data=tar.extractfile('tdd_qwen36_high3/'+str(run.relative_to(ROOT))+'/'+name)
            rows=list(map(json.loads,data));assert len(rows)==700 and len({(r['task_id'],r['condition']) for r in rows})==700
    print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'files':len(records),'verified':True}))
if __name__=='__main__':main()
