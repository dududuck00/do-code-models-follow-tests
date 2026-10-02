"""Bundle the surface experiment, original states, and source needed for analysis."""
import json
from pathlib import Path
from package_semantic_representation import pack,files_under
ROOT=Path(__file__).resolve().parents[1]
def main():
    base=ROOT/'outputs/semantic_surface';v=json.loads((base/'verification.json').read_text());assert v['complete'] and len(v['models'])==5
    paper=files_under(ROOT/'paper/semantic_surface')
    source=[p for p in paper if p.suffix!='.pdf']+[ROOT/'scripts/build_semantic_surface.sh']
    files=paper+files_under(base)+files_under(ROOT/'outputs/semantic_representation')+files_under(ROOT/'src')+files_under(ROOT/'data/model_tokenizers/deepseek-coder-6.7b-instruct')
    files += [ROOT/p for p in ['README.md','PROJECT_STATE.md','requirements.txt','requirements-qwen36.txt','configs/controlled.yaml','configs/semantic_representation.yaml','configs/semantic_surface.yaml','data/controlled/semantic/tasks.jsonl','docs/semantic_activation_design.md','scripts/collect_generation_states.py','scripts/build_semantic_surface.sh']]
    files += list((ROOT/'scripts').glob('*semantic_surface*.py'))+list((ROOT/'scripts').glob('*semantic_representation*.py'))
    for m in v['models']:
        files += [ROOT/f'outputs/controlled/semantic/{m}/{n}' for n in ['generations.jsonl','eval_results.jsonl','protocol.json']]
    assert sum(p.suffix=='.npz' for p in files)==3600
    out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    result={'latex':pack(out/'tdd_semantic_surface_latex_20260909.tar.xz',source),'complete':pack(out/'tdd_semantic_surface_20260909.tar.xz',files)}
    (out/'semantic_surface_package.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
