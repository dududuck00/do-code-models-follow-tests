"""Package the completed representation supplement and its reproducible evidence."""
import json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def files_under(path):
    return [p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
def pack(destination,files):
    partial=destination.with_suffix(destination.suffix+'.partial')
    with partial.open('wb') as out:
        compressor=subprocess.Popen(['xz','-T4','-3','-c'],stdin=subprocess.PIPE,stdout=out)
        with tarfile.open(fileobj=compressor.stdin,mode='w|') as archive:
            for p in sorted(set(files)):
                archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        compressor.stdin.close()
        assert compressor.wait()==0
    expected={str(p.relative_to(ROOT)):p.stat().st_size for p in files}
    with tarfile.open(partial,'r|xz') as archive:
        actual={}
        for member in archive:
            assert member.isfile() and member.name in expected
            data=archive.extractfile(member)
            size=0
            while chunk:=data.read(1024*1024):size+=len(chunk)
            assert size==expected[member.name]
            actual[member.name]=size
    assert actual==expected
    partial.replace(destination)
    return {'path':str(destination.relative_to(ROOT)),'bytes':destination.stat().st_size,'files':len(actual),'entries_and_compressed_stream_verified':True}
def main():
    out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    base=ROOT/'outputs/semantic_representation'
    verification=json.loads((base/'verification.json').read_text());assert verification['complete']
    paper=files_under(ROOT/'paper/semantic_representation')
    source=[p for p in paper if p.name!='semantic_representation.pdf']+[ROOT/'scripts/build_semantic_representation.sh']
    files=files_under(base)+paper+files_under(ROOT/'src')+files_under(ROOT/'data/model_tokenizers/deepseek-coder-6.7b-instruct')
    files += [ROOT/p for p in ['README.md','PROJECT_STATE.md','requirements.txt','requirements-qwen36.txt','requirements-model.txt','configs/controlled.yaml','configs/semantic_representation.yaml','data/controlled/semantic/tasks.jsonl','tests/test_semantic_representation.py','scripts/collect_generation_states.py','scripts/build_semantic_representation.sh']]
    files += list((ROOT/'scripts').glob('*semantic_representation*.py'))
    for model in verification['models']:
        files += [ROOT/f'outputs/controlled/semantic/{model}/{name}' for name in ('generations.jsonl','eval_results.jsonl','protocol.json')]
    assert sum(p.suffix=='.npz' for p in files)==verification['unique_prompt_states']
    result={'latex':pack(out/'tdd_semantic_representation_latex_20260908.tar.xz',source),'complete':pack(out/'tdd_semantic_representation_20260908.tar.xz',files)}
    (out/'semantic_representation_package.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
