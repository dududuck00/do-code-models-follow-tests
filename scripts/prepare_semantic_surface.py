"""Prepare executable-equivalent surface variants of the frozen A/B visible tests."""
import ast,json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def variants(tests):
    parenthesized=['assert ('+s.removeprefix('assert ')+')' for s in tests]
    assert all(ast.dump(ast.parse(a))==ast.dump(ast.parse(b)) for a,b in zip(tests,parenthesized))
    return {'parenthesized':parenthesized,'reverse_order':tests[::-1]}
def verdicts(code,tests):
    ns={};exec(code,ns)
    flags=[]
    for test in tests:
        try:exec(test,ns);flags.append(True)
        except AssertionError:flags.append(False)
    return flags

def main():
    cfg=yaml.safe_load((ROOT/'configs/semantic_surface.yaml').read_text())
    paths=yaml.safe_load((ROOT/'configs/controlled.yaml').read_text())['models']
    tasks=[json.loads(s) for s in (ROOT/cfg['dataset']).read_text().splitlines()]
    records=[]
    for t in tasks:
        for rule in ['a','b']:
            c='tests_'+rule;tests=t['condition_tests'][c];assert len(tests)==3
            for s in tests:
                node=ast.parse(s).body[0].test
                assert isinstance(node,ast.Compare) and len(node.ops)==1 and isinstance(node.ops[0],ast.Eq)
                call=node.left;assert isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='solve'
                for arg in call.args:ast.literal_eval(arg)
                ast.literal_eval(node.comparators[0])
            for v,new_tests in variants(tests).items():
                for reference in ['a','b']:
                    old=verdicts(t['reference_'+reference],tests)
                    new=verdicts(t['reference_'+reference],new_tests)
                    assert new==(old[::-1] if v=='reverse_order' else old)
                    if reference==rule:assert all(new)
                before,tail=t['condition_prompts'][c].split('### Visible Test Cases\n',1)
                old_section,after=tail.split('\n\n### Answer',1)
                section='\n\n'.join(f'Test {i+1}\n{s}' for i,s in enumerate(new_tests))
                prompt=before+'### Visible Test Cases\n'+section+'\n\n### Answer'+after
                assert prompt!=t['condition_prompts'][c]
                records.append(dict(task_id=t['task_id'],family_id=t['family_id'],condition=c+'__'+v,rule=rule,variant=v,prompt=prompt,prompt_tests=new_tests,prompt_format='chat',thinking='disabled',source_condition=c,reference_equivalence_verified=True))
    assert len(records)==480
    out=ROOT/'outputs/semantic_surface';out.mkdir(parents=True,exist_ok=True)
    for m in cfg['models']:
        dest=out/m;dest.mkdir(exist_ok=True)
        (dest/'prompts.jsonl').write_text(''.join(json.dumps(dict(r,model_path=paths[m]),ensure_ascii=False)+'\n' for r in records))
    (out/'prompt_validation.json').write_text(json.dumps(dict(tasks=120,families=20,variants=cfg['variants'],prompts_per_model=480,models=cfg['models'],parentheses_ast_identical=True,reference_verdicts_preserved=True,order_audit='All 20 families use per-call reference functions with literal inputs and no persistent cross-call state.'),indent=2)+'\n')
    print(out)
if __name__=='__main__':main()
