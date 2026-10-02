"""Integrate completed Qwen3.6 high3 statistics into the paper and project state."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
    results=json.loads((ROOT/'outputs/analysis/historical_lcb_results.json').read_text());assert results['complete']
    rows=results['runs'];new=next(r for r in rows if r['model']=='Qwen3.6-27B' and r['test_source']=='Synthetic high3')
    import yaml
    config=yaml.safe_load((ROOT/'configs/historical_lcb_high3.yaml').read_text())
    protocol_path=ROOT/config['output_dir']/'protocol.json'
    protocol=json.loads(protocol_path.read_text());assert protocol['complete']
    protocol['evaluation']=config['evaluation'];protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    c=new['counts'];e=new['effects_vs_nl']['nl_tests'];repeat=e['per_repeat']['0'];net=c['nl_tests']-c['nl_only'];ci=e['ci95'];pval=repeat['exact_mcnemar_p']
    paper=ROOT/'aaai/AuthorKit27/AuthorKit27/paper.tex';s=paper.read_text()
    start=s.index('Model & Test source & NL-only & Relevant & Shuffled & Irrelevant')
    end=s.index('\\bottomrule',start)
    table=['Model & Test source & NL-only & Relevant & Shuffled & Irrelevant \\\\',r'\midrule']
    for row in rows:
        table.append(row['model']+' & '+row['test_source']+' & '+' & '.join(str(row['counts'][k]) for k in ['nl_only','nl_tests','shuffled_tests','irrelevant_tests'])+r' \\')
    s=s[:start]+'\n'.join(table)+'\n'+s[end:]
    marker='The two Qwen3.6 NL-only runs use identical prompts but have only 44 identical completion strings, motivating the repeated-run protocol in the main experiment.'
    replacement='The original-test and high5 Qwen3.6 runs use identical NL-only prompts but have only 44 identical completion strings, motivating the repeated-run protocol in the main experiment.'
    if marker in s:s=s.replace(marker,replacement)
    begin='% BEGIN QWEN36 HIGH3 COMPLETION';finish='% END QWEN36 HIGH3 COMPLETION'
    text=begin+'\n'+(
        f'The Qwen3.6 high3 run solves {c["nl_only"]} tasks under NL-only, {c["nl_tests"]} with relevant tests, {c["shuffled_tests"]} with shuffled outputs, and {c["irrelevant_tests"]} with irrelevant tests.\n'
        f'Relevant-minus-NL is ${net:+d}$ tasks ({e["paired_difference"]*100:+.1f} percentage points, 95\\% task-bootstrap interval [{ci[0]*100:.1f}, {ci[1]*100:.1f}]; exact McNemar $p={pval:.3f}$).\n'
        'This run reuses all four prompt conditions from the historical Qwen2.5 high3 run and the archived code-extraction routine, with BF16, non-thinking greedy decoding, an 8192-token output budget, and one GPU replica.\n'
        'The high3 suites contain three tests for 151 tasks, two for 18, and one for six; evaluation uses the official public and private tests with the historical wrapper\'s 10-second default.\n'
    )+finish
    if begin in s:
        a=s.index(begin);b=s.index(finish,a)+len(finish);s=s[:a]+text+s[b:]
    else:s=s.replace(replacement,replacement+'\n\n'+text)
    paper.write_text(s)
    state=ROOT/'PROJECT_STATE.md';state_text=state.read_text();title='## Qwen3.6 historical high3 completion'
    section=f'''{title}

The previously missing high3 row is complete: 175 tasks, four conditions, one run, 700 generations and evaluations. It remains separate from the 67,650 controlled observations.

Solved counts (NL-only / relevant / shuffled / irrelevant): **{c['nl_only']} / {c['nl_tests']} / {c['shuffled_tests']} / {c['irrelevant_tests']}**. Relevant − NL is {net:+d} tasks, {e['paired_difference']*100:+.1f} percentage points [{ci[0]*100:.1f}, {ci[1]*100:.1f}], exact McNemar p={pval:.3f}.

The frozen prompts reproduce the Qwen2.5 high3 conditions; the archived extraction routine reproduces all 700 historical Qwen3.6 high5 candidates. Evaluation retains the historical wrapper's default timeout of 10 seconds; historical command-line overrides are not recorded. Run configuration: `configs/historical_lcb_high3.yaml`. Results: `outputs/analysis/historical_lcb_report.md` and `historical_lcb_results.json`. The six-row appendix table includes the new run.
'''
    if title in state_text:state_text=state_text[:state_text.index(title)].rstrip()+'\n'
    state.write_text(state_text+'\n'+section)
    print(json.dumps({'counts':c,'effect':e},indent=2))
if __name__=='__main__':main()
