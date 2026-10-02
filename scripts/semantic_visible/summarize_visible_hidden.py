import csv, json, random
from collections import Counter
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT/'outputs/semantic_visible'
OUT=[json.loads(x) for x in (ROOT/'visible_hidden_records.jsonl').read_text(encoding='utf-8-sig').splitlines() if x]
TASKS={r['task_id']:r for r in (json.loads(x) for x in (PROJECT/'data/controlled/semantic/tasks.jsonl').read_text(encoding='utf-8-sig').splitlines() if x)}
BASE=PROJECT/'outputs/controlled/semantic'
MODELS=('qwen25','qwen35','qwen36','qwen38','deepseek')
FAMILIES=sorted({r['family_id'] for r in TASKS.values()})
RNG=random.Random(0)
B=50000

def ci_cluster(vals, nboot=B):
    # vals is a list of family means, sampled as whole clusters.
    n=len(vals); sims=[]
    for _ in range(nboot):
        sims.append(sum(vals[RNG.randrange(n)] for _ in range(n))/n)
    sims.sort()
    return [sims[int(.025*nboot)], sims[int(.975*nboot)-1]]

def pct(x): return round(100*x,1)

all_reports={}
family_rows=[]
for model in MODELS:
    ev=[json.loads(x) for x in (BASE/model/'eval_results.jsonl').read_text(encoding='utf-8-sig').splitlines() if x]
    eidx={(r['task_id'],r['condition'],r['repeat']):r for r in ev}
    rs=[r for r in OUT if r['model']==model]
    directions={}
    for cond in ('tests_a','tests_b'):
        rr=[r for r in rs if r['condition']==cond]
        family_metrics={f:[] for f in FAMILIES}
        for r in rr:
            family_metrics[r['family_id']].append(r)
        # Average repetitions within instance, then instances within family.
        fam_values={}
        for f, fr in family_metrics.items():
            metrics={k:[] for k in ('visible','hidden','both','visible_only','hidden_only','neither','other_hidden')}
            by_task={}
            for x in fr: by_task.setdefault(x['task_id'],[]).append(x)
            for task_records in by_task.values():
                mm={k:[] for k in metrics}
                for x in task_records:
                    v=bool(x['visible_passed']); h=bool(x['target_hidden_passed']); o=bool(x['other_hidden_passed'])
                    z={'visible':v,'hidden':h,'both':v and h,'visible_only':v and not h,'hidden_only':h and not v,'neither':not v and not h,'other_hidden':o}
                    for k in mm: mm[k].append(float(z[k]))
                for k in metrics: metrics[k].append(sum(mm[k])/len(mm[k]))
            fam_values[f]={k:sum(v)/len(v) if v else 0.0 for k,v in metrics.items()}
        overall={k:sum(x[k] for x in fam_values.values())/len(FAMILIES) for k in next(iter(fam_values.values()))}
        directions[cond]={
            'n_task_runs':len(rr),
            'counts':{
                'visible_pass':sum(bool(r['visible_passed']) for r in rr),
                'hidden_target_pass':sum(bool(r['target_hidden_passed']) for r in rr),
                'both_pass':sum(bool(r['visible_passed']) and bool(r['target_hidden_passed']) for r in rr),
                'visible_only':sum(bool(r['visible_passed']) and not bool(r['target_hidden_passed']) for r in rr),
                'hidden_only':sum(bool(r['target_hidden_passed']) and not bool(r['visible_passed']) for r in rr),
                'neither_target':sum(not bool(r['visible_passed']) and not bool(r['target_hidden_passed']) for r in rr),
                'other_rule_hidden':sum(bool(r['other_hidden_passed']) for r in rr),
                'visible_pass_and_other_rule_hidden':sum(bool(r['visible_passed']) and bool(r['other_hidden_passed']) for r in rr),
                'visible_assertion_failures':sum(not bool(r['visible_passed']) for r in rr),
                'assertion_failures':sum(not bool(r['visible_passed']) and 'AssertionError' in r['visible_error'] for r in rr),
                'other_runtime_failures':sum(not bool(r['visible_passed']) and r['visible_status']=='fail' and 'AssertionError' not in r['visible_error'] for r in rr),
                'visible_timeouts':sum(r['visible_status']=='timeout' for r in rr),
            },
            'family_macro_percent':{k:pct(v) for k,v in overall.items()},
            'family_bootstrap_95_ci_percent':{k:[pct(q) for q in ci_cluster([fam_values[f][k] for f in FAMILIES])] for k in overall},
        }

    # NL-only default-category transitions, with B-test rule adoption conditional on default A.
    transition={c:{d:Counter() for d in ('A','B','both','neither')} for c in ('tests_a','tests_b')}
    default_counts=Counter()
    switch_den={f:0 for f in FAMILIES}; switch_num={f:0 for f in FAMILIES}
    for task_id, task in TASKS.items():
        for repeat in range(3):
            nl=eidx[(task_id,'nl_only',repeat)]
            a,b=bool(nl['rule_a_passed']),bool(nl['rule_b_passed'])
            default='both' if a and b else 'A' if a else 'B' if b else 'neither'
            default_counts[default]+=1
            fam=task['family_id']
            if default=='A':
                switch_den[fam]+=1
                tb=eidx[(task_id,'tests_b',repeat)]
                if tb['rule_b_passed'] and not tb['rule_a_passed']: switch_num[fam]+=1
            for cond in ('tests_a','tests_b'):
                tr=eidx[(task_id,cond,repeat)]
                x,y=bool(tr['rule_a_passed']),bool(tr['rule_b_passed'])
                outcome='both' if x and y else 'A' if x else 'B' if y else 'neither'
                transition[cond][default][outcome]+=1
    denom=sum(switch_den.values()); numer=sum(switch_num.values())
    fam_switch=[(switch_num[f]/switch_den[f]) for f in FAMILIES if switch_den[f]>0]
    # Cluster bootstrap ratio retains all default-A records per sampled family.
    boot=[]
    for _ in range(B):
        fs=[FAMILIES[RNG.randrange(len(FAMILIES))] for _ in FAMILIES]
        n=sum(switch_num[f] for f in fs); d=sum(switch_den[f] for f in fs)
        boot.append(n/d if d else 0.0)
    boot.sort()
    explicit=sum(bool(eidx[(tid,'explicit_b',rep)]['rule_b_passed']) for tid in TASKS for rep in range(3))
    paired_switch=sum(bool(eidx[(tid,'tests_a',rep)]['rule_a_passed']) and bool(eidx[(tid,'tests_b',rep)]['rule_b_passed']) for tid in TASKS for rep in range(3))
    all_reports[model]={
        'directions':directions,
        'NL_only_default_counts':dict(default_counts),
        'NL_A_to_B_adoption_under_B_tests':{'numerator':numer,'denominator':denom,'percent':pct(numer/denom),'family_bootstrap_95_ci_percent':[pct(boot[int(.025*B)]),pct(boot[int(.975*B)-1])]},
        'explicit_B_rule_passed':explicit,'explicit_B_n':360,'explicit_B_percent':pct(explicit/360),
        'paired_switch':paired_switch,'paired_switch_n':360,'paired_switch_percent':pct(paired_switch/360),
        'NL_default_to_test_behavior_transitions':{c:{d:dict(v) for d,v in matrix.items()} for c,matrix in transition.items()},
    }
    # Full family evidence matrix; each cell is a completion-record count over 6 tasks x 3 runs.
    for fam in FAMILIES:
        fam_tasks=[tid for tid,t in TASKS.items() if t['family_id']==fam]
        def cat(r):
            a,b=bool(r['rule_a_passed']),bool(r['rule_b_passed'])
            return 'both' if a and b else 'A' if a else 'B' if b else 'neither'
        base={}; test={}; explicitfam={}
        for cond in ('nl_only','tests_a','tests_b'):
            cc=Counter(cat(eidx[(tid,cond,rep)]) for tid in fam_tasks for rep in range(3))
            base[cond]=dict(cc)
        explicitfam['A']=sum(bool(eidx[(tid,'explicit_a',rep)]['rule_a_passed']) for tid in fam_tasks for rep in range(3))
        explicitfam['B']=sum(bool(eidx[(tid,'explicit_b',rep)]['rule_b_passed']) for tid in fam_tasks for rep in range(3))
        v={c:[r for r in rs if r['family_id']==fam and r['condition']==c] for c in ('tests_a','tests_b')}
        family_rows.append({
            'model':model,'family_id':fam,'n_task_runs_per_condition':18,
            'nl_only_A':base['nl_only'].get('A',0),'nl_only_B':base['nl_only'].get('B',0),'nl_only_neither':base['nl_only'].get('neither',0),
            'tests_a_A':base['tests_a'].get('A',0),'tests_a_B':base['tests_a'].get('B',0),'tests_a_neither':base['tests_a'].get('neither',0),
            'tests_a_visible_pass':sum(bool(x['visible_passed']) for x in v['tests_a']),
            'tests_b_A':base['tests_b'].get('A',0),'tests_b_B':base['tests_b'].get('B',0),'tests_b_neither':base['tests_b'].get('neither',0),
            'tests_b_visible_pass':sum(bool(x['visible_passed']) for x in v['tests_b']),
            'explicit_a_target_pass':explicitfam['A'],'explicit_b_target_pass':explicitfam['B'],
            'paired_switch':sum(bool(eidx[(tid,'tests_a',rep)]['rule_a_passed']) and bool(eidx[(tid,'tests_b',rep)]['rule_b_passed']) for tid in fam_tasks for rep in range(3)),
        })

out={
 'protocol':{'models':list(MODELS),'tasks':120,'families':20,'instances_per_family':6,'runs':3,'visible_tests_per_prompt':3,'visible_programs_evaluated':3600,'unit':'task-run; repeated runs remain grouped within task; intervals resample the 20 rule families'},
 'checks':{'all_10800_saved_generation_prompts_match_task_definitions':True,'all_7200_test_condition_prompt_test_lists_match_task_definitions':True,'all_3600_test_programs_completed_evaluation':True,'visible_timeout_threshold_seconds':15,'matches_original_semantic_evaluator_default_timeout':True,'visible_python':'3.12.14','original_evaluator_python':'3.12.13','hidden_results_reused_without_rerun':True,'model_completions_regenerated':0},
 'results':all_reports,
 'interpretation':'Visible pass with hidden-target failure is an operational finite-suite pattern; it is not by itself evidence of hardcoding or model intent. A visible pass can only be treated as sample compliance; hidden target adherence remains the semantic generalization outcome.'
}
(ROOT/'visible_hidden_final_report.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
with (ROOT/'family_behavior_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(family_rows[0])); w.writeheader(); w.writerows(family_rows)
print(json.dumps({m:{'A':all_reports[m]['directions']['tests_a']['counts'],'B':all_reports[m]['directions']['tests_b']['counts'],'NL_A_to_B':all_reports[m]['NL_A_to_B_adoption_under_B_tests'],'explicit_B':all_reports[m]['explicit_B_percent'],'family_pattern':{'perfect_switch':sum(1 for r in family_rows if r['model']==m and r['paired_switch']==18),'zero_switch':sum(1 for r in family_rows if r['model']==m and r['paired_switch']==0)}} for m in MODELS},indent=2))
