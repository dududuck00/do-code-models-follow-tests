"""Export the six observed historical LCB configurations, including Qwen3.6 high3."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tddexp.statistics import paired_effect
RUNS=[('Qwen2.5-Coder-7B','Original','livecodebench_v6_minus_v5_qwen_full_fixed'),('Qwen2.5-Coder-7B','Synthetic high3','livecodebench_v6_new_qwen_synthetic_high3'),('Qwen2.5-Coder-7B','Synthetic high5','livecodebench_v6_new_qwen_synthetic_high5'),('Qwen3.6-27B','Original','livecodebench_v6_new_qwen36_27b_nothink'),('Qwen3.6-27B','Synthetic high3','livecodebench_v6_new_qwen36_27b_synth_high3_nothink'),('Qwen3.6-27B','Synthetic high5','livecodebench_v6_new_qwen36_27b_synth_high5_nothink')]
CONDITIONS=['nl_only','nl_tests','shuffled_tests','irrelevant_tests']
def main():
    results=[]
    for model,source,folder in RUNS:
        rows=[];counts={};tasksets=[]
        for c in CONDITIONS:
            group=list(map(json.loads,(ROOT/'outputs'/folder/'eval'/f'{c}_eval_all.jsonl').open()))
            assert len(group)==175 and len({r['task_id'] for r in group})==175
            rows.extend(group);counts[c]=sum(r['passed'] for r in group);tasksets.append({r['task_id'] for r in group})
        assert all(t==tasksets[0] for t in tasksets)
        effects={c:paired_effect(rows,'nl_only',c) for c in CONDITIONS[1:]}
        results.append(dict(model=model,test_source=source,output_dir='outputs/'+folder,tasks=175,counts=counts,effects_vs_nl=effects))
    dest=ROOT/'outputs/analysis';dest.mkdir(exist_ok=True)
    (dest/'historical_lcb_results.json').write_text(json.dumps({'complete':True,'runs':results},ensure_ascii=False,indent=2)+'\n')
    lines=['# LiveCodeBench 合成测试实验','', '每行覆盖 175 题、四个条件、一次生成。Qwen3.6 high3 为本次补充，其余五行为历史结果。各条件与本行 NL-only 配对比较。','', '| 模型 | 测试来源 | NL-only | 相关测试 | 打乱输出 | 无关测试 |','|---|---|---:|---:|---:|---:|']
    for r in results:lines.append('| '+r['model']+' | '+r['test_source']+' | '+' | '.join(str(r['counts'][c]) for c in CONDITIONS)+' |')
    new=next(r for r in results if r['model']=='Qwen3.6-27B' and r['test_source']=='Synthetic high3')
    lines += ['', 'Qwen3.6 high3 相对本次 NL-only 的配对结果：', '', '| 条件 | 差值（百分点） | 95% 区间 | 获益题数 | 受损题数 | 精确 McNemar p |', '|---|---:|---|---:|---:|---:|']
    for condition,label in [('nl_tests','相关测试'),('shuffled_tests','打乱输出'),('irrelevant_tests','无关测试')]:
        effect=new['effects_vs_nl'][condition];rep=next(iter(effect['per_repeat'].values()));lo,hi=effect['ci95']
        lines.append(f"| {label} | {effect['paired_difference']*100:+.1f} | [{lo*100:.1f}, {hi*100:.1f}] | {rep['gains']} | {rep['losses']} | {rep['exact_mcnemar_p']:.3f} |")
    lines+=['','新 high3 行的完整配对差值、任务 bootstrap 区间与精确 McNemar 检验见 `historical_lcb_results.json`。','', '本次冻结复用 Qwen2.5 high3 历史运行的 700 条提示，保持测试内容、打乱输出和无关测试分配一致。采用 Qwen3.6、BF16、关闭 thinking、贪心解码、8192 输出上限、单 GPU 副本。151 题有三条测试，18 题有两条，6 题有一条。','', '评测使用 `evaluate_lcb_subset.py` 和官方公开及私有测试，采用该历史包装脚本默认的 10 秒超时，通过管道连接的子进程调用官方 `run_test`。历史运行的具体超时命令未保存。本次使用一个推理副本，历史 high5 使用多个独立副本；两行各自报告同次运行的 NL-only 基线。','', '合成测试的期望输出未经过参考实现验证，high3/high5 表示历史选择配置。']
    (dest/'historical_lcb_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(new,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
