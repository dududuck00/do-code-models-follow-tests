"""Export the fixed surface-control contrasts to Markdown and LaTeX."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES={'qwen25':'Qwen2.5-Coder-7B','qwen36':'Qwen3.6-27B','qwen35':'Qwen3.5-9B','qwen38':'Qwen3.8-27B','deepseek':'DeepSeek-Coder-6.7B'}
VARIANTS={'parenthesized':'括号等价写法','reverse_order':'断言倒序'}
def main():
    base=ROOT/'outputs/semantic_surface';r=json.loads((base/'analysis.json').read_text());assert r['complete']
    lines=['# 同规则不同表述对照','', '五个模型沿用原 120 个实例、20 个规则家族。新增两种表述，每种包含 A/B 两个规则：共 2,400 次前向状态提取，复用原 1,200 份状态，没有重新生成代码。', '',
    '括号变体在整个比较表达式外增加括号，AST 与原断言相同。倒序变体将三个断言倒序并重新编号。现有参考实现均按单次调用处理字面量输入，没有跨调用状态；两种参考实现对全部变体的测试判定均与原断言一致。描述、接口和可见输入集合保持不变。', '',
    '对每个实例、变体和层，异规则距离是原表述 A/B 距离与变体 A/B 距离的平均；同规则距离是 A 原表述/变体距离与 B 原表述/变体距离的平均。主差值为异规则减同规则。先在实例内配对，再按 20 个家族 bootstrap 5,000 次（seed=0）。两种变体分别报告，等权平均作为次要结果。最终层仍是固定汇报层，所有层均保留。', '',
    '这个差值衡量规则输出改动相对于指定表述改动的距离差，不是表示中“语义信息占比”的分解。', '']
    lines += ['Qwen3.6 和 Qwen3.8 在两种表述、两种距离下的差值区间均为正。其余模型的区间模式随表述或距离指标变化；完整数值分别列于下表。', '']
    table=[]; l2_table=[]
    for metric,title in [('cosine_distance','最终层余弦距离'),('relative_l2','最终层相对 L2')]:
        lines += ['## '+title,'','| 模型 | 变体 | 异规则距离 | 同规则距离 | 差值 [95% CI] |','|---|---|---:|---:|---|']
        for m,model in r['models'].items():
            for v,stats in model['variants'].items():
                z=stats[metric]['final_layer'];lo,hi=z['ci95'];lines.append(f"| {NAMES[m]} | {VARIANTS[v]} | {z['different_rule_mean']:.6g} | {z['same_rule_mean']:.6g} | {z['difference']:.6g} [{lo:.6g}, {hi:.6g}] |")
                if metric=='cosine_distance':
                    # A common multiplier keeps the table compact without mixing model scales.
                    table.append(NAMES[m]+' & '+('Parentheses' if v=='parenthesized' else 'Reordered')+f" & {1000*z['different_rule_mean']:.3f} & {1000*z['same_rule_mean']:.3f} & {1000*z['difference']:.3f} [{1000*lo:.3f}, {1000*hi:.3f}]"+r' \\')
                else:
                    l2_table.append(NAMES[m]+' & '+('Parentheses' if v=='parenthesized' else 'Reordered')+f" & {z['different_rule_mean']:.3f} & {z['same_rule_mean']:.3f} & {z['difference']:.3f} [{lo:.3f}, {hi:.3f}]"+r' \\')
    lines += ['', '## 文件', '', '- `analysis.json`：全部层、两种距离及变体平均结果。','- `pair_layer_metrics.csv`：逐实例、逐变体、逐层数据。','- `prompt_validation.json`：等价表述验证。','- 各模型 `prompts.jsonl`、`replay/generations.jsonl` 和 `replay/states/`：提示及原始状态。']
    (base/'report.md').write_text('\n'.join(lines)+'\n')
    paper=ROOT/'paper/semantic_surface';paper.mkdir(parents=True,exist_ok=True)
    (paper/'results.tex').write_text('\\begin{table}[htbp]\n\\centering\\small\n\\begin{tabular}{llrrl}\n\\toprule\nModel & Surface variant & Different rule & Same rule & Difference [95\\% CI] \\\\\n\\midrule\n'+'\n'.join(table)+'\n\\bottomrule\n\\end{tabular}\n\\caption{Final-layer cosine distances, multiplied by $10^3$. Differences pair rule and surface contrasts within each task. Intervals resample specification families.}\n\\label{tab:semantic-surface}\n\\end{table}\n')
    l2=(paper/'results.tex').read_text().replace('\n'.join(table),'\n'.join(l2_table)).replace('Final-layer cosine distances, multiplied by $10^3$.','Final-layer relative Euclidean distances.').replace('tab:semantic-surface','tab:semantic-surface-l2')
    (paper/'results_l2.tex').write_text(l2)
    def names(models):
        if len(models)==5:return 'all five models'
        if set(models)=={'qwen25','qwen36','qwen35','qwen38'}:return 'the four Qwen models'
        return ', '.join(NAMES[m] for m in models)
    findings=[]
    for metric,metric_label in [('cosine_distance','Cosine distance'),('relative_l2','Relative Euclidean distance')]:
        sentences=[]
        for v,label in [('parenthesized','parentheses'),('reverse_order','reversed assertion order')]:
            positive=[];negative=[];overlap=[]
            for m,model in r['models'].items():
                lo,hi=model['variants'][v][metric]['final_layer']['ci95']
                (positive if lo>0 else negative if hi<0 else overlap).append(m)
            clauses=[]
            if positive:clauses.append('the between-rule distance exceeds the within-rule surface distance with positive difference intervals for '+names(positive))
            if negative:clauses.append('the difference intervals are negative for '+names(negative))
            if overlap:clauses.append('the difference interval includes zero for '+names(overlap))
            sentences.append('With '+label+', '+clauses[0]+'.'+''.join(' '+c[0].upper()+c[1:]+'.' for c in clauses[1:]))
        findings.append('\\paragraph{'+metric_label+'.} '+ ' '.join(sentences))
    findings.append('The contrast depends on the checkpoint, surface transformation, and distance metric. Qwen3.6 and Qwen3.8 show positive difference intervals for both transformations under both metrics. These comparisons quantify relative sensitivity to the specified rule-output and surface changes.')
    (paper/'findings.tex').write_text('\n\n'.join(findings)+'\n')
    print(base/'report.md')
if __name__=='__main__':main()
