"""Render checked representation--behavior results as a Chinese report and LaTeX supplement."""
from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[1]
MODELS=['qwen25','qwen36','qwen35','qwen38','deepseek']
NAMES={'qwen25':'Qwen2.5-Coder-7B','qwen36':'Qwen3.6-27B','qwen35':'Qwen3.5-9B','qwen38':'Qwen3.8-27B','deepseek':'DeepSeek-Coder-6.7B'}
def fmt(x):return 'NA' if x is None else f'{x:.3f}'
def estimate(e):return f"{fmt(e['estimate'])} [{fmt(e['ci95'][0])}, {fmt(e['ci95'][1])}]"
def main():
    base=ROOT/'outputs/semantic_representation';a=json.loads((base/'representation_analysis.json').read_text());p=json.loads((base/'family_probe.json').read_text());labels=json.loads((base/'label_summary.json').read_text())
    assert a['complete'] and p['complete']
    lines=['# 成对语义提示的表示—行为分析','',
      '已完成五个模型的 1,200 次唯一提示前向提取，关联 3,600 条已有行为记录（120 个实例 × A/B × 3 次 × 5 模型）。每个模型以 120 个实例为分析单位，以 20 个规则家族为不确定性估计与泛化评测单位。','',
      '四个 Qwen 模型的表示距离在后部层达到最大中位数。最终层变化幅度与切换率的关系随家族和长度调整而变化；控制提示长度后，四个 Qwen 模型的区间均跨零。其跨规则家族线性探针的 MAE 均高于训练家族中位数基线，MSE 也均高于训练家族均值基线。','',
      '## 逐层变化与行为对应','',
      '固定描述、可见输入和接口，仅改变三个可见断言的期望输出。每个模型重放 240 个唯一提示，保存提示末端各层隐藏状态。Transformers 使用 BF16 和关闭 thinking 的聊天格式，重放 token 序列逐条匹配已有 vLLM 生成记录。三次行为结果关联同一份提示表示，并在实例内平均。','',
      '几何量为余弦距离，以及以两向量平均范数归一化的差向量 L2 范数。最终层是预先指定的汇总位置；所有层的曲线和数据同时保留。','',
      '| 模型 | 最大中位距离所在层 | 最终层总体相关 | 家族内相关 | 家族内、长度调整后 |',
      '|---|---|---|---|---|']
    table_rows=[]
    for m in MODELS:
        r=a['models'][m];peak=max(r['layers'],key=lambda x:x['median_cosine_distance']);s=r['final_layer']['associations']['cosine_distance']['paired_switch']
        vals=[estimate(s[k]) for k in ['pooled_spearman','within_family_spearman','within_family_length_adjusted_correlation']]
        lines.append(f"| {NAMES[m]} | {peak['layer']}/{r['layers_including_embedding']-1} | "+' | '.join(vals)+' |')
        table_rows.append(NAMES[m]+' & '+' & '.join(vals)+r' \\')
    lines += ['', 'DeepSeek 作为另一模型家族的验证，完整沿用既定协议。它与 Qwen2.5 的平均成对切换率同为 11.1%，但中位余弦距离在第 17/32 层达到峰值，Qwen2.5 则在第 28/28 层达到峰值。DeepSeek 最终层长度调整后的相关为 0.100 [−0.080, 0.282]，相对 L2 为 0.080 [−0.131, 0.278]。行为—表示关联的区间模式相同，逐层几何轨迹有所不同。', '', 'DeepSeek 的表示探针 MAE 为 0.225，中位数基线为 0.111，差值为 +0.114 [0.055, 0.196]；MSE 为 0.138，均值基线为 0.105。该检查点同样未显示跨规则家族预测优势。', '', '新增实验前已核验原四模型的原始记录；加入 DeepSeek 后，原四模型逐层分析、探针结果及家族划分复算后完全一致。审计记录为 `deepseek_extension_audit.json`。']
    lines+=['','数值为秩相关及 95% 家族 bootstrap 区间。总体相关采用 Spearman 相关。家族内分析先在各家族的六个实例中取秩并中心化；长度调整进一步残差化平均提示 token 数和 A/B token 数绝对差。重复采样 5,000 次；Qwen2.5 有 13 次采样未包含可变的切换结果，相关未定义，区间使用其余 4,987 次。DeepSeek 的切换相关区间使用 4,996 个有定义的采样。逐层区间是逐点区间。相对 L2 得到相近的最终层关系，完整数据在 JSON 中。','',
      '![逐层几何变化](figures/layer_geometry.png)','',
      '四个 Qwen 面板各展示全部 120 对提示的中位数和四分位区间，纵轴按模型分别设置。绝对距离不用于模型理解能力排名。','',
      '![逐层表示—行为关联](figures/layer_behavior_association.png)','',
      '## 未见规则家族上的预测','',
      '预测目标是每个实例三次运行的平均成对切换率。外层五折每次保留四个完整家族；内层四折仅使用训练家族选择层与正则系数。表示特征为归一化的 A/B 差向量，候选层位于 25%、50%、75% 和 100% 深度，正则系数为 0.0001、0.01、1、100。训练核中心化和尺度归一化只使用训练家族。预测裁剪至 [0,1]。','',
      '文本基线对 A/B 期望输出使用字符 2–4 gram TF-IDF，词表和 IDF 在训练家族上拟合，同样通过内层验证选择正则系数。MAE 的常数基线使用训练家族中位数，MSE 的常数基线使用训练家族均值。','',
      '| 模型 | 中位数 MAE | 输出文本 MAE | 表示 MAE | 表示 − 中位数 [95% 区间] |',
      '|---|---:|---:|---:|---|']
    probe_rows=[];mse_rows=[]
    for m in MODELS:
        r=p['models'][m];v=r['metrics'];e=r['contrasts']['representation_minus_median'];delta=f"{fmt(e['mae_difference'])} [{fmt(e['ci95'][0])}, {fmt(e['ci95'][1])}]"
        lines.append(f"| {NAMES[m]} | {fmt(v['median']['mae'])} | {fmt(v['output_text']['mae'])} | {fmt(v['representation']['mae'])} | {delta} |")
        probe_rows.append(f"{NAMES[m]} & {fmt(v['median']['mae'])} & {fmt(v['output_text']['mae'])} & {fmt(v['representation']['mae'])} & {delta}"+r' \\')
        mse_rows.append(f"| {NAMES[m]} | {fmt(v['mean']['mse'])} | {fmt(v['output_text']['mse'])} | {fmt(v['representation']['mse'])} |")
    lines+=['','误差越低越好；正差表示探针误差高于常数基线。区间对外层保留家族的预测误差进行 bootstrap，不在每次 bootstrap 中重新训练模型。结果针对这一线性探针和四个候选深度。','', '| 模型 | 均值 MSE | 输出文本 MSE | 表示 MSE |','|---|---:|---:|---:|',*mse_rows,'',
      '## 目标规则、另一规则与未满足两者','',
      '| 模型 | 目标规则 | 另一规则 | 两者均未通过 |','|---|---:|---:|---:|']
    behavior_rows=[]
    for m in MODELS:
        c=labels['models'][m]['outcomes'];vals=[c[k] for k in ['target_rule','alternative_rule','neither_rule']]
        lines.append('| '+NAMES[m]+' | '+' | '.join(str(v) for v in vals)+' |');behavior_rows.append(NAMES[m]+' & '+' & '.join(str(v) for v in vals)+r' \\')
    lines+=['','每行 720 条行为记录。这些记录关联 240 个提示表示，不能作为 720 个独立表示样本。','',
      f'对 {sum(v["outcomes"]["neither_rule"] for v in labels["models"].values())} 条“两者均未通过”记录涉及的 {labels["unique_diagnostic_programs"]} 个不同任务—程序组合重放逐测试诊断，原行为类别全部保持一致：','',
      '| 模型 | 初始化/运行错误或超时 | 共同输入不匹配 | 其余两规则输出不匹配 |','|---|---:|---:|---:|']
    for m in MODELS:
        d=labels['models'][m]['diagnostics'];runtime=sum(v for k,v in d.items() if k not in ['common_case_mismatch','neither_rule_output_mismatch'])
        lines.append(f"| {NAMES[m]} | {runtime} | {d.get('common_case_mismatch',0)} | {d.get('neither_rule_output_mismatch',0)} |")
    lines+=['','共同输入不匹配表示基本功能在共享测试上未通过；其余输出不匹配保留为未满足两规则，未推断程序意图。没有共同输入的实例记录为共同功能未观测。','',
      '## 文件与复现','',
      '- 配置：`configs/semantic_representation.yaml`。','- 唯一提示与状态：`outputs/semantic_representation/{model}/`。','- 行为与错误诊断：`behavior_labels.jsonl`、`paired_behavior.jsonl`、`label_summary.json`。','- 逐层与逐实例数据：`representation_analysis.json`、`layer_metrics.csv`、`pair_layer_metrics.csv`。','- 家族划分与外层预测：`family_folds.json`、`family_probe.json`、`family_probe_predictions.jsonl`。','- 可插入的论文章节及独立 PDF：`paper/semantic_representation/`。','',
      '当前工作区未找到 FSE 主文件，因此提供独立章节和图表；未修改现有 AAAI 稿。']
    (base/'report.md').write_text('\n'.join(lines)+'\n')
    paper=ROOT/'paper/semantic_representation';(paper/'tables').mkdir(parents=True,exist_ok=True);(paper/'figures').mkdir(exist_ok=True)
    def table(name,cols,header,rows,caption,label):
        text='\\begin{table}[htbp]\n\\centering\n\\small\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+header+r' \\'+'\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{table}\n'
        (paper/'tables'/name).write_text(text)
    table('final_association.tex','lccc','Model & Pooled & Within family & Length adjusted',table_rows,'Final-layer cosine-distance association with mean paired switching. Entries are rank correlations with 95\\% family-bootstrap intervals.','tab:semantic-rep-association')
    table('family_probe.tex','lrrrl','Model & Median & Output text & Representation & Rep. $-$ median',probe_rows,'Mean absolute error on unseen specification families. The final column gives paired MAE differences with 95\\% family-bootstrap intervals. Lower error is better.','tab:semantic-rep-probe')
    table('behavior.tex','lrrr','Model & Target rule & Alternative rule & Neither rule',behavior_rows,'Behavioral outcomes across the two test conditions and three runs, totaling 720 observations per model.','tab:semantic-rep-behavior')
    for name in ['layer_geometry','layer_behavior_association']:
        shutil.copy2(base/f'figures/{name}.pdf',paper/f'figures/{name}.pdf')
    (paper/'main.tex').write_text(r'''\newcommand{\DoNotLoadEpstopdf}{}
\documentclass[10pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{booktabs,graphicx,amsmath}
\setlength{\emergencystretch}{2em}
\title{Paired Representations and Rule-Following Behavior}
\author{}
\date{}
\begin{document}
\maketitle
\input{representation_section}
\end{document}
''')
    (paper/'README.md').write_text('''# Paired representation analysis: LaTeX sources

Compile `main.tex` with pdfLaTeX, or run `bash scripts/build_semantic_representation.sh` from the project root.

To integrate into an existing manuscript, copy this directory into the manuscript project, load `booktabs`, `graphicx`, and `amsmath`, define `\\newcommand{\\SemanticRepDir}{semantic_representation}`, then use `\\input{semantic_representation/representation_section}`. Adjust `figure`/`table` to starred floats if needed for a two-column layout. The standalone PDF uses a generic article layout, not an FSE submission template.

All numbers are exported by `scripts/write_semantic_representation_report.py`. The original FSE main file was not available in the current workspace.
''')
    print(base/'report.md')
if __name__=='__main__':main()
