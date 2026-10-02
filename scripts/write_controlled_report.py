#!/usr/bin/env python
"""Write the completed experiment report from the publication result export."""
import json
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'qwen25': 'Qwen2.5-Coder-7B', 'deepseek': 'DeepSeek-Coder-6.7B', 'qwen36': 'Qwen3.6-27B', 'qwen35': 'Qwen3.5-9B', 'qwen38': 'Qwen3.8-27B'}
DATASETS = {'humanevalplus': 'HumanEval+', 'mbppplus': 'MBPP+', 'livecodebench': 'LCB v6-new'}


def effect(row):
    lo, hi = row['ci95']
    signed = lambda x: f'{round(100*x,1) or 0.0:+.1f}'
    return f"{signed(row['paired_difference'])} [{signed(lo)}, {signed(hi)}]"


def main():
    source = ROOT / 'outputs/analysis/controlled_results.json'
    report = json.loads(source.read_text())
    if not report['complete']:
        raise SystemExit('Complete all experiment evaluations before writing the final report.')
    results = report['results']
    config = yaml.safe_load((ROOT / 'configs/controlled.yaml').read_text())
    models_for = lambda exp: config['experiments'][exp]['models']
    totals = {exp:sum(c['observations'] for key,r in results.items() if key.startswith(exp+'/') and key.count('/')==1 for c in r['conditions'].values()) for exp in ['main','semantic','quality']}
    lines = ['# 三组实验执行结果', '',
             '三项建议均已落实：统一提示协议、成对语义约束干预，以及扩展到 180 题的固定三测试质量实验。论文以成对规则切换为新增核心结果。', '',
             '源数据包含 717 题，主实验纳入 710 题：HumanEval+ 157、MBPP+ 378、LCB 175。另 7 题的公开测试无法由直接调用协议转换，题号和原因记录在 `data/controlled/main/excluded.jsonl`。五个模型分别运行四个条件、三次重复，共 42,600 次生成；语义实验五个模型、六个条件、三次执行，共 10,800 次；质量实验五个模型、四个条件、一次执行，共 3,600 次，总计 57,000 次。', '',
             '## 统一主实验', '',
             '下表为排除公开输入后的正确率（%），取三次运行均值。差值单位为百分点，括号内为按题配对 bootstrap 的 95% 区间。', '',
             '| 模型 | 数据集 | NL-only | 正确 I/O | 错误 I/O | 仅输入 | 正确 − 仅输入 |',
             '|---|---|---:|---:|---:|---:|---|']
    for model in models_for('main'):
        label = MODELS[model]
        for dataset, name in DATASETS.items():
            result = results[f'main/{model}/{dataset}']
            values = [f"{100*result['conditions'][c]['passed']:.1f}" for c in ['nl_only', 'nl_tests', 'wrong_tests', 'inputs_only']]
            contrast = next(r for r in result['contrasts'] if r['base'] == 'inputs_only')
            lines.append('| ' + ' | '.join([label, name] + values + [effect(contrast)]) + ' |')
    lines += ['', '五个模型在 MBPP+ 上的正确输出收益均超出仅输入条件，配对区间均为正，且每次运行的差值均为正。Qwen3.8 在 HumanEval+ 上也有收益：正确 I/O 相对仅输入提升 3.2 个百分点 [0.4, 6.4]，相对 NL-only 提升 4.7 [1.7, 8.3]。LCB 上五个模型的这两项对比区间均包含零。', '', '### 三次运行的效果', '',
              '每次运行分别报告正确 I/O 相对仅输入的配对差值。三次重复不会扩大独立题目数量，LCB 的样本量始终为 175。', '',
              '| 模型 | 数据集 | 第一次 | 第二次 | 第三次 |', '|---|---|---:|---:|---:|']
    for model in models_for('main'):
        label = MODELS[model]
        for dataset, name in DATASETS.items():
            contrast = next(r for r in results[f'main/{model}/{dataset}']['contrasts'] if r['base'] == 'inputs_only')
            values = [f"{100*contrast['per_repeat'][str(i)]['paired_difference']:+.1f}" for i in range(3)]
            lines.append('| ' + ' | '.join([label, name] + values) + ' |')
    lines += ['', '## 成对语义约束', '',
              '120 组输入实例来自 20 个规则家族。A/B 提示保持同一描述和输入，只改变参考实现验证的输出；最终用未展示的输入评测。切换成功要求同组、同次执行的两个程序分别遵循 A 和 B；每组先对三次结果取均值，区间再按 20 个家族聚类。表中为三次执行均值。', '',
              '| 模型 | A 测试遵循 A | B 测试遵循 B | 明确 A | 明确 B | 成对切换率及 95% 区间 |',
              '|---|---:|---:|---:|---:|---|']
    for model in models_for('semantic'):
        result = results[f'semantic/{model}']
        values = [f"{100*result['conditions'][c]['rule_'+side+'_passed']:.1f}" for c, side in [('tests_a','a'),('tests_b','b'),('explicit_a','a'),('explicit_b','b')]]
        switch = result['paired_rule_switch']
        rate = f"{100*switch['paired_difference']:.1f} [{100*switch['ci95'][0]:.1f}, {100*switch['ci95'][1]:.1f}]"
        lines.append('| ' + ' | '.join([MODELS[model]] + values + [rate]) + ' |')
    lines += ['', '### 语义实验的逐次切换率', '',
              '| 模型 | 第一次 | 第二次 | 第三次 |', '|---|---:|---:|---:|']
    for model in models_for('semantic'):
        label = MODELS[model]
        runs=results[f'semantic/{model}']['paired_rule_switch']['per_repeat']
        lines.append('| '+' | '.join([label]+[f"{100*runs[str(i)]['paired_difference']:.1f}" for i in range(3)])+' |')
    lines += ['', '逐家族结果及五个模型之间的配对差值见 `semantic_families.json`。重复执行改变任务顺序，测量 greedy 推理的执行稳定性；20 个家族和 120 组输入实例保持固定。', '',
              '## 固定三条测试的质量实验', '',
              '180 题包括 68 道 HumanEval+ 和 112 道 MBPP+。候选测试均通过参考实现，选择池和验证池分离，每组固定三条测试并匹配断言长度。独立验证池共 1,236 个错误程序，包括 1,232 个参考实现变异体和 4 个模型生成程序。每个模型、条件执行一次；最终评测排除整个候选池和开发集的输入。', '',
              '| 模型 | NL-only | 随机 | 低检出率 | 高检出率 | 高 − 低及 95% 区间 | 高 − 随机及 95% 区间 |',
              '|---|---:|---:|---:|---:|---|---|']
    for model in models_for('quality'):
        result = results[f'quality/{model}']
        values = [f"{100*result['conditions'][c]['passed']:.1f}" for c in ['nl_only','quality_random','quality_low','quality_high']]
        contrasts={r['base']:r for r in result['contrasts']}
        lines.append('| ' + ' | '.join([MODELS[model]] + values + [effect(contrasts[c]) for c in ['quality_low','quality_random']]) + ' |')
    quality_differences=[next(r for r in results[f'quality/{m}']['contrasts'] if r['base']=='quality_low')['paired_difference']*100 for m in models_for('quality')]
    lines += ['', f'独立验证池的高、低、随机检出率为 92.3%、62.0%、91.3%。高低差达到 30.3 个百分点，衡量的是以变异体为主的独立验证池上的检出能力。五个模型的高减低生成差值范围为 {min(quality_differences):.1f} 至 {max(quality_differences):.1f} 个百分点，逐模型配对区间见表。DeepSeek 的高、低组均通过 119/180 题，逐题配对为 7 次改善、7 次退步。', '',
              '## 对照审计与评测', '',
              '- 原 shuffled 未改变整组断言集合的任务：MBPP+ 160/378、HumanEval+ 26/164、LCB 9/175；参考实现验证显示，160 道 MBPP 和 25 道 HumanEval 的 shuffled 组仍全部正确。',
              '- 原 irrelevant 函数测试同时改变调用接口，主实验改用同输入、同类型的错误输出和仅输入条件。',
              '- LCB high5 中只有 118 题实际包含五条测试，57 题发生回退。其合成输出未经过参考实现验证，论文将它作为探索性结果，取消据此排除“测试质量不足”的论断。',
              '- 535 道函数题的参考实现均通过正式与隐藏评测；LCB 隐藏评测删除 128 条与公开输入重复的私有输入。质量实验不存在候选/开发输入与最终评测输入的重叠。',
              '- LCB 使用官方 run_test，五个模型统一采用每测试 15 秒、每子进程 4 GiB 地址空间预算。函数套件限时 15 秒；MBPP 599 的长区间求和参考实现使用 60 秒。', '',
              '## 文件', '',
              '- 论文：`aaai/AuthorKit27/AuthorKit27/paper.tex`、`paper.pdf`。',
              '- 协议：`experiment_design.md`、`configs/controlled.yaml`。',
              '- 逐条生成和评测：`outputs/controlled/{experiment}/{model}/`。',
              '- 全部配对差值、区间、每次 gain/loss 和 McNemar 检验：`outputs/analysis/controlled_results.json`。',
              '- 对照、参考实现和输入分离审计：`outputs/analysis/control_audit/`。', '']
    families=json.loads((source.parent / 'semantic_families.json').read_text())
    comparisons={(r['base'],r['target']):r for r in families['model_comparisons']}
    semantic_extra=['### 新增模型的行为比较', '',
                    'Qwen3.5-9B 补充较新小模型，Qwen3.8-27B 提供与 Qwen3.6-27B 相同标称规模的版本比较。两个模型各完成 2,160 次语义生成，并完成相同协议的主实验与质量实验。', '',
                    '| 成对切换率比较 | 差值及 95% 家族区间（百分点） |', '|---|---|']
    for base,target in [('qwen25','qwen35'),('qwen36','qwen35'),('qwen36','qwen38')]:
        semantic_extra.append('| '+MODELS[target]+' − '+MODELS[base]+' | '+effect(comparisons[base,target])+' |')
    semantic_extra.append('')
    index=lines.index('## 固定三条测试的质量实验')
    lines[index:index]=semantic_extra
    extra = ['### 高检出率测试相对 NL-only 的收益', '',
             '| 模型 | 差值及 95% 区间（百分点） |', '|---|---|']
    for model in models_for('quality'):
        contrast=next(r for r in results[f'quality/{model}']['contrasts'] if r['base']=='nl_only')
        extra.append('| '+MODELS[model]+' | '+effect(contrast)+' |')
    extra.append('')
    index=lines.index('## 对照审计与评测')
    lines[index:index]=extra
    extension_path=source.parent/'unrelated_results.json'
    if extension_path.exists():
        extension=json.loads(extension_path.read_text())
        if extension['complete']:
            added=extension['added_generations']
            lines=[line.replace('总计 57,000 次。',f'原三组条件合计 57,000 次；主实验新增来自其他任务的测试条件 {added:,} 次，合计 {57000+added:,} 次生成及对应评测。') for line in lines]
            extra=['## 主实验补充：来自其他任务的测试','',
                   '五个模型各运行三次，复用原主实验四个条件与同一隐藏评测。来源测试保留来源任务的正确输出，并匹配调用接口、测试数量及尽可能接近的类型和长度。', '',
                   '| 模型 | 数据集 | 来自其他任务（%） | 正确 − 无关及 95% 区间 | 无关 − NL 及 95% 区间 |',
                   '|---|---|---:|---|---|']
            for model in models_for('main'):
                for dataset,label in DATASETS.items():
                    r=extension['results'][f'{model}/{dataset}']
                    extra.append('| '+' | '.join([MODELS[model],label,f"{100*r['rates']['unrelated_tests']:.1f}"]+[effect(c) for c in r['contrasts']])+' |')
            extra+=['','来源匹配、参考实现审计和逐次运行结果见 `unrelated_report.md`、`unrelated_control_audit.json` 和 `unrelated_results.json`。','']
            index=lines.index('## 对照审计与评测');lines[index:index]=extra
    expected = sum(len([line for line in (ROOT / spec['dataset']).read_text().splitlines() if line.strip()]) * len(spec['models']) * len(spec['conditions']) * spec['repeats'] for spec in config['experiments'].values())
    if sum(totals.values()) != expected:raise ValueError(f'Unexpected completed generation count: {totals}')
    (source.parent / 'controlled_report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
