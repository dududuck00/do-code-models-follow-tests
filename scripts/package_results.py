#!/usr/bin/env python
"""Package completed results, source code, and the compiled manuscript."""
from pathlib import Path
import io,json,os,subprocess,tarfile,time
import yaml
root=Path(__file__).resolve().parents[1]
os.chdir(root)
for variable in ['TMPDIR','TMP','TEMP']:os.environ[variable]=str(root/'.tmp')
(root/'.tmp/package_results').mkdir(parents=True,exist_ok=True)
(root/'artifacts').mkdir(exist_ok=True)
report=json.loads((root/'outputs/analysis/controlled_results.json').read_text())
if not report['complete']:raise SystemExit('Complete the experiment export before packaging.')
observations=sum(c['observations'] for key,r in report['results'].items() if key.count('/')==1 for c in r['conditions'].values())
config=yaml.safe_load((root/'configs/controlled.yaml').read_text())
expected=sum(sum(1 for line in (root/spec['dataset']).open() if line.strip())*len(spec['models'])*len(spec['conditions'])*spec['repeats'] for spec in config['experiments'].values())
if observations!=expected:raise SystemExit(f'Expected {expected} completed observations, found {observations}')
extension_path=root/'configs/unrelated.yaml'
extension_count=0
if extension_path.exists():
    extension=yaml.safe_load(extension_path.read_text())
    extra=json.loads((root/'outputs/analysis/unrelated_results.json').read_text())
    extension_count=sum(sum(1 for line in (root/spec['dataset']).open() if line.strip())*len(spec['models'])*len(spec['conditions'])*spec['repeats'] for spec in extension['experiments'].values())
    if not extra['complete'] or extra['added_generations']!=extension_count or extra['added_evaluations']!=extension_count:
        raise SystemExit('Complete the other-task test control before packaging.')
    config['experiments'].update(extension['experiments'])
target=root/'artifacts/tdd_results_20260908.tar.xz'

paths=set()
skip={'.git','__pycache__','.pytest_cache','.DS_Store'}
def include(path):
    path=root/path
    if not path.exists():raise FileNotFoundError(path)
    if path.is_file():paths.add(path)
    else:
        for p in path.rglob('*'):
            if any(part in skip for part in p.relative_to(path).parts):continue
            if p.is_symlink():raise ValueError(f'Unexpected symlink: {p}')
            if p.is_file() and p.suffix not in {'.pyc','.pyo'}:paths.add(p)
for name in ['README.md','PROJECT_STATE.md','experiment_design.md','Makefile','AGENTS.md','.gitignore','environment.yml']:
    include(name)
for p in root.glob('requirements*.txt'):include(p.relative_to(root))
for name in ['src','scripts','tests','configs','data/controlled','data/model_tokenizers','outputs/analysis','.tmp/LiveCodeBench']:
    include(name)
for p in (root/'data').glob('*.jsonl'):include(p.relative_to(root))
for source in ['humanevalplus','mbppplus']:
    include(f'data/synthetic_public_tests/{source}/synthetic_test_candidates.jsonl')
for exp,spec in config['experiments'].items():
    models=spec['models']
    for model in models:
        for name in ['generations.jsonl','eval_results.jsonl','summary.json','protocol.json']:
            include(f'outputs/controlled/{exp}/{model}/{name}')
paper='aaai/AuthorKit27/AuthorKit27'
for name in ['paper.tex','paper.pdf','paper.bib','aaai2027.sty','aaai2027.bst','Tables','Figures']:
    include(f'{paper}/{name}')
records=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size} for p in sorted(paths)]
manifest={'archive_root':'tdd_results','source_files':len(records),'source_bytes':sum(r['bytes'] for r in records),'files':records}
readme='''# 实验结果归档

从本目录解压后保留原项目相对路径。

优先阅读：
- outputs/analysis/controlled_report.md：中文实验结果报告。
- aaai/AuthorKit27/AuthorKit27/paper.pdf：最终论文 PDF。
- experiment_design.md：实验协议。
- outputs/analysis/controlled_results.json：完整配对统计与逐次运行效果。

内容包括主实验 42,600 条、语义实验 10,800 条、质量实验 3,600 条，共 57,000 条生成及对应评测记录、冻结数据、参考实现与错误程序池、审计结果、代码、依赖说明、论文源码及表格。
LCB 官方评分器源码保留在 .tmp/LiveCodeBench，以匹配 configs/controlled.yaml 中的路径；该目录不含 Git 元数据。
模型权重、虚拟环境、TeX 安装、隐藏状态大文件、运行检查点、日志和试运行目录未包含。
outputs/analysis 中保留论文使用的历史分析；新的主结果位于 outputs/controlled。

复现：安装 requirements 文件指定的依赖；代码运行指令见 README.md。
模型路径需要按本机位置修改 configs/controlled.yaml。评测生成代码时使用代码执行沙箱。
论文编译需要 pdfLaTeX 和 BibTeX，执行 bash scripts/build_paper.sh。
冻结数据与最终结果可直接用于重新统计和导出；不需要重新生成错误程序池。

MANIFEST.json 列出了归档中的项目文件及其未压缩大小。
'''
if extension_count:
    readme += f'\n补充条件：来自其他任务的测试，新增 {extension_count:,} 条生成及评测，总计 {observations+extension_count:,} 条生成及对应评测。结果见 outputs/analysis/unrelated_report.md 与 unrelated_results.json。\n'
partial=root/'.tmp/package_results/archive.tar.xz.partial'
print(json.dumps({'source_files':len(records),'source_bytes':manifest['source_bytes'],'target':str(target)}),flush=True)
with partial.open('wb') as out:
    proc=subprocess.Popen(['xz','-T4','-6','-c'],stdin=subprocess.PIPE,stdout=out)
    try:
        with tarfile.open(fileobj=proc.stdin,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for name,content in [('ARCHIVE_README.md',readme),('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')]:
                data=content.encode();info=tarfile.TarInfo('tdd_results/'+name);info.size=len(data);info.mode=0o644;info.mtime=int(time.time())
                archive.addfile(info,io.BytesIO(data))
            for p in sorted(paths):archive.add(p,arcname='tdd_results/'+str(p.relative_to(root)),recursive=False)
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('xz compression failed')
    except BaseException:
        proc.kill();proc.wait();raise
os.replace(partial,target)
(root/'.tmp/package_results/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'archive':str(target),'compressed_bytes':target.stat().st_size}),flush=True)
