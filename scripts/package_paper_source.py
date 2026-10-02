#!/usr/bin/env python
"""Package standalone LaTeX sources for local builds and Overleaf."""
from pathlib import Path
import io
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'aaai/AuthorKit27/AuthorKit27'
README = '''# 论文 LaTeX 源码

主文件：`paper.tex`。三组实验均包含 Qwen2.5-Coder-7B、DeepSeek-Coder-6.7B、Qwen3.6-27B、Qwen3.5-9B、Qwen3.8-27B。主实验与语义实验各执行三次，质量实验各执行一次。主实验还包含五模型、三次运行的“来自其他任务的测试”补充对照。LCB 探索性附录已补齐 Qwen3.6 high3 结果。

- `paper.bib`：参考文献。
- `aaai2027.sty`、`aaai2027.bst`：AAAI 模板和参考文献样式。
- `Tables/`：正文及附录表格。
- `Figures/`：项目论文图件。

本地编译：安装 TeX Live 或 MiKTeX 后执行 `bash build.sh`，生成 `paper.pdf`。
所需宏包包括 newtxtext、helvet、courier、placeins、natbib、caption、booktabs、algorithm 和 algorithmic，可通过 TeX 发行版安装。

Overleaf：上传 ZIP 包，主文件选择 `paper.tex`，编译器选择 pdfLaTeX。
表格数值已包含在源码中，编译无需运行实验代码。
'''
BUILD = '''#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$source_root"
mkdir -p .build
export TMPDIR="$source_root/.build" TMP="$source_root/.build" TEMP="$source_root/.build"
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=.build paper.tex
(cd .build && BIBINPUTS="$source_root:" BSTINPUTS="$source_root:" bibtex paper)
for pass in 1 2 3; do
    pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=.build paper.tex
done
cp .build/paper.pdf paper.pdf
'''


def main():
    paths = [PAPER / name for name in ['paper.tex', 'paper.bib', 'aaai2027.sty', 'aaai2027.bst']]
    paths += sorted((PAPER / 'Tables').glob('*.tex'))
    paths += sorted(p for p in (PAPER / 'Figures').iterdir() if p.is_file())
    files = {str(p.relative_to(PAPER)): p.read_bytes() for p in paths}
    files.update({'README.md': README.encode(), 'build.sh': BUILD.encode()})
    dest = ROOT / 'artifacts'
    dest.mkdir(exist_ok=True)
    tar_path = dest / 'tdd_paper_latex.tar.xz'
    zip_path = dest / 'tdd_paper_latex.zip'
    with tarfile.open(tar_path, 'w:xz') as archive:
        for name, data in files.items():
            info = tarfile.TarInfo('tdd_paper_latex/' + name)
            info.size = len(data)
            info.mode = 0o755 if name == 'build.sh' else 0o644
            archive.addfile(info, io.BytesIO(data))
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with tarfile.open(tar_path, 'r:xz') as archive:
        actual = {m.name.removeprefix('tdd_paper_latex/'): archive.extractfile(m).read() for m in archive}
        assert actual == files
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert {name: archive.read(name) for name in archive.namelist()} == files
    print(f'{len(files)} verified source files: {tar_path}, {zip_path}')


if __name__ == '__main__':
    main()
