#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
paper_root="$project_root/aaai/AuthorKit27/AuthorKit27"
build_root="$project_root/.tmp/paper_build"
mkdir -p "$build_root"
export TMPDIR="$project_root/.tmp" TMP="$project_root/.tmp" TEMP="$project_root/.tmp"
export TEXMFVAR="$project_root/.tmp/texmf-var" TEXMFCONFIG="$project_root/.tmp/texmf-config"
if [[ -x "$project_root/.tmp/.TinyTeX/bin/x86_64-linux/pdflatex" ]]; then
    tex_bin="$project_root/.tmp/.TinyTeX/bin/x86_64-linux"
    pdf_engine="$tex_bin/pdflatex"
    bib_engine="$tex_bin/bibtex"
else
    pdf_engine="$(command -v pdflatex)"
    bib_engine="$(command -v bibtex)"
fi
cd "$paper_root"
"$pdf_engine" -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory="$build_root" paper.tex > "$build_root/compile.log" 2>&1
(
    cd "$build_root"
    BIBINPUTS="$paper_root:" BSTINPUTS="$paper_root:" "$bib_engine" paper > bibtex.log 2>&1
)
"$pdf_engine" -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory="$build_root" paper.tex >> "$build_root/compile.log" 2>&1
"$pdf_engine" -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory="$build_root" paper.tex > "$build_root/final_compile.log" 2>&1
cp "$build_root/paper.pdf" "$paper_root/paper.pdf"
printf '%s\n' "$paper_root/paper.pdf"
