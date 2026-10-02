#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
paper_root="$project_root/paper/semantic_representation"
build_root="$project_root/.tmp/semantic_representation/latex"
mkdir -p "$build_root"
export TMPDIR="$project_root/.tmp" TMP="$project_root/.tmp" TEMP="$project_root/.tmp"
export TEXMFVAR="$project_root/.tmp/texmf-var" TEXMFCONFIG="$project_root/.tmp/texmf-config"
pdf_engine="$project_root/.tmp/.TinyTeX/bin/x86_64-linux/pdflatex"
if [[ ! -x "$pdf_engine" ]]; then pdf_engine="$(command -v pdflatex)"; fi
cd "$paper_root"
for pass in 1 2 3; do
    "$pdf_engine" -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory="$build_root" main.tex > "$build_root/final_compile.log" 2>&1
done
cp "$build_root/main.pdf" "$paper_root/semantic_representation.pdf"
printf '%s\n' "$paper_root/semantic_representation.pdf"
