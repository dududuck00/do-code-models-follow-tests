#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    tmp = ROOT / ".tmp"
    tmp.mkdir(exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp))
    os.environ.setdefault("TMP", str(tmp))
    os.environ.setdefault("TEMP", str(tmp))
    out = "outputs/smoke"
    run(
        [
            sys.executable,
            "scripts/run_generation.py",
            "--dataset",
            "data/humaneval.jsonl",
            "--dataset-name",
            "humaneval",
            "--conditions",
            "nl_only",
            "nl_tests",
            "shuffled_tests",
            "irrelevant_tests",
            "--limit",
            str(args.limit),
            "--visible-tests",
            "3",
            "--dry-run",
            "--output-dir",
            out,
        ]
    )
    run(
        [
            sys.executable,
            "scripts/evaluate_generations.py",
            "--generations",
            f"{out}/generations.jsonl",
            "--output",
            f"{out}/eval_results.jsonl",
            "--tmp-root",
            ".tmp",
        ]
    )
    run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            "--eval-results",
            f"{out}/eval_results.jsonl",
            "--output",
            f"{out}/summary.csv",
        ]
    )


if __name__ == "__main__":
    main()
