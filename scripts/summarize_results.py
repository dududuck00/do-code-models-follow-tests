#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.DataFrame(read_jsonl(args.eval_results))
    summary = (
        df.groupby("condition")
        .agg(
            n=("task_id", "count"),
            visible_pass_rate=("visible_passed", "mean"),
            hidden_pass_rate=("hidden_passed", "mean"),
        )
        .reset_index()
        .sort_values("condition")
    )
    text = summary.to_string(index=False)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, index=False)


if __name__ == "__main__":
    main()
