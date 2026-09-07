#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.evaluation import evaluate_code
from tddexp.io import append_jsonl, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--tmp-root", default=".tmp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.generations)
    output = Path(args.output)
    if output.exists():
        output.unlink()

    for row in tqdm(rows, desc="evaluate"):
        visible_result = evaluate_code(
            row["candidate_code"],
            row["visible_tests"],
            timeout=args.timeout,
            tmp_root=args.tmp_root,
        )
        hidden_result = evaluate_code(
            row["candidate_code"],
            row["hidden_tests"],
            timeout=args.timeout,
            tmp_root=args.tmp_root,
        )
        row.update(
            {
                "visible_passed": visible_result.passed,
                "visible_error": visible_result.error,
                "hidden_passed": hidden_result.passed,
                "hidden_error": hidden_result.error,
            }
        )
        append_jsonl(output, row)

    print(f"Wrote evaluation results to {output}")


if __name__ == "__main__":
    main()
