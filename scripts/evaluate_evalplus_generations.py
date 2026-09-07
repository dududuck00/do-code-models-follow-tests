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
    parser = argparse.ArgumentParser(description="Evaluate HumanEval+/MBPP+ generations with EvalPlus tests.")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--tmp-root", default=".tmp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_rows = {str(row["task_id"]): row for row in read_jsonl(args.dataset)}
    generation_rows = read_jsonl(args.generations)
    output = Path(args.output)
    if output.exists():
        output.unlink()

    for row in tqdm(generation_rows, desc="evalplus"):
        task_id = str(row["task_id"])
        dataset_row = dataset_rows.get(task_id)
        if dataset_row is None:
            raise SystemExit(f"Missing dataset row for task_id={task_id}")

        visible_result = evaluate_code(
            row["candidate_code"],
            row.get("visible_tests") or [],
            timeout=args.timeout,
            tmp_root=args.tmp_root,
        )
        private_result = evaluate_code(
            row["candidate_code"],
            [build_evalplus_test_block(dataset_row)],
            timeout=args.timeout,
            tmp_root=args.tmp_root,
        )
        row.update(
            {
                "visible_passed": visible_result.passed,
                "visible_error": visible_result.error,
                "hidden_passed": private_result.passed,
                "hidden_error": private_result.error,
                "evalplus_passed": private_result.passed,
                "evalplus_error": private_result.error,
            }
        )
        append_jsonl(output, row)

    print(f"Wrote EvalPlus evaluation results to {output}")


def build_evalplus_test_block(row: dict) -> str:
    test = str(row.get("test", ""))
    if "canonical_solution" in row:
        entry_point = row["entry_point"]
        return test.rstrip() + f"\n\ncheck({entry_point})\n"
    return test


if __name__ == "__main__":
    main()
