#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.io import append_jsonl, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded generation JSONL files.")
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expect-conditions", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for shard_dir in args.shard_dirs:
        path = Path(shard_dir) / "generations.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing shard output: {path}")
        rows.extend(read_jsonl(path))

    if not rows:
        raise SystemExit("No generation rows found.")

    seen: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    for row in rows:
        key = (str(row["task_id"]), str(row["condition"]))
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise SystemExit(f"Found duplicate task/condition rows, first={duplicates[:5]}")

    if args.expect_conditions:
        validate_condition_coverage(rows, args.expect_conditions)

    rows = sorted(rows, key=lambda row: (str(row["task_id"]), str(row["condition"])))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    for row in rows:
        append_jsonl(output, row)
    print(f"Wrote {len(rows)} merged rows to {output}")


def validate_condition_coverage(rows: list[dict], conditions: list[str]) -> None:
    by_task: dict[str, set[str]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), set()).add(str(row["condition"]))
    expected = set(conditions)
    bad = [
        (task_id, sorted(expected - seen_conditions), sorted(seen_conditions - expected))
        for task_id, seen_conditions in by_task.items()
        if seen_conditions != expected
    ]
    if bad:
        raise SystemExit(
            "Condition coverage mismatch, first="
            + json.dumps(bad[:5], ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
