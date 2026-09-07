#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.io import append_jsonl, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge eval_results JSONL files without duplicate task/condition rows.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-duplicates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []

    for raw_path in args.inputs:
        path = Path(raw_path)
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")
        for row in read_jsonl(path):
            key = (str(row["task_id"]), str(row["condition"]))
            if key in seen:
                duplicates.append(key)
                if not args.allow_duplicates:
                    continue
            seen.add(key)
            rows.append(row)

    if duplicates and not args.allow_duplicates:
        raise SystemExit(f"Found duplicate task/condition rows, first={duplicates[:5]}")
    if not rows:
        raise SystemExit("No rows to merge.")

    rows = sorted(rows, key=lambda row: (str(row.get("dataset_name", "")), str(row["task_id"]), str(row["condition"])))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    for row in rows:
        append_jsonl(output, row)
    print(f"Wrote {len(rows)} merged eval rows to {output}")


if __name__ == "__main__":
    main()
