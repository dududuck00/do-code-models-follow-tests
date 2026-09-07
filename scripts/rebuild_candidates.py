#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.code_utils import assemble_candidate
from tddexp.io import append_jsonl, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.generations)
    output = Path(args.output)
    if output.exists():
        output.unlink()

    for row in rows:
        row["candidate_code"] = assemble_candidate(
            row["prompt"],
            row["completion"],
            entry_point=row.get("entry_point"),
            prompt_is_code_prefix=is_code_prefix_dataset(str(row.get("dataset_name", ""))),
        )
        append_jsonl(output, row)

    print(f"Wrote rebuilt generations to {output}")


def is_code_prefix_dataset(dataset_name: str) -> bool:
    return not dataset_name.lower().startswith(("livecodebench", "mbppplus", "humanevalplus"))


if __name__ == "__main__":
    main()
