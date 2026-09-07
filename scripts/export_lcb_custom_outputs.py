#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.code_utils import strip_markdown_fences
from tddexp.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--condition", default="nl_tests")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.generations)
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("condition") != args.condition:
            continue
        code = row.get("candidate_code") or row.get("completion") or ""
        grouped[row["task_id"]].append(strip_markdown_fences(code).strip() + "\n")

    payload = [{"question_id": task_id, "code_list": codes} for task_id, codes in grouped.items()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload)} LCB custom outputs to {output}")


if __name__ == "__main__":
    main()
