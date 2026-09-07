#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quixbugs-root", required=True)
    parser.add_argument("--output", default="data/quixbugs_repair.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.quixbugs_root)
    buggy_dir = root / "python_programs"
    correct_dir = root / "correct_python_programs"
    tests_dir = root / "python_testcases"
    if not buggy_dir.exists():
        raise SystemExit(f"Missing QuixBugs python_programs directory: {buggy_dir}")

    rows = []
    for buggy_file in sorted(buggy_dir.glob("*.py")):
        name = buggy_file.stem
        correct_file = correct_dir / buggy_file.name
        test_file = tests_dir / f"test_{name}.py"
        rows.append(
            {
                "task_id": f"QuixBugs/{name}",
                "program_name": name,
                "buggy_code": buggy_file.read_text(encoding="utf-8"),
                "correct_code": correct_file.read_text(encoding="utf-8") if correct_file.exists() else "",
                "test_file": str(test_file),
                "prompt": build_repair_prompt(name, buggy_file.read_text(encoding="utf-8")),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} QuixBugs repair rows to {output}")


def build_repair_prompt(name: str, buggy_code: str) -> str:
    return (
        f"### Repair Task: {name}\n\n"
        "The following Python program contains a bug. Fix the program while preserving its public API. "
        "Return only the corrected Python code.\n\n"
        "```python\n"
        + buggy_code.rstrip()
        + "\n```\n\n"
        "### Corrected code\n```python\n"
    )


if __name__ == "__main__":
    main()
