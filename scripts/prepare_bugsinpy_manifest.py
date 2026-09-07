#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bugsinpy-root", required=True)
    parser.add_argument("--output", default="data/bugsinpy_manifest.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.bugsinpy_root)
    if not root.exists():
        raise SystemExit(f"BugsInPy root does not exist: {root}")

    rows = []
    for project_dir in sorted((root / "projects").glob("*")) if (root / "projects").exists() else []:
        if not project_dir.is_dir():
            continue
        bugs_dir = project_dir / "bugs"
        if not bugs_dir.exists():
            continue
        for bug_dir in sorted(bugs_dir.glob("*")):
            rows.append(
                {
                    "task_id": f"BugsInPy/{project_dir.name}/{bug_dir.name}",
                    "project": project_dir.name,
                    "bug_id": bug_dir.name,
                    "bug_dir": str(bug_dir),
                    "note": "Use BugsInPy tooling to checkout, reproduce failing tests, and extract patches.",
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} BugsInPy manifest rows to {output}")


if __name__ == "__main__":
    main()
