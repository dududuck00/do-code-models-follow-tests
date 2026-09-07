#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="livecodebench/code_generation_lite")
    parser.add_argument(
        "--local-path",
        default=None,
        help="Local clone/download of livecodebench/code_generation_lite. If set, read JSONL files directly.",
    )
    parser.add_argument("--version-tag", default="release_v6")
    parser.add_argument(
        "--exclude-version-tag",
        default=None,
        help="Keep only tasks in --version-tag whose stable ids are absent from this older release.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="data/livecodebench_release_v6.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--platform", choices=["leetcode", "atcoder", "codeforces"], default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--keep-question-public-tests",
        action="store_true",
        help="Keep sample/example tests inside question_content. By default they are stripped into question_content_no_public_tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_path:
        rows = load_local_rows(
            local_path=Path(args.local_path),
            version_tag=args.version_tag,
            exclude_version_tag=args.exclude_version_tag,
        )
        rows = apply_filters(rows, args)
        if args.limit is not None:
            rows = rows[: args.limit]
        if not args.keep_question_public_tests:
            rows = add_sanitized_question_content(rows)
        write_rows(rows, args.output)
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required. Install with `pip install -r requirements-data.txt`."
        ) from exc

    excluded_ids = set()
    if args.exclude_version_tag:
        excluded = load_dataset(
            args.dataset,
            version_tag=args.exclude_version_tag,
            split=args.split,
            trust_remote_code=True,
        )
        excluded_ids = {stable_task_id(dict(row)) for row in excluded}
        print(f"Loaded {len(excluded_ids)} task ids from {args.exclude_version_tag} to exclude")

    ds = load_dataset(
        args.dataset,
        version_tag=args.version_tag,
        split=args.split,
        trust_remote_code=True,
    )
    rows = []
    for row in ds:
        row = dict(row)
        if stable_task_id(row) in excluded_ids:
            continue
        rows.append(row)
    rows = apply_filters(rows, args)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not args.keep_question_public_tests:
        rows = add_sanitized_question_content(rows)

    write_rows(rows, args.output)


def write_rows(rows: list[dict], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} LiveCodeBench rows to {output}")


def apply_filters(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    filtered = []
    for row in rows:
        if args.platform and str(row.get("platform", "")).lower() != args.platform:
            continue
        if args.difficulty and str(row.get("difficulty", "")).lower() != args.difficulty.lower():
            continue
        date = str(row.get("contest_date", ""))
        if args.start_date and date < args.start_date:
            continue
        if args.end_date and date > args.end_date:
            continue
        filtered.append(row)
    return filtered


def add_sanitized_question_content(rows: list[dict]) -> list[dict]:
    sanitized = []
    for row in rows:
        row = dict(row)
        original = str(row.get("question_content", ""))
        row["question_content_no_public_tests"] = strip_public_examples(original)
        sanitized.append(row)
    return sanitized


def strip_public_examples(content: str) -> str:
    text = content.replace("\r\n", "\n")

    # LeetCode style: examples appear before Constraints.
    text = re.sub(
        r"\n\s*Example\s+1\s*:.*?(\n\s*Constraints\s*:)",
        r"\1",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # AtCoder style: Sample Input/Output sections usually come after constraints.
    text = re.sub(
        r"\n\s*Sample\s+Input\s+1\b.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Codeforces-like style, retained as a fallback for future releases.
    text = re.sub(
        r"\n\s*Examples?\s*\n.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return text.strip()


def load_local_rows(local_path: Path, version_tag: str, exclude_version_tag: str | None) -> list[dict]:
    if not local_path.exists():
        raise SystemExit(f"Local LiveCodeBench path does not exist: {local_path}")

    include_files = files_for_version(version_tag)
    exclude_files = set(files_for_version(exclude_version_tag)) if exclude_version_tag else set()
    target_files = [name for name in include_files if name not in exclude_files]
    if not target_files:
        return []

    rows = []
    for name in target_files:
        path = local_path / name
        if not path.exists():
            raise SystemExit(f"Missing local LiveCodeBench file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    print(f"Loaded local LiveCodeBench files: {', '.join(target_files)}")
    return rows


def files_for_version(version_tag: str | None) -> list[str]:
    if version_tag is None:
        return []
    cumulative = {
        "release_v1": ["test.jsonl"],
        "release_v2": ["test.jsonl", "test2.jsonl"],
        "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
        "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
        "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
        "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
        "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    }
    if version_tag in cumulative:
        return cumulative[version_tag]
    if version_tag in {"v1", "v2", "v3", "v4", "v5", "v6"}:
        idx = int(version_tag[1:])
        return ["test.jsonl" if idx == 1 else f"test{idx}.jsonl"]
    if "_" in version_tag:
        start, end = version_tag.split("_", 1)
        if start.startswith("v") and end.startswith("v"):
            return ["test.jsonl" if idx == 1 else f"test{idx}.jsonl" for idx in range(int(start[1:]), int(end[1:]) + 1)]
    raise SystemExit(f"Unsupported LiveCodeBench version tag for local loading: {version_tag}")


def stable_task_id(row: dict) -> str:
    """Return a stable identifier across LiveCodeBench releases."""
    for key in ("question_id", "id", "task_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    title = str(row.get("question_title", "")).strip()
    platform = str(row.get("platform", "")).strip()
    contest_id = str(row.get("contest_id", "")).strip()
    return "::".join(part for part in (platform, contest_id, title) if part)


if __name__ == "__main__":
    main()
