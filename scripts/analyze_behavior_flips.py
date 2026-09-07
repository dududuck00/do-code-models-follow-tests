#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.io import read_jsonl
from tddexp.statistics import paired_effect


DEFAULT_PAIRS = [
    ("nl_only", "nl_tests"),
    ("nl_only", "shuffled_tests"),
    ("nl_only", "irrelevant_tests"),
    ("shuffled_tests", "nl_tests"),
    ("irrelevant_tests", "nl_tests"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze per-task behavioral flips between prompt conditions. "
            "Inputs can be EvalPlus-style eval_results.jsonl files or "
            "LiveCodeBench eval directories containing *_eval_all.jsonl files."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run spec in the form name=path. Path may be an eval_results.jsonl file "
            "or a directory with *_eval_all.jsonl files. Can be repeated."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/analysis/behavior")
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=[f"{left}:{right}" for left, right in DEFAULT_PAIRS],
        help="Condition pairs as base:target. A gain means base failed and target passed.",
    )
    parser.add_argument(
        "--pass-field",
        default=None,
        help=(
            "Optional field to use as the pass label. Defaults to hidden_passed, "
            "then evalplus_passed, then passed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = parse_pairs(args.pairs)

    all_summary: list[dict] = []
    all_pairwise: list[dict] = []
    all_task_rows: list[dict] = []
    flip_payload: dict[str, dict] = {}

    for run_name, path in parse_run_specs(args.run):
        rows = load_eval_rows(path)
        if not rows:
            raise SystemExit(f"No eval rows found for {run_name}: {path}")
        matrix = build_task_matrix(rows, pass_field=args.pass_field)
        metadata = collect_task_metadata(rows)

        all_summary.extend(condition_summary(run_name, rows, pass_field=args.pass_field))
        pairwise_rows, run_flips = pairwise_flips(run_name, matrix, pairs, metadata)
        all_pairwise.extend(pairwise_rows)
        flip_payload[run_name] = run_flips
        all_task_rows.extend(task_matrix_rows(run_name, matrix, metadata))

    write_csv(output_dir / "behavior_summary.csv", all_summary)
    write_csv(output_dir / "pairwise_flips.csv", all_pairwise)
    write_csv(output_dir / "task_condition_matrix.csv", all_task_rows)
    (output_dir / "behavior_flips.json").write_text(
        json.dumps(flip_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote behavior summary to {output_dir / 'behavior_summary.csv'}")
    print(f"Wrote pairwise flips to {output_dir / 'pairwise_flips.csv'}")
    print(f"Wrote flip task lists to {output_dir / 'behavior_flips.json'}")
    print_compact_report(all_summary, all_pairwise)


def parse_run_specs(specs: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" in spec:
            name, raw_path = spec.split("=", 1)
            name = name.strip()
            path = Path(raw_path.strip())
        else:
            path = Path(spec)
            name = path.parent.name if path.is_file() else path.name
        if not name:
            raise SystemExit(f"Run name is empty in spec: {spec}")
        if not path.exists():
            raise SystemExit(f"Run path does not exist for {name}: {path}")
        runs.append((name, path))
    return runs


def parse_pairs(specs: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"Pair must use base:target format: {spec}")
        left, right = [part.strip() for part in spec.split(":", 1)]
        if not left or not right:
            raise SystemExit(f"Pair must use non-empty condition names: {spec}")
        pairs.append((left, right))
    return pairs


def load_eval_rows(path: Path) -> list[dict]:
    if path.is_file():
        return read_jsonl(path)
    rows: list[dict] = []
    for file_path in sorted(path.glob("*_eval_all.jsonl")):
        rows.extend(read_jsonl(file_path))
    return rows


def pass_value(row: dict, pass_field: str | None = None) -> bool:
    if pass_field:
        if pass_field not in row:
            raise SystemExit(f"Requested pass field is missing: {pass_field}")
        return bool(row[pass_field])
    for field in ("hidden_passed", "evalplus_passed", "passed"):
        if field in row:
            return bool(row[field])
    raise SystemExit("Could not find a pass field in eval row.")


def visible_value(row: dict) -> bool | None:
    if "visible_passed" in row:
        return bool(row["visible_passed"])
    return None


def build_task_matrix(rows: list[dict], pass_field: str | None = None) -> dict[str, dict[str, bool]]:
    matrix: dict[str, dict[str, bool]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        condition = str(row["condition"])
        matrix.setdefault(task_id, {})[condition] = pass_value(row, pass_field=pass_field)
    return matrix


def collect_task_metadata(rows: list[dict]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in metadata:
            continue
        keep = {}
        for field in (
            "dataset_name",
            "difficulty",
            "platform",
            "contest_date",
            "question_id",
            "entry_point",
        ):
            if field in row and row[field] not in (None, ""):
                keep[field] = row[field]
        metadata[task_id] = keep
    return metadata


def condition_summary(run_name: str, rows: list[dict], pass_field: str | None = None) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)

    summary: list[dict] = []
    for condition in sorted(grouped):
        condition_rows = grouped[condition]
        passed = sum(1 for row in condition_rows if pass_value(row, pass_field=pass_field))
        visible_known = [visible_value(row) for row in condition_rows if visible_value(row) is not None]
        visible_passed = sum(1 for value in visible_known if value)
        summary.append(
            {
                "run": run_name,
                "condition": condition,
                "n": len(condition_rows),
                "passed": passed,
                "pass_rate": passed / len(condition_rows) if condition_rows else "",
                "visible_n": len(visible_known),
                "visible_passed": visible_passed if visible_known else "",
                "visible_pass_rate": visible_passed / len(visible_known) if visible_known else "",
            }
        )
    return summary


def pairwise_flips(
    run_name: str,
    matrix: dict[str, dict[str, bool]],
    pairs: list[tuple[str, str]],
    metadata: dict[str, dict],
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    payload: dict[str, dict] = {}
    for base, target in pairs:
        comparable = sorted(
            task_id
            for task_id, conditions in matrix.items()
            if base in conditions and target in conditions
        )
        base_pass = [task_id for task_id in comparable if matrix[task_id][base]]
        target_pass = [task_id for task_id in comparable if matrix[task_id][target]]
        gains = [task_id for task_id in comparable if not matrix[task_id][base] and matrix[task_id][target]]
        losses = [task_id for task_id in comparable if matrix[task_id][base] and not matrix[task_id][target]]
        both_pass = [task_id for task_id in comparable if matrix[task_id][base] and matrix[task_id][target]]
        both_fail = [task_id for task_id in comparable if not matrix[task_id][base] and not matrix[task_id][target]]
        pair_name = f"{base}_to_{target}"
        paired = paired_effect(
            [{"task_id": task_id, "condition": condition, "passed": matrix[task_id][condition]}
             for task_id in comparable for condition in (base, target)], base, target,
        ) if comparable else None

        rows.append(
            {
                "run": run_name,
                "base_condition": base,
                "target_condition": target,
                "n_comparable": len(comparable),
                "base_passed": len(base_pass),
                "target_passed": len(target_pass),
                "base_pass_rate": len(base_pass) / len(comparable) if comparable else "",
                "target_pass_rate": len(target_pass) / len(comparable) if comparable else "",
                "gain_count": len(gains),
                "loss_count": len(losses),
                "net_gain": len(gains) - len(losses),
                "net_gain_rate": (len(gains) - len(losses)) / len(comparable) if comparable else "",
                "both_pass_count": len(both_pass),
                "both_fail_count": len(both_fail),
                "exact_mcnemar_p": paired["per_repeat"]["0"]["exact_mcnemar_p"] if paired else None,
                "paired_difference_ci95_low": paired["ci95"][0] if paired else None,
                "paired_difference_ci95_high": paired["ci95"][1] if paired else None,
            }
        )
        payload[pair_name] = {
            "base_condition": base,
            "target_condition": target,
            "comparable_count": len(comparable),
            "gains": task_entries(gains, metadata),
            "losses": task_entries(losses, metadata),
            "both_pass": task_entries(both_pass, metadata),
            "both_fail": task_entries(both_fail, metadata),
        }
    return rows, payload


def task_entries(task_ids: Iterable[str], metadata: dict[str, dict]) -> list[dict]:
    entries: list[dict] = []
    for task_id in task_ids:
        entry = {"task_id": task_id}
        entry.update(metadata.get(task_id, {}))
        entries.append(entry)
    return entries


def task_matrix_rows(
    run_name: str,
    matrix: dict[str, dict[str, bool]],
    metadata: dict[str, dict],
) -> list[dict]:
    conditions = sorted({condition for values in matrix.values() for condition in values})
    rows: list[dict] = []
    for task_id in sorted(matrix):
        row = {"run": run_name, "task_id": task_id}
        row.update(metadata.get(task_id, {}))
        for condition in conditions:
            value = matrix[task_id].get(condition)
            row[condition] = "" if value is None else int(value)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    preferred = [
        "run",
        "condition",
        "base_condition",
        "target_condition",
        "task_id",
        "n",
        "n_comparable",
        "passed",
        "pass_rate",
        "base_passed",
        "target_passed",
        "base_pass_rate",
        "target_pass_rate",
        "gain_count",
        "loss_count",
        "net_gain",
        "net_gain_rate",
        "both_pass_count",
        "both_fail_count",
        "visible_n",
        "visible_passed",
        "visible_pass_rate",
    ]
    ordered = [field for field in preferred if field in fieldnames] + [
        field for field in fieldnames if field not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def print_compact_report(summary: list[dict], pairwise: list[dict]) -> None:
    print("\nCondition pass rates:")
    for row in summary:
        print(
            f"{row['run']:32s} {row['condition']:18s} "
            f"{row['passed']}/{row['n']} = {float(row['pass_rate']):.4f}"
        )

    print("\nPairwise flips:")
    for row in pairwise:
        print(
            f"{row['run']:32s} {row['base_condition']} -> {row['target_condition']}: "
            f"gain={row['gain_count']} loss={row['loss_count']} "
            f"net={row['net_gain']} ({float(row['net_gain_rate']):+.4f})"
        )


if __name__ == "__main__":
    main()
