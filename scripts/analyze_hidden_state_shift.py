#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure layer-wise hidden-state shifts between a baseline condition and target conditions."
    )
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--baseline-condition", default="nl_only")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--output-dir", default="outputs/analysis/hidden_shift")
    parser.add_argument("--pass-field", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.eval_results)
    if not rows:
        raise SystemExit(f"No rows found: {args.eval_results}")

    by_task = group_by_task_condition(rows)
    conditions = args.conditions or sorted(
        {
            str(row["condition"])
            for row in rows
            if str(row["condition"]) != args.baseline_condition
        }
    )

    task_rows: list[dict] = []
    layer_accumulator: dict[tuple[str, int], list[dict]] = {}
    for task_id, condition_rows in tqdm(by_task.items(), desc="hidden shifts", unit="task"):
        baseline_row = condition_rows.get(args.baseline_condition)
        if not baseline_row:
            continue
        baseline = load_state(baseline_row)
        if baseline is None:
            continue
        for condition in conditions:
            target_row = condition_rows.get(condition)
            if not target_row:
                continue
            target = load_state(target_row)
            if target is None or target.shape != baseline.shape:
                continue
            prompt_tests = infer_prompt_tests(target_row)
            row_meta = {
                "task_id": task_id,
                "condition": condition,
                "baseline_condition": args.baseline_condition,
                "prompt_test_count": len(prompt_tests),
                "target_passed": int(pass_value(target_row, args.pass_field)),
                "baseline_passed": int(pass_value(baseline_row, args.pass_field)),
                "dataset_name": target_row.get("dataset_name", ""),
            }
            for layer_idx in range(baseline.shape[0]):
                metrics = state_shift_metrics(baseline[layer_idx], target[layer_idx])
                task_row = {"layer": layer_idx, **row_meta, **metrics}
                task_rows.append(task_row)
                layer_accumulator.setdefault((condition, layer_idx), []).append(metrics)

    if not task_rows:
        raise SystemExit("No comparable state pairs found.")

    layer_rows = aggregate_layer_rows(layer_accumulator)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "hidden_shift_by_task_layer.csv", task_rows)
    write_csv(output_dir / "hidden_shift_by_layer.csv", layer_rows)
    write_json(output_dir / "hidden_shift_summary.json", layer_rows)
    print(f"Wrote task-layer shifts to {output_dir / 'hidden_shift_by_task_layer.csv'}")
    print(f"Wrote layer summary to {output_dir / 'hidden_shift_by_layer.csv'}")
    print_best_shifts(layer_rows)


def group_by_task_condition(rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["task_id"]), {})[str(row["condition"])] = row
    return grouped


def load_state(row: dict) -> np.ndarray | None:
    raw_path = row.get("state_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.exists():
        return None
    try:
        arr = np.load(path)
        return np.asarray(arr["prompt_end_hidden"], dtype=np.float32)
    except Exception:
        return None


def pass_value(row: dict, pass_field: str | None = None) -> bool:
    if pass_field:
        return bool(row[pass_field])
    for field in ("hidden_passed", "evalplus_passed", "passed"):
        if field in row:
            return bool(row[field])
    return False


def infer_prompt_tests(row: dict) -> list:
    prompt_tests = row.get("prompt_tests")
    if prompt_tests is not None:
        return prompt_tests or []
    if str(row.get("condition")) == "nl_only":
        return []
    return row.get("visible_tests") or []


def state_shift_metrics(base: np.ndarray, target: np.ndarray) -> dict:
    base_norm = float(np.linalg.norm(base))
    target_norm = float(np.linalg.norm(target))
    diff = target - base
    l2 = float(np.linalg.norm(diff))
    denom = max(base_norm * target_norm, 1e-12)
    cosine_similarity = float(np.dot(base, target) / denom)
    return {
        "cosine_distance": 1.0 - cosine_similarity,
        "l2_distance": l2,
        "relative_l2_distance": l2 / max(base_norm, 1e-12),
        "baseline_norm": base_norm,
        "target_norm": target_norm,
        "norm_delta": target_norm - base_norm,
    }


def aggregate_layer_rows(layer_accumulator: dict[tuple[str, int], list[dict]]) -> list[dict]:
    rows: list[dict] = []
    metric_names = [
        "cosine_distance",
        "l2_distance",
        "relative_l2_distance",
        "baseline_norm",
        "target_norm",
        "norm_delta",
    ]
    for (condition, layer_idx), values in sorted(layer_accumulator.items()):
        row = {"condition": condition, "layer": layer_idx, "n": len(values)}
        for metric in metric_names:
            arr = np.asarray([item[metric] for item in values], dtype=np.float64)
            row[f"{metric}_mean"] = float(arr.mean())
            row[f"{metric}_std"] = float(arr.std(ddof=0))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    preferred = [
        "condition",
        "baseline_condition",
        "task_id",
        "layer",
        "n",
        "prompt_test_count",
        "baseline_passed",
        "target_passed",
    ]
    ordered = [field for field in preferred if field in fieldnames] + [
        field for field in fieldnames if field not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def print_best_shifts(rows: list[dict]) -> None:
    by_condition: dict[str, list[dict]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    print("\nLargest mean cosine shifts:")
    for condition, condition_rows in sorted(by_condition.items()):
        best = max(condition_rows, key=lambda row: row["cosine_distance_mean"])
        print(
            f"{condition:24s} layer={best['layer']} "
            f"cos_dist={best['cosine_distance_mean']:.4f} "
            f"rel_l2={best['relative_l2_distance_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
