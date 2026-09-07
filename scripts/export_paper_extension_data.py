#!/usr/bin/env python3
"""Export normalized task-level tables for the AAAI paper extension analyses."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    dataset: str
    model: str
    test_source: str
    eval_path: str
    generation_path: str
    pass_field: str
    conditions: tuple[str, ...]
    baseline_eval_path: str | None = None
    baseline_generation_path: str | None = None


@dataclass(frozen=True)
class ShiftSpec:
    run_id: str
    path: str
    condition_map: dict[str, str]


RUNS = (
    RunSpec(
        "q25_mbpp_original",
        "MBPP+",
        "Qwen2.5-Coder-7B-Instruct",
        "original",
        "outputs/mbppplus_qwen_full/eval_results.jsonl",
        "outputs/mbppplus_qwen_full/generations.jsonl",
        "hidden_passed",
        ("nl_only", "nl_tests", "shuffled_tests", "irrelevant_tests"),
    ),
    RunSpec(
        "q25_mbpp_synth_high3",
        "MBPP+",
        "Qwen2.5-Coder-7B-Instruct",
        "synthetic_high3",
        "outputs/mbppplus_qwen_synthetic_high3/eval_results.jsonl",
        "outputs/mbppplus_qwen_synthetic_high3/generations.jsonl",
        "hidden_passed",
        ("nl_only", "nl_tests"),
        "outputs/mbppplus_qwen_full/eval_results.jsonl",
        "outputs/mbppplus_qwen_full/generations.jsonl",
    ),
    RunSpec(
        "q25_humaneval_original",
        "HumanEval+",
        "Qwen2.5-Coder-7B-Instruct",
        "original",
        "outputs/humanevalplus_qwen_structured/eval_results.jsonl",
        "outputs/humanevalplus_qwen_structured/generations.jsonl",
        "hidden_passed",
        ("nl_only", "nl_tests", "shuffled_tests", "irrelevant_tests"),
    ),
    RunSpec(
        "q25_humaneval_synth_high3",
        "HumanEval+",
        "Qwen2.5-Coder-7B-Instruct",
        "synthetic_high3",
        "outputs/humanevalplus_qwen_synthetic_high3/eval_results.jsonl",
        "outputs/humanevalplus_qwen_synthetic_high3/generations.jsonl",
        "hidden_passed",
        ("nl_only", "nl_tests"),
        "outputs/humanevalplus_qwen_structured/eval_results.jsonl",
        "outputs/humanevalplus_qwen_structured/generations.jsonl",
    ),
    RunSpec(
        "q25_lcb_original",
        "LiveCodeBench-v6-new",
        "Qwen2.5-Coder-7B-Instruct",
        "original",
        "outputs/analysis/livecodebench_v6_new_eval_results_combined.jsonl",
        "outputs/livecodebench_v6_minus_v5_qwen_full_fixed/generations.jsonl",
        "passed",
        ("nl_only", "nl_tests", "shuffled_tests", "irrelevant_tests"),
    ),
    RunSpec(
        "q25_lcb_synth_high5",
        "LiveCodeBench-v6-new",
        "Qwen2.5-Coder-7B-Instruct",
        "synthetic_high5",
        "outputs/livecodebench_v6_new_qwen_synthetic_high5/eval_results_combined.jsonl",
        "outputs/livecodebench_v6_new_qwen_synthetic_high5/generations.jsonl",
        "passed",
        ("nl_only", "nl_tests", "shuffled_tests", "irrelevant_tests"),
    ),
    RunSpec(
        "q36_lcb_original",
        "LiveCodeBench-v6-new",
        "Qwen3.6-27B",
        "original",
        "outputs/livecodebench_v6_new_qwen36_27b_nothink/eval_results_combined.jsonl",
        "outputs/livecodebench_v6_new_qwen36_27b_nothink/generations.jsonl",
        "passed",
        ("nl_only", "nl_tests", "shuffled_tests", "irrelevant_tests"),
    ),
    RunSpec(
        "q36_lcb_synth_high5",
        "LiveCodeBench-v6-new",
        "Qwen3.6-27B",
        "synthetic_high5",
        "outputs/livecodebench_v6_new_qwen36_27b_synth_high5_nothink/eval_results_combined.jsonl",
        "outputs/livecodebench_v6_new_qwen36_27b_synth_high5_nothink/generations.jsonl",
        "passed",
        ("nl_only", "nl_tests", "shuffled_tests", "irrelevant_tests"),
    ),
)


SHIFTS = (
    ShiftSpec(
        "q25_mbpp_original",
        "outputs/analysis/test_mechanism/mbppplus_hidden_shift/hidden_shift_by_task_layer.csv",
        {
            "nl_tests": "nl_tests",
            "shuffled_tests": "shuffled_tests",
            "irrelevant_tests": "irrelevant_tests",
        },
    ),
    ShiftSpec(
        "q25_mbpp_synth_high3",
        "outputs/analysis/synthetic_tests/mbppplus_hidden_shift/hidden_shift_by_task_layer.csv",
        {"synth_high3_nl_tests": "nl_tests"},
    ),
    ShiftSpec(
        "q25_humaneval_original",
        "outputs/analysis/test_mechanism/humanevalplus_hidden_shift/hidden_shift_by_task_layer.csv",
        {
            "nl_tests": "nl_tests",
            "shuffled_tests": "shuffled_tests",
            "irrelevant_tests": "irrelevant_tests",
        },
    ),
    ShiftSpec(
        "q25_humaneval_synth_high3",
        "outputs/analysis/synthetic_tests/humanevalplus_hidden_shift/hidden_shift_by_task_layer.csv",
        {"synth_high3_nl_tests": "nl_tests"},
    ),
    ShiftSpec(
        "q25_lcb_original",
        "outputs/analysis/livecodebench_v6_new_hidden_shift/hidden_shift_by_task_layer.csv",
        {
            "nl_tests": "nl_tests",
            "shuffled_tests": "shuffled_tests",
            "irrelevant_tests": "irrelevant_tests",
        },
    ),
    ShiftSpec(
        "q25_lcb_synth_high5",
        "outputs/livecodebench_v6_new_qwen_synthetic_high5/hidden_shift/hidden_shift_by_task_layer.csv",
        {
            "nl_tests": "nl_tests",
            "shuffled_tests": "shuffled_tests",
            "irrelevant_tests": "irrelevant_tests",
        },
    ),
    ShiftSpec(
        "q36_lcb_original",
        "outputs/livecodebench_v6_new_qwen36_27b_nothink/hidden_shift/hidden_shift_by_task_layer.csv",
        {
            "nl_tests": "nl_tests",
            "shuffled_tests": "shuffled_tests",
            "irrelevant_tests": "irrelevant_tests",
        },
    ),
    ShiftSpec(
        "q36_lcb_synth_high5",
        "outputs/livecodebench_v6_new_qwen36_27b_synth_high5_nothink/hidden_shift/hidden_shift_by_task_layer.csv",
        {
            "nl_tests": "nl_tests",
            "shuffled_tests": "shuffled_tests",
            "irrelevant_tests": "irrelevant_tests",
        },
    ),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def as_int_bool(value: Any) -> int:
    if value in (True, 1, "1", "True", "true"):
        return 1
    if value in (False, 0, "0", "False", "false"):
        return 0
    raise ValueError(f"Expected a binary value, got {value!r}")


def optional_int(value: Any) -> int | str:
    return "" if value in (None, "") else int(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def indexed(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["task_id"]), str(row["condition"]))
        if key in result:
            raise ValueError(f"Duplicate task-condition row: {key}")
        result[key] = row
    return result


def export_correctness(
    repo: Path, output: Path
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for spec in RUNS:
        target_eval = indexed(read_jsonl(repo / spec.eval_path))
        target_gen = indexed(read_jsonl(repo / spec.generation_path))
        baseline_eval = (
            indexed(read_jsonl(repo / spec.baseline_eval_path))
            if spec.baseline_eval_path
            else target_eval
        )
        baseline_gen = (
            indexed(read_jsonl(repo / spec.baseline_generation_path))
            if spec.baseline_generation_path
            else target_gen
        )
        task_ids = sorted(
            {task_id for task_id, condition in baseline_eval if condition == "nl_only"}
        )
        for task_id in task_ids:
            for condition in spec.conditions:
                use_baseline = condition == "nl_only" and spec.baseline_eval_path
                eval_map = baseline_eval if use_baseline else target_eval
                gen_map = baseline_gen if use_baseline else target_gen
                key = (task_id, condition)
                if key not in eval_map or key not in gen_map:
                    raise ValueError(f"Missing {spec.run_id} row {key}")
                evaluation = eval_map[key]
                generation = gen_map[key]
                prompt = str(generation.get("prompt", ""))
                completion = str(generation.get("completion", ""))
                row = {
                    "task_id": task_id,
                    "dataset": spec.dataset,
                    "model": spec.model,
                    "run_id": spec.run_id,
                    "test_source": spec.test_source,
                    "condition": condition,
                    "passed": as_int_bool(evaluation[spec.pass_field]),
                    "difficulty": evaluation.get("difficulty") or "unknown",
                    "platform": evaluation.get("platform") or "unknown",
                    "prompt_chars": len(prompt),
                    "prompt_tokens": optional_int(
                        generation.get("prompt_token_count")
                    ),
                    "completion_chars": len(completion),
                    "source_eval_file": (
                        spec.baseline_eval_path
                        if use_baseline
                        else spec.eval_path
                    ),
                    "source_generation_file": (
                        spec.baseline_generation_path
                        if use_baseline
                        else spec.generation_path
                    ),
                }
                lookup[(spec.run_id, task_id, condition)] = row
                rows.append(row)
    rows.sort(key=lambda row: (row["dataset"], row["model"], row["run_id"], row["task_id"], row["condition"]))
    fields = [
        "task_id",
        "dataset",
        "model",
        "run_id",
        "test_source",
        "condition",
        "passed",
        "difficulty",
        "platform",
        "prompt_chars",
        "prompt_tokens",
        "completion_chars",
        "source_eval_file",
        "source_generation_file",
    ]
    write_csv(output / "task_level_correctness.csv", rows, fields)
    return rows, lookup


def export_shifts(
    repo: Path,
    output: Path,
    correctness: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_specs = {spec.run_id: spec for spec in RUNS}
    layer_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for shift_spec in SHIFTS:
        run = run_specs[shift_spec.run_id]
        source_path = repo / shift_spec.path
        with source_path.open(newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                if source["condition"] not in shift_spec.condition_map:
                    continue
                condition = shift_spec.condition_map[source["condition"]]
                task_id = str(source["task_id"])
                baseline = correctness[(run.run_id, task_id, "nl_only")]
                target = correctness[(run.run_id, task_id, condition)]
                baseline_passed = as_int_bool(source["baseline_passed"])
                target_passed = as_int_bool(source["target_passed"])
                if baseline_passed != baseline["passed"] or target_passed != target["passed"]:
                    raise ValueError(
                        f"Pass mismatch in {shift_spec.path}: "
                        f"{run.run_id}/{task_id}/{condition}"
                    )
                baseline_tokens = baseline["prompt_tokens"]
                target_tokens = target["prompt_tokens"]
                added_tokens: int | str = ""
                if baseline_tokens != "" and target_tokens != "":
                    added_tokens = int(target_tokens) - int(baseline_tokens)
                row = {
                    "task_id": task_id,
                    "dataset": run.dataset,
                    "model": run.model,
                    "run_id": run.run_id,
                    "test_source": run.test_source,
                    "condition": condition,
                    "baseline_condition": "nl_only",
                    "layer": int(source["layer"]),
                    "cosine_distance": float(source["cosine_distance"]),
                    "baseline_passed": baseline_passed,
                    "target_passed": target_passed,
                    "difficulty": target["difficulty"],
                    "platform": target["platform"],
                    "baseline_prompt_chars": baseline["prompt_chars"],
                    "target_prompt_chars": target["prompt_chars"],
                    "added_prompt_chars": target["prompt_chars"]
                    - baseline["prompt_chars"],
                    "baseline_prompt_tokens": baseline_tokens,
                    "target_prompt_tokens": target_tokens,
                    "added_prompt_tokens": added_tokens,
                    "source_shift_file": shift_spec.path,
                }
                layer_rows.append(row)
                grouped[(run.run_id, task_id, condition)].append(row)

    layer_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["model"],
            row["run_id"],
            row["task_id"],
            row["condition"],
            row["layer"],
        )
    )
    layer_fields = [
        "task_id",
        "dataset",
        "model",
        "run_id",
        "test_source",
        "condition",
        "baseline_condition",
        "layer",
        "cosine_distance",
        "baseline_passed",
        "target_passed",
        "difficulty",
        "platform",
        "baseline_prompt_chars",
        "target_prompt_chars",
        "added_prompt_chars",
        "baseline_prompt_tokens",
        "target_prompt_tokens",
        "added_prompt_tokens",
        "source_shift_file",
    ]
    write_csv(output / "hidden_shift_by_task_layer.csv", layer_rows, layer_fields)

    summary_rows: list[dict[str, Any]] = []
    flip_names = {
        (0, 0): "stable_fail",
        (0, 1): "rescue",
        (1, 0): "harm",
        (1, 1): "stable_pass",
    }
    for _, values in grouped.items():
        values.sort(key=lambda row: row["layer"])
        non_embedding = [row for row in values if row["layer"] > 0]
        analysis_values = non_embedding or values
        peak = max(analysis_values, key=lambda row: row["cosine_distance"])
        final = max(values, key=lambda row: row["layer"])
        first = values[0]
        summary_rows.append(
            {
                "task_id": first["task_id"],
                "dataset": first["dataset"],
                "model": first["model"],
                "run_id": first["run_id"],
                "test_source": first["test_source"],
                "condition": first["condition"],
                "baseline_passed": first["baseline_passed"],
                "target_passed": first["target_passed"],
                "flip_type": flip_names[
                    (first["baseline_passed"], first["target_passed"])
                ],
                "difficulty": first["difficulty"],
                "platform": first["platform"],
                "shift_mean_nonembedding": fmean(
                    row["cosine_distance"] for row in analysis_values
                ),
                "shift_final": final["cosine_distance"],
                "shift_peak": peak["cosine_distance"],
                "shift_peak_layer": peak["layer"],
                "baseline_prompt_chars": first["baseline_prompt_chars"],
                "target_prompt_chars": first["target_prompt_chars"],
                "added_prompt_chars": first["added_prompt_chars"],
                "baseline_prompt_tokens": first["baseline_prompt_tokens"],
                "target_prompt_tokens": first["target_prompt_tokens"],
                "added_prompt_tokens": first["added_prompt_tokens"],
            }
        )
    summary_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["model"],
            row["run_id"],
            row["task_id"],
            row["condition"],
        )
    )
    summary_fields = [
        "task_id",
        "dataset",
        "model",
        "run_id",
        "test_source",
        "condition",
        "baseline_passed",
        "target_passed",
        "flip_type",
        "difficulty",
        "platform",
        "shift_mean_nonembedding",
        "shift_final",
        "shift_peak",
        "shift_peak_layer",
        "baseline_prompt_chars",
        "target_prompt_chars",
        "added_prompt_chars",
        "baseline_prompt_tokens",
        "target_prompt_tokens",
        "added_prompt_tokens",
    ]
    write_csv(output / "task_level_analysis.csv", summary_rows, summary_fields)
    return layer_rows, summary_rows


def export_probes(repo: Path, output: Path) -> list[dict[str, Any]]:
    mappings = {
        "mbppplus": (
            "MBPP+",
            "Qwen2.5-Coder-7B-Instruct",
            "q25_mbpp_original",
        ),
        "humanevalplus": (
            "HumanEval+",
            "Qwen2.5-Coder-7B-Instruct",
            "q25_humaneval_original",
        ),
        "qwen36_lcb": (
            "LiveCodeBench-v6-new",
            "Qwen3.6-27B",
            "q36_lcb_original",
        ),
    }
    rows: list[dict[str, Any]] = []
    probe_dir = repo / "outputs/analysis/probes"
    for path in sorted(probe_dir.glob("*.csv")):
        prefix = next((name for name in mappings if path.stem.startswith(name + "_")), None)
        if prefix is None:
            continue
        target = path.stem[len(prefix) + 1 :]
        dataset, model, run_id = mappings[prefix]
        with path.open(newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "run_id": run_id,
                        "target": target,
                        "layer": int(source["layer"]),
                        "accuracy": float(source["accuracy"]),
                        "macro_f1": float(source["macro_f1"]),
                        "auc": source["auc"],
                        "fold": "",
                        "source_file": str(path.relative_to(repo)),
                    }
                )
    fields = [
        "dataset",
        "model",
        "run_id",
        "target",
        "layer",
        "fold",
        "accuracy",
        "macro_f1",
        "auc",
        "source_file",
    ]
    write_csv(output / "probe_layer_metrics.csv", rows, fields)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    correctness_rows, correctness = export_correctness(repo, output)
    layer_rows, summary_rows = export_shifts(repo, output, correctness)
    probe_rows = export_probes(repo, output)
    print(f"Wrote {len(correctness_rows)} correctness rows")
    print(f"Wrote {len(layer_rows)} task-layer shift rows")
    print(f"Wrote {len(summary_rows)} task-level analysis rows")
    print(f"Wrote {len(probe_rows)} probe-layer rows")


if __name__ == "__main__":
    main()
