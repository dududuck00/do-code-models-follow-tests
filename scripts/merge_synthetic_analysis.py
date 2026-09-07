#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.io import append_jsonl, read_jsonl


DATASET_CONFIGS = {
    "mbppplus": {
        "base_eval": Path("outputs/mbppplus_qwen_full/eval_results.jsonl"),
        "base_summary": Path("outputs/mbppplus_qwen_full/summary.csv"),
        "synth_data_dir": Path("data/synthetic_public_tests/mbppplus"),
        "synth_output_prefix": "outputs/mbppplus_qwen_synthetic_",
    },
    "humanevalplus": {
        "base_eval": Path("outputs/humanevalplus_qwen_structured/eval_results.jsonl"),
        "base_summary": Path("outputs/humanevalplus_qwen_structured/summary.csv"),
        "synth_data_dir": Path("data/synthetic_public_tests/humanevalplus"),
        "synth_output_prefix": "outputs/humanevalplus_qwen_synthetic_",
    },
}

BASE_CONDITION_NAMES = {
    "nl_only": "nl_only",
    "nl_tests": "orig_nl_tests",
    "shuffled_tests": "orig_shuffled_tests",
    "irrelevant_tests": "orig_irrelevant_tests",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge synthetic public-test experiment outputs into analysis-ready tables."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mbppplus", "humanevalplus"],
        choices=sorted(DATASET_CONFIGS),
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["high1", "high2", "high3", "low3", "random3", "diverse3"],
        help="Synthetic settings to merge, e.g. high1 high2 high3 low3 random3 diverse3.",
    )
    parser.add_argument("--conditions", nargs="+", default=["nl_tests", "asserts_only"])
    parser.add_argument("--output-dir", default="outputs/analysis/synthetic_tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows: list[dict] = []
    all_coverage_rows: list[dict] = []
    for dataset in args.datasets:
        config = DATASET_CONFIGS[dataset]
        summary_rows, coverage_rows = merge_dataset(
            dataset=dataset,
            config=config,
            settings=args.settings,
            conditions=args.conditions,
            output_dir=output_dir,
        )
        all_summary_rows.extend(summary_rows)
        all_coverage_rows.extend(coverage_rows)

    write_csv(output_dir / "synthetic_summary.csv", all_summary_rows)
    write_csv(output_dir / "synthetic_coverage.csv", all_coverage_rows)
    print(f"Wrote combined summary to {output_dir / 'synthetic_summary.csv'}")
    print(f"Wrote coverage table to {output_dir / 'synthetic_coverage.csv'}")


def merge_dataset(
    dataset: str,
    config: dict,
    settings: list[str],
    conditions: list[str],
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    base_eval = Path(config["base_eval"])
    base_summary = Path(config["base_summary"])
    if not base_eval.exists():
        raise SystemExit(f"Missing base eval results for {dataset}: {base_eval}")
    if not base_summary.exists():
        raise SystemExit(f"Missing base summary for {dataset}: {base_summary}")

    tagged_path = output_dir / f"{dataset}_synthetic_eval_results_tagged.jsonl"
    if tagged_path.exists():
        tagged_path.unlink()

    base_summary_rows = read_summary(base_summary)
    base_nl_only = hidden_rate(base_summary_rows, "nl_only")
    base_orig_tests = hidden_rate(base_summary_rows, "nl_tests")
    summary_rows = base_summary_for_output(
        dataset=dataset,
        base_summary_rows=base_summary_rows,
        base_nl_only=base_nl_only,
        base_orig_tests=base_orig_tests,
    )

    for row in read_jsonl(base_eval):
        condition = str(row["condition"])
        if condition not in BASE_CONDITION_NAMES:
            continue
        tagged = dict(row)
        tagged["source_condition"] = condition
        tagged["condition"] = BASE_CONDITION_NAMES[condition]
        tagged["synthetic_setting"] = ""
        tagged["synthetic_selector"] = ""
        tagged["synthetic_count"] = ""
        append_jsonl(tagged_path, tagged)

    coverage_rows: list[dict] = []
    for setting in settings:
        selector, count = parse_setting(setting)
        coverage = synthetic_coverage(
            dataset=dataset,
            setting=setting,
            synth_data_dir=Path(config["synth_data_dir"]),
        )
        coverage_rows.append(coverage)

        synth_output = Path(str(config["synth_output_prefix"]) + setting)
        summary_path = synth_output / "summary.csv"
        eval_path = synth_output / "eval_results.jsonl"
        if not summary_path.exists():
            raise SystemExit(f"Missing synthetic summary for {dataset}/{setting}: {summary_path}")
        if not eval_path.exists():
            raise SystemExit(f"Missing synthetic eval results for {dataset}/{setting}: {eval_path}")

        synth_summary_rows = read_summary(summary_path)
        for condition in conditions:
            row = synth_summary_rows.get(condition)
            if not row:
                continue
            visible = float(row["visible_pass_rate"])
            hidden = float(row["hidden_pass_rate"])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "selector": selector,
                    "count": count,
                    "condition": condition,
                    "tagged_condition": tagged_condition(setting, condition),
                    "n": int(row["n"]),
                    "visible_pass_rate": visible,
                    "hidden_pass_rate": hidden,
                    "delta_vs_nl_only": hidden - base_nl_only,
                    "delta_vs_original_nl_tests": hidden - base_orig_tests,
                    **coverage_summary_fields(coverage),
                }
            )

        for row in read_jsonl(eval_path):
            source_condition = str(row["condition"])
            if source_condition not in conditions:
                continue
            tagged = dict(row)
            tagged["source_condition"] = source_condition
            tagged["condition"] = tagged_condition(setting, source_condition)
            tagged["synthetic_setting"] = setting
            tagged["synthetic_selector"] = selector
            tagged["synthetic_count"] = count
            append_jsonl(tagged_path, tagged)

    write_csv(output_dir / f"{dataset}_synthetic_summary.csv", summary_rows)
    print(f"Wrote tagged eval rows to {tagged_path}")
    print(f"Wrote dataset summary to {output_dir / f'{dataset}_synthetic_summary.csv'}")
    return summary_rows, coverage_rows


def read_summary(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["condition"]: row for row in csv.DictReader(handle)}


def hidden_rate(rows: dict[str, dict], condition: str) -> float:
    if condition not in rows:
        raise SystemExit(f"Missing condition in summary: {condition}")
    return float(rows[condition]["hidden_pass_rate"])


def base_summary_for_output(
    dataset: str,
    base_summary_rows: dict[str, dict],
    base_nl_only: float,
    base_orig_tests: float,
) -> list[dict]:
    rows = []
    for source_condition, tagged in BASE_CONDITION_NAMES.items():
        row = base_summary_rows.get(source_condition)
        if not row:
            continue
        hidden = float(row["hidden_pass_rate"])
        rows.append(
            {
                "dataset": dataset,
                "setting": "original",
                "selector": "original",
                "count": "",
                "condition": source_condition,
                "tagged_condition": tagged,
                "n": int(row["n"]),
                "visible_pass_rate": float(row["visible_pass_rate"]),
                "hidden_pass_rate": hidden,
                "delta_vs_nl_only": hidden - base_nl_only,
                "delta_vs_original_nl_tests": hidden - base_orig_tests,
                "fallback_tasks": "",
                "fallback_rate": "",
                "min_prompt_tests": "",
                "max_prompt_tests": "",
            }
        )
    return rows


def parse_setting(setting: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z_]+)(\d+)", setting)
    if not match:
        raise SystemExit(f"Setting must look like high3 or random1: {setting}")
    return match.group(1), int(match.group(2))


def tagged_condition(setting: str, condition: str) -> str:
    return f"synth_{setting}_{condition}"


def synthetic_coverage(dataset: str, setting: str, synth_data_dir: Path) -> dict:
    path = synth_data_dir / f"{dataset}_synth_{setting}.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing synthetic dataset variant: {path}")
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"No rows found in synthetic dataset variant: {path}")
    fallback = 0
    lengths = []
    valid_counts = []
    for row in rows:
        tests = row.get("synthetic_public_tests") or []
        lengths.append(len(tests))
        metadata = row.get("synthetic_public_tests_metadata") or {}
        fallback += int(bool(metadata.get("fallback_public_used")))
        valid_count = metadata.get("valid_count")
        if valid_count not in (None, ""):
            valid_counts.append(int(valid_count))
    selector, count = parse_setting(setting)
    return {
        "dataset": dataset,
        "setting": setting,
        "selector": selector,
        "count": count,
        "tasks": len(rows),
        "fallback_tasks": fallback,
        "fallback_rate": fallback / len(rows),
        "min_prompt_tests": min(lengths),
        "max_prompt_tests": max(lengths),
        "avg_valid_count": sum(valid_counts) / len(valid_counts) if valid_counts else "",
    }


def coverage_summary_fields(coverage: dict) -> dict:
    return {
        "fallback_tasks": coverage["fallback_tasks"],
        "fallback_rate": coverage["fallback_rate"],
        "min_prompt_tests": coverage["min_prompt_tests"],
        "max_prompt_tests": coverage["max_prompt_tests"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit(f"No rows to write: {path}")
    fieldnames = sorted({field for row in rows for field in row})
    preferred = [
        "dataset",
        "setting",
        "selector",
        "count",
        "condition",
        "tagged_condition",
        "n",
        "visible_pass_rate",
        "hidden_pass_rate",
        "delta_vs_nl_only",
        "delta_vs_original_nl_tests",
        "fallback_tasks",
        "fallback_rate",
        "min_prompt_tests",
        "max_prompt_tests",
    ]
    ordered = [field for field in preferred if field in fieldnames] + [
        field for field in fieldnames if field not in preferred
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
