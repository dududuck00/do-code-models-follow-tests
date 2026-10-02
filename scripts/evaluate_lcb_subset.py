#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LiveCodeBench subset with the official LiveCodeBench evaluator."
    )
    parser.add_argument("--livecodebench-root", default="third_party/LiveCodeBench")
    parser.add_argument("--dataset", default="data/livecodebench_release_v6_minus_v5.jsonl")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-process-evaluate", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pipe-workers", action="store_true",
                        help="Use pipe-connected workers with the same official scoring function.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_lcb_to_path(args.livecodebench_root)

    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation import codegen_metrics, extract_instance_results

    rows = read_jsonl(args.dataset)
    by_id = {row["question_id"]: row for row in rows}

    gen_rows = [
        row for row in read_jsonl(args.generations) if row.get("condition") == args.condition
    ]
    if not gen_rows:
        raise SystemExit(f"No generation rows found for condition: {args.condition}")

    missing = [row["task_id"] for row in gen_rows if row["task_id"] not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} generated task ids are missing from dataset, first={missing[:3]}")

    gen_rows = sorted(gen_rows, key=lambda row: str(row["task_id"]))
    problems = [CodeGenerationProblem(**official_problem_fields(by_id[row["task_id"]])) for row in gen_rows]
    samples = [problem.get_evaluation_sample() for problem in problems]
    generations = [[row["candidate_code"]] for row in gen_rows]

    if args.pipe_workers:
        metrics = pipe_metrics(samples, generations, args)
    else:
        metrics = codegen_metrics(
            samples,
            generations,
            k_list=[1],
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.timeout,
            debug=args.debug,
        )
    graded = extract_instance_results(metrics[1])
    metadata = metrics[2]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"{args.condition}_eval_all.jsonl"
    summary_path = output_dir / f"{args.condition}_eval_summary.json"

    detail_rows = []
    for problem, row, grade_list, meta_list in zip(problems, gen_rows, graded, metadata):
        passed = bool(grade_list[0])
        detail_rows.append(
            {
                "task_id": row["task_id"],
                "question_id": problem.question_id,
                "condition": args.condition,
                "passed": passed,
                "graded_list": grade_list,
                "metadata": meta_list,
                "state_path": row.get("state_path"),
                "difficulty": problem.difficulty.value,
                "platform": problem.platform.value,
                "contest_date": problem.contest_date.isoformat(),
            }
        )

    with detail_path.open("w", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "condition": args.condition,
        "num_tasks": len(detail_rows),
        "pass_at_1": metrics[0].get("pass@1"),
        "num_passed": sum(row["passed"] for row in detail_rows),
        "num_failed": sum(not row["passed"] for row in detail_rows),
        "by_difficulty": pass_rate_by_key(detail_rows, "difficulty"),
        "by_platform": pass_rate_by_key(detail_rows, "platform"),
        "metrics": metrics[0],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


def add_lcb_to_path(livecodebench_root: str) -> None:
    root = Path(livecodebench_root).resolve()
    if not (root / "lcb_runner").exists():
        raise SystemExit(f"LiveCodeBench root does not contain lcb_runner: {root}")
    sys.path.insert(0, str(root))


def pipe_metrics(samples, generations, args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tddexp.lcb_execution import grade
    from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results
    results = {}
    metadata = {}
    with ThreadPoolExecutor(max_workers=args.num_process_evaluate) as pool:
        futures = {pool.submit(grade, sample, code[0], args.livecodebench_root,
                               args.timeout, memory_limit_bytes=None): i
                   for i, (sample, code) in enumerate(zip(samples, generations))}
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            scores = result.get('scores')
            if scores is None:
                scores = [-1] if result['status'] == 'timeout' else [-2]
            results[i] = [scores]
            metadata[i] = [json.dumps(result.get('metadata', result), default=str)]
            print(f"Evaluated {len(results)}/{len(samples)}", flush=True)
    results = dict(sorted(results.items()))
    return [compute_metrics_from_results(results, k_list=[1]), results,
            [metadata[i] for i in sorted(metadata)]]


def official_problem_fields(row: dict) -> dict:
    keys = {
        "question_title",
        "question_content",
        "platform",
        "question_id",
        "contest_id",
        "contest_date",
        "starter_code",
        "difficulty",
        "public_test_cases",
        "private_test_cases",
        "metadata",
    }
    return {key: row[key] for key in keys if key in row}


def pass_rate_by_key(rows: list[dict], key: str) -> dict[str, dict]:
    counts: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] += 1
        if row["passed"]:
            passed[value] += 1
    return {
        key_value: {
            "n": counts[key_value],
            "passed": passed[key_value],
            "pass_rate": passed[key_value] / counts[key_value],
        }
        for key_value in sorted(counts)
    }


if __name__ == "__main__":
    main()
