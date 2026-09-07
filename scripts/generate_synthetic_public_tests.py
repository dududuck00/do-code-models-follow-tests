#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import re
import socket
import sys
import time
from urllib import request, error

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.data import CodeTask, load_tasks
from tddexp.evaluation import evaluate_code
from tddexp.io import read_jsonl, append_jsonl
from tddexp.prompts import select_quality_tests, stable_seed, test_quality_score


DEFAULT_API_KEY = ""
DEFAULT_BASE_URL = "https://api.deepseek.com"

DEFAULT_MODEL = "deepseek-v4-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic public tests with a strong OpenAI-compatible model, "
            "validate them with canonical solutions, and write dataset variants."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url-env", default="DEEPSEEK_BASE_URL")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--num-candidates", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent worker threads for API calls and test validation.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not call API; use original public tests as mock candidates.")
    parser.add_argument("--validation-timeout", type=float, default=5.0)
    parser.add_argument("--tmp-root", default=".tmp")
    parser.add_argument("--max-test-chars", type=int, default=500)
    parser.add_argument(
        "--weak-generations",
        action="append",
        default=[],
        help="Optional eval_results/generations JSONL containing weak candidate_code implementations.",
    )
    parser.add_argument("--max-weak-per-task", type=int, default=4)
    parser.add_argument("--kill-rate-weight", type=float, default=10.0)
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--selectors", nargs="+", default=["high", "low", "diverse", "random"])
    parser.add_argument("--fallback-public", action="store_true", default=True)
    parser.add_argument("--no-fallback-public", dest="fallback_public", action="store_false")
    parser.add_argument("--include-original-duplicates", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", help="Abort when one API request fails after retries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.dataset)
    tasks = load_tasks(args.dataset, dataset_name=args.dataset_name)
    tasks_by_id = {task.task_id: task for task in tasks}
    if args.limit is not None:
        rows = rows[: args.limit]

    api_key = os.environ.get(args.api_key_env, DEFAULT_API_KEY)
    base_url = os.environ.get(args.base_url_env, args.base_url).rstrip("/")
    if not args.dry_run and not api_key:
        raise SystemExit(
            f"Missing API key. Set {args.api_key_env}=... before running this script."
        )
    weak_codes = load_weak_candidate_codes(
        paths=[Path(path) for path in args.weak_generations],
        max_per_task=args.max_weak_per_task,
    )

    candidates_path = output_dir / "synthetic_test_candidates.jsonl"
    seen_results = load_existing_results(candidates_path) if args.resume else {}
    if candidates_path.exists() and not args.resume:
        candidates_path.unlink()

    results: dict[str, dict] = {}
    pending_jobs: list[tuple[dict, CodeTask]] = []
    for row in rows:
        task_id = str(row["task_id"])
        task = tasks_by_id.get(task_id)
        if task is None:
            continue
        if task_id in seen_results:
            results[task_id] = seen_results[task_id]
            continue
        pending_jobs.append((row, task))

    worker_count = max(1, args.workers)
    if worker_count == 1 or len(pending_jobs) <= 1:
        iterator = tqdm(pending_jobs, desc="synthetic tests", unit="task")
        for row, task in iterator:
            result = process_task(
                row=row,
                task=task,
                args=args,
                base_url=base_url,
                api_key=api_key,
                weak_candidate_codes=weak_codes.get(str(row["task_id"]), []),
            )
            maybe_log_task_warning(result)
            append_jsonl(candidates_path, result)
            results[result["task_id"]] = result
            if args.sleep > 0 and not args.dry_run:
                time.sleep(args.sleep)
    else:
        print(f"Using {worker_count} worker threads for {len(pending_jobs)} pending tasks.")
        futures = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for row, task in pending_jobs:
                future = executor.submit(
                    process_task,
                    row=row,
                    task=task,
                    args=args,
                    base_url=base_url,
                    api_key=api_key,
                    weak_candidate_codes=weak_codes.get(str(row["task_id"]), []),
                )
                futures.append(future)
                if args.sleep > 0 and not args.dry_run:
                    time.sleep(args.sleep)
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"synthetic tests x{worker_count}",
                unit="task",
            ):
                result = future.result()
                maybe_log_task_warning(result)
                append_jsonl(candidates_path, result)
                results[result["task_id"]] = result

    write_dataset_variants(
        source_rows=read_jsonl(args.dataset),
        tasks_by_id=tasks_by_id,
        results=results,
        output_dir=output_dir,
        dataset_stem=Path(args.dataset).stem,
        selectors=args.selectors,
        counts=args.counts,
        fallback_public=args.fallback_public,
        model=args.model,
    )
    write_report(output_dir / "synthetic_test_report.csv", results)
    print(f"Wrote candidates to {candidates_path}")
    print(f"Wrote dataset variants to {output_dir}")


def process_task(
    row: dict,
    task: CodeTask,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
    weak_candidate_codes: list[str],
) -> dict:
    task_id = str(row["task_id"])
    raw_response = ""
    api_error = ""
    if args.dry_run:
        generated_tests = list(task.tests)
        raw_response = "<dry-run>"
    else:
        try:
            raw_response = call_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                messages=build_messages(task, args.num_candidates),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
            generated_tests = parse_tests_from_response(raw_response)
        except Exception as exc:
            if args.fail_fast:
                raise
            generated_tests = []
            api_error = repr(exc)

    try:
        result = validate_and_score_tests(
            task=task,
            generated_tests=generated_tests,
            original_public_tests=task.tests,
            validation_timeout=args.validation_timeout,
            tmp_root=args.tmp_root,
            max_test_chars=args.max_test_chars,
            include_original_duplicates=args.include_original_duplicates,
            weak_candidate_codes=weak_candidate_codes,
            kill_rate_weight=args.kill_rate_weight,
        )
    except Exception as exc:
        if args.fail_fast:
            raise
        result = empty_failed_result()
        api_error = f"{api_error}; validation_failed={exc!r}" if api_error else f"validation_failed={exc!r}"

    result.update(
        {
            "task_id": task_id,
            "dataset_name": args.dataset_name,
            "model": args.model,
            "raw_response": raw_response,
            "api_error": api_error,
            "num_requested": args.num_candidates,
        }
    )
    return result


def empty_failed_result() -> dict:
    return {
        "generated_count": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "valid_tests": [],
        "invalid_tests": [],
    }


def maybe_log_task_warning(result: dict) -> None:
    api_error = result.get("api_error")
    if api_error:
        tqdm.write(f"[warn] task_id={result.get('task_id')} failed: {api_error}")


def load_existing_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {str(row["task_id"]): row for row in read_jsonl(path)}


def build_messages(task: CodeTask, num_candidates: int) -> list[dict]:
    original_tests = "\n".join(f"- {test}" for test in task.tests[:3])
    user = f"""
Generate {num_candidates} high-quality Python assert test cases for this programming problem.

Requirements:
- Return only valid JSON with this schema: {{"tests": ["assert ...", "..."]}}.
- Each test must be a single deterministic Python assert statement.
- Each test must call the target function `{task.entry_point}`.
- Do not include imports, helper functions, prose, markdown fences, or property-based tests.
- Prefer boundary cases, corner cases, diverse input structures, and cases that distinguish plausible wrong implementations.
- Avoid duplicating the existing public tests exactly.

Problem prompt:
{task.prompt}

Existing public tests:
{original_tests}
""".strip()
    return [
        {
            "role": "system",
            "content": (
                "You are a careful software testing assistant. "
                "You write concise, executable Python assert statements."
            ),
        },
        {"role": "user", "content": user},
    ]


def call_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = None
    for attempt in range(retries):
        req = request.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            if not str(content).strip():
                raise RuntimeError("empty chat completion content")
            return content
        except (TimeoutError, socket.timeout, error.URLError, error.HTTPError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if isinstance(exc, error.HTTPError):
                try:
                    body = exc.read().decode("utf-8", errors="replace")[:500]
                    last_error = RuntimeError(f"HTTP {exc.code}: {body}")
                except Exception:
                    last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Chat completion failed after {retries} retries: {last_error}")


def parse_tests_from_response(text: str) -> list[str]:
    payload = extract_json_payload(text)
    tests: list[str] = []
    if isinstance(payload, dict) and isinstance(payload.get("tests"), list):
        tests.extend(str(item).strip() for item in payload["tests"])
    elif isinstance(payload, list):
        tests.extend(str(item).strip() for item in payload)
    if not tests:
        tests.extend(extract_assert_lines(text))
    return [test for test in tests if test]


def extract_json_payload(text: str) -> object | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    for candidate in (stripped, extract_braced_block(stripped)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def extract_braced_block(text: str) -> str:
    start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not start_candidates:
        return ""
    start = min(start_candidates)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return ""
    return text[start : end + 1]


def extract_assert_lines(text: str) -> list[str]:
    tests = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("-*0123456789. ")
        if stripped.startswith("assert "):
            tests.append(stripped)
    return tests


def validate_and_score_tests(
    task: CodeTask,
    generated_tests: list[str],
    original_public_tests: list[str],
    validation_timeout: float,
    tmp_root: str,
    max_test_chars: int,
    include_original_duplicates: bool,
    weak_candidate_codes: list[str],
    kill_rate_weight: float,
) -> dict:
    original_norms = {normalize_test(test) for test in original_public_tests}
    seen = set()
    valid = []
    invalid = []
    for test in generated_tests:
        cleaned = clean_test(test)
        reason = reject_reason(cleaned, task.entry_point, max_test_chars)
        norm = normalize_test(cleaned)
        if not reason and norm in seen:
            reason = "duplicate_generated"
        if not reason and not include_original_duplicates and norm in original_norms:
            reason = "duplicate_original_public"
        if reason:
            invalid.append({"test": cleaned, "reason": reason})
            continue
        eval_result = evaluate_code(
            task.canonical_code,
            [cleaned],
            timeout=validation_timeout,
            tmp_root=tmp_root,
        )
        if not eval_result.passed:
            invalid.append({"test": cleaned, "reason": "canonical_failed", "error": eval_result.error})
            continue
        seen.add(norm)
        heuristic_score = test_quality_score(cleaned)
        kill_rate, killed_count = score_kill_rate(
            test=cleaned,
            weak_candidate_codes=weak_candidate_codes,
            validation_timeout=validation_timeout,
            tmp_root=tmp_root,
        )
        valid.append(
            {
                "test": cleaned,
                "quality_score": heuristic_score + kill_rate_weight * kill_rate,
                "heuristic_score": heuristic_score,
                "kill_rate": kill_rate,
                "killed_weak_count": killed_count,
                "weak_count": len(weak_candidate_codes),
            }
        )
    return {
        "generated_count": len(generated_tests),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "valid_tests": valid,
        "invalid_tests": invalid,
    }


def load_weak_candidate_codes(paths: list[Path], max_per_task: int) -> dict[str, list[str]]:
    weak: dict[str, list[str]] = {}
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Weak generations file does not exist: {path}")
        for row in read_jsonl(path):
            if row.get("hidden_passed") is True or row.get("evalplus_passed") is True or row.get("passed") is True:
                continue
            code = str(row.get("candidate_code") or "").strip()
            if not code:
                continue
            task_id = str(row["task_id"])
            bucket = weak.setdefault(task_id, [])
            if len(bucket) < max_per_task and code not in bucket:
                bucket.append(code)
    return weak


def score_kill_rate(
    test: str,
    weak_candidate_codes: list[str],
    validation_timeout: float,
    tmp_root: str,
) -> tuple[float, int]:
    if not weak_candidate_codes:
        return 0.0, 0
    killed = 0
    for code in weak_candidate_codes:
        result = evaluate_code(
            code,
            [test],
            timeout=validation_timeout,
            tmp_root=tmp_root,
        )
        if not result.passed:
            killed += 1
    return killed / len(weak_candidate_codes), killed


def clean_test(test: str) -> str:
    test = test.strip()
    if test.startswith("```"):
        test = re.sub(r"^```(?:python)?", "", test, flags=re.IGNORECASE).strip()
        test = re.sub(r"```$", "", test).strip()
    return test.splitlines()[0].strip() if "\n" in test else test


def reject_reason(test: str, entry_point: str, max_test_chars: int) -> str:
    if not test.startswith("assert "):
        return "not_assert"
    if len(test) > max_test_chars:
        return "too_long"
    if entry_point and entry_point not in test:
        return "missing_entry_point"
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return "syntax_error"
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return "not_single_assert"
    return ""


def normalize_test(test: str) -> str:
    return re.sub(r"\s+", "", test)


def write_dataset_variants(
    source_rows: list[dict],
    tasks_by_id: dict[str, CodeTask],
    results: dict[str, dict],
    output_dir: Path,
    dataset_stem: str,
    selectors: list[str],
    counts: list[int],
    fallback_public: bool,
    model: str,
) -> None:
    for selector in selectors:
        for count in counts:
            path = output_dir / f"{dataset_stem}_synth_{selector}{count}.jsonl"
            if path.exists():
                path.unlink()
            for row in source_rows:
                task_id = str(row["task_id"])
                task = tasks_by_id.get(task_id)
                result = results.get(task_id, {})
                candidates = result.get("valid_tests", [])
                valid_tests = [item["test"] for item in candidates]
                selected = select_tests(valid_tests, selector, count, task_id,
                                        scores={item["test"]: item["quality_score"] for item in candidates})
                fallback_used = False
                if fallback_public and task is not None and len(selected) < count:
                    selected = fill_with_public_tests(selected, task.tests, count)
                    fallback_used = len(selected) >= count
                new_row = dict(row)
                new_row["synthetic_public_tests"] = selected
                new_row["synthetic_public_tests_metadata"] = {
                    "selector": selector,
                    "selection_metric": "quality_score" if selector in {"high", "low"} else selector,
                    "count": count,
                    "model": model,
                    "valid_count": len(valid_tests),
                    "fallback_public_used": fallback_used,
                }
                append_jsonl(path, new_row)


def select_tests(tests: list[str], selector: str, count: int, task_id: str,
                 scores: dict[str, float] | None = None) -> list[str]:
    if selector in {"high", "low"} and scores is not None:
        sign = -1 if selector == "high" else 1
        ordered = sorted(tests, key=lambda t: (sign * scores[t], stable_seed(task_id + t)))
    elif selector == "high":
        ordered = select_quality_tests(tests, task_id=task_id, quality="high")
    elif selector == "low":
        ordered = select_quality_tests(tests, task_id=task_id, quality="low")
    elif selector == "diverse":
        ordered = select_quality_tests(tests, task_id=task_id, quality="diverse")
    elif selector == "random":
        ordered = list(tests)
        random.Random(stable_seed(task_id + selector)).shuffle(ordered)
    else:
        raise ValueError(f"Unknown selector: {selector}")
    return ordered[:count]


def fill_with_public_tests(selected: list[str], public_tests: list[str], count: int) -> list[str]:
    seen = {normalize_test(test) for test in selected}
    output = list(selected)
    for test in public_tests:
        if len(output) >= count:
            break
        norm = normalize_test(test)
        if norm in seen:
            continue
        output.append(test)
        seen.add(norm)
    return output


def write_report(path: Path, results: dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("task_id,generated_count,valid_count,invalid_count\n")
        for task_id, result in sorted(results.items()):
            handle.write(
                f"{task_id},{result.get('generated_count', 0)},"
                f"{result.get('valid_count', 0)},{result.get('invalid_count', 0)}\n"
            )


if __name__ == "__main__":
    main()
