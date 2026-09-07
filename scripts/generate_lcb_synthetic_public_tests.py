#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import re
import sys
import time

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tddexp.io import append_jsonl, read_jsonl
from tddexp.prompts import stable_seed

from generate_synthetic_public_tests import (  # noqa: E402
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    call_chat_completion,
    extract_json_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic LiveCodeBench public tests with an OpenAI-compatible "
            "strong model and write dataset variants with replaced public_test_cases."
        )
    )
    parser.add_argument("--dataset", default="data/livecodebench_release_v6_minus_v5.jsonl")
    parser.add_argument("--output-dir", default="data/synthetic_public_tests/livecodebench_v6_new")
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        help="Re-parse and re-validate existing raw API responses before deciding what to regenerate.",
    )
    parser.add_argument(
        "--keep-failed-results",
        action="store_true",
        help="When resuming, keep previous rows with zero valid tests instead of regenerating them.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use original public tests instead of calling the API.")
    parser.add_argument("--counts", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--selectors", nargs="+", default=["high", "random", "diverse"])
    parser.add_argument("--fallback-public", action="store_true", default=True)
    parser.add_argument("--no-fallback-public", dest="fallback_public", action="store_false")
    parser.add_argument("--include-original-duplicates", action="store_true")
    parser.add_argument("--max-input-chars", type=int, default=2000)
    parser.add_argument("--max-output-chars", type=int, default=2000)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]

    api_key = DEFAULT_API_KEY if args.dry_run else get_api_key(args)
    base_url = get_base_url(args)
    candidates_path = output_dir / "lcb_synthetic_test_candidates.jsonl"
    existing_results = load_existing_results(
        candidates_path,
        keep_failed=True,
    ) if args.resume else {}
    if args.reprocess_existing and existing_results:
        existing_results = reprocess_existing_results(existing_results, rows, args)
        rewrite_candidates(candidates_path, existing_results)
    seen_results = {
        task_id: result
        for task_id, result in existing_results.items()
        if args.keep_failed_results or int(result.get("valid_count", 0)) > 0
    }
    if candidates_path.exists() and not args.resume:
        candidates_path.unlink()

    results: dict[str, dict] = {}
    pending_rows = []
    for row in rows:
        task_id = task_id_for_row(row)
        if task_id in seen_results:
            results[task_id] = seen_results[task_id]
        else:
            pending_rows.append(row)

    worker_count = max(1, args.workers)
    if worker_count == 1 or len(pending_rows) <= 1:
        for row in tqdm(pending_rows, desc="lcb synthetic tests", unit="task"):
            result = process_row(row, args, base_url, api_key)
            maybe_log_warning(result)
            append_jsonl(candidates_path, result)
            results[result["task_id"]] = result
            if args.sleep > 0 and not args.dry_run:
                time.sleep(args.sleep)
    else:
        print(f"Using {worker_count} worker threads for {len(pending_rows)} pending LCB tasks.")
        futures = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for row in pending_rows:
                futures.append(executor.submit(process_row, row, args, base_url, api_key))
                if args.sleep > 0 and not args.dry_run:
                    time.sleep(args.sleep)
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"lcb synthetic tests x{worker_count}",
                unit="task",
            ):
                result = future.result()
                maybe_log_warning(result)
                append_jsonl(candidates_path, result)
                results[result["task_id"]] = result

    write_dataset_variants(
        source_rows=read_jsonl(args.dataset),
        results=results,
        output_dir=output_dir,
        selectors=args.selectors,
        counts=args.counts,
        fallback_public=args.fallback_public,
        model=args.model,
    )
    write_report(output_dir / "lcb_synthetic_test_report.csv", results)
    print(f"Wrote candidates to {candidates_path}")
    print(f"Wrote dataset variants to {output_dir}")


def get_api_key(args: argparse.Namespace) -> str:
    import os

    api_key = os.environ.get(args.api_key_env, DEFAULT_API_KEY)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env}=... before running this script.")
    return api_key


def get_base_url(args: argparse.Namespace) -> str:
    import os

    return os.environ.get(args.base_url_env, args.base_url).rstrip("/")


def process_row(row: dict, args: argparse.Namespace, base_url: str, api_key: str) -> dict:
    task_id = task_id_for_row(row)
    original_public_tests = parse_lcb_public_tests(row)
    api_error = ""
    raw_response = ""
    if args.dry_run:
        generated_tests = original_public_tests
        raw_response = "<dry-run>"
    else:
        try:
            raw_response = call_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                messages=build_messages(row, original_public_tests, args.num_candidates),
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

    result = validate_and_score_tests(
        generated_tests=generated_tests,
        original_public_tests=original_public_tests,
        expected_testtype=expected_testtype(row, original_public_tests),
        max_input_chars=args.max_input_chars,
        max_output_chars=args.max_output_chars,
        include_original_duplicates=args.include_original_duplicates,
    )
    result.update(
        {
            "task_id": task_id,
            "question_id": str(row.get("question_id", task_id)),
            "platform": str(row.get("platform", "")),
            "difficulty": str(row.get("difficulty", "")),
            "model": args.model,
            "raw_response": raw_response,
            "api_error": api_error,
            "num_requested": args.num_candidates,
        }
    )
    return result


def build_messages(row: dict, original_public_tests: list[dict], num_candidates: int) -> list[dict]:
    testtype = expected_testtype(row, original_public_tests)
    format_note = (
        "For stdin tasks, each input/output must be the exact raw stdin/stdout text."
        if testtype == "stdin"
        else (
            "For functional tasks, input must be a JSON string encoding the function arguments "
            "in the same format as the examples, and output must be a JSON string encoding the return value."
        )
    )
    examples = json.dumps(original_public_tests[:3], ensure_ascii=False, indent=2)
    content = str(row.get("question_content_no_public_tests") or row.get("question_content") or "").strip()
    starter = str(row.get("starter_code") or "").strip()
    metadata = row.get("metadata") or "{}"
    user = f"""
Generate {num_candidates} high-quality public test cases for this LiveCodeBench programming problem.

Requirements:
- Return only valid JSON with this schema: {{"tests": [{{"input": "...", "output": "...", "testtype": "{testtype}"}}]}}.
- Every test must use testtype "{testtype}".
- Each output must be the exact expected answer for the corresponding input.
- Prefer boundary cases, corner cases, tricky cases, and diverse cases that distinguish plausible wrong algorithms.
- Avoid duplicating the existing public tests exactly.
- Do not include prose, markdown fences, explanations, or code.
- {format_note}

Problem:
{content}

Starter code:
```python
{starter}
```

Metadata:
{metadata}

Existing public tests:
{examples}
""".strip()
    return [
        {
            "role": "system",
            "content": (
                "You are a careful competitive-programming test designer. "
                "You produce exact input/output test cases."
            ),
        },
        {"role": "user", "content": user},
    ]


def parse_tests_from_response(text: str) -> list[dict]:
    payloads = [extract_json_payload(text), *extract_json_candidates(text)]
    for payload in reversed([payload for payload in payloads if payload is not None]):
        tests = tests_from_payload(payload)
        if tests:
            return tests
    return []


def extract_json_candidates(text: str) -> list[object]:
    decoder = json.JSONDecoder()
    candidates: list[object] = []
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        candidates.append(payload)
    return candidates


def tests_from_payload(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get("tests")
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def validate_and_score_tests(
    generated_tests: list[dict],
    original_public_tests: list[dict],
    expected_testtype: str,
    max_input_chars: int,
    max_output_chars: int,
    include_original_duplicates: bool,
) -> dict:
    original_norms = {normalize_test(test) for test in original_public_tests}
    seen = set()
    valid = []
    invalid = []
    for test in generated_tests:
        cleaned = clean_test(test, expected_testtype)
        reason = reject_reason(cleaned, expected_testtype, max_input_chars, max_output_chars)
        norm = normalize_test(cleaned)
        if not reason and norm in seen:
            reason = "duplicate_generated"
        if not reason and not include_original_duplicates and norm in original_norms:
            reason = "duplicate_original_public"
        if reason:
            invalid.append({"test": cleaned, "reason": reason})
            continue
        seen.add(norm)
        valid.append(
            {
                "test": cleaned,
                "quality_score": lcb_quality_score(cleaned),
            }
        )
    return {
        "generated_count": len(generated_tests),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "valid_tests": valid,
        "invalid_tests": invalid,
    }


def clean_test(test: dict, expected_testtype: str) -> dict:
    cleaned = {
        "input": stringify_test_field(test.get("input", "")),
        "output": stringify_test_field(test.get("output", "")),
        "testtype": stringify_test_field(test.get("testtype", expected_testtype)).strip() or expected_testtype,
    }
    return cleaned


def stringify_test_field(value: object) -> str:
    if isinstance(value, str):
        return value.strip("\n")
    return json.dumps(value, ensure_ascii=False)


def reject_reason(test: dict, expected_testtype: str, max_input_chars: int, max_output_chars: int) -> str:
    if test.get("testtype") != expected_testtype:
        return "wrong_testtype"
    if not str(test.get("input", "")).strip():
        return "empty_input"
    if str(test.get("output", "")) == "":
        return "empty_output"
    if len(str(test.get("input", ""))) > max_input_chars:
        return "input_too_long"
    if len(str(test.get("output", ""))) > max_output_chars:
        return "output_too_long"
    if expected_testtype == "functional":
        if not valid_lcb_functional_input(str(test["input"])):
            return "functional_json_parse_error"
        try:
            json.loads(str(test["output"]))
        except json.JSONDecodeError:
            return "functional_output_json_parse_error"
    return ""


def valid_lcb_functional_input(value: str) -> bool:
    lines = [line.strip() for line in value.strip().splitlines() if line.strip()]
    if not lines:
        return False
    try:
        for line in lines:
            json.loads(line)
    except json.JSONDecodeError:
        return False
    return True


def lcb_quality_score(test: dict) -> float:
    text = f"{test.get('input', '')}\n{test.get('output', '')}"
    numbers = [int(value) for value in re.findall(r"-?\d+", text)[:100]]
    score = min(len(text), 1000) / 200.0
    score += min(len(numbers), 20) * 0.2
    score += sum(value in {-1, 0, 1} for value in numbers) * 0.2
    score += sum(abs(value) >= 100 for value in numbers) * 0.1
    score += sum(token in text for token in ("[]", "{}", "\"\"", "''", "true", "false", "null"))
    score += text.count("\n") * 0.1
    return score


def write_dataset_variants(
    source_rows: list[dict],
    results: dict[str, dict],
    output_dir: Path,
    selectors: list[str],
    counts: list[int],
    fallback_public: bool,
    model: str,
) -> None:
    for selector in selectors:
        for count in counts:
            path = output_dir / f"livecodebench_v6_new_synth_{selector}{count}.jsonl"
            if path.exists():
                path.unlink()
            for row in source_rows:
                task_id = task_id_for_row(row)
                result = results.get(task_id, {})
                valid_tests = [item["test"] for item in result.get("valid_tests", [])]
                selected = select_tests(valid_tests, selector, count, task_id)
                fallback_used = False
                if fallback_public and len(selected) < count:
                    before = len(selected)
                    selected = fill_with_public_tests(selected, parse_lcb_public_tests(row), count)
                    fallback_used = len(selected) > before
                new_row = dict(row)
                new_row["public_test_cases"] = json.dumps(selected, ensure_ascii=False)
                new_row["synthetic_public_tests_metadata"] = {
                    "selector": selector,
                    "count": count,
                    "model": model,
                    "valid_count": len(valid_tests),
                    "fallback_public_used": fallback_used,
                    "original_public_count": len(parse_lcb_public_tests(row)),
                }
                append_jsonl(path, new_row)


def select_tests(tests: list[dict], selector: str, count: int, task_id: str) -> list[dict]:
    if selector == "high":
        ordered = sorted(
            tests,
            key=lambda test: (-lcb_quality_score(test), stable_seed(task_id + normalize_test(test))),
        )
    elif selector == "low":
        ordered = sorted(
            tests,
            key=lambda test: (lcb_quality_score(test), stable_seed(task_id + normalize_test(test))),
        )
    elif selector == "random":
        ordered = list(tests)
        random.Random(stable_seed(task_id + selector)).shuffle(ordered)
    elif selector == "diverse":
        ordered = select_diverse_tests(tests, task_id)
    else:
        raise ValueError(f"Unknown selector: {selector}")
    return ordered[:count]


def select_diverse_tests(tests: list[dict], task_id: str) -> list[dict]:
    if not tests:
        return []
    ordered = sorted(
        tests,
        key=lambda test: (-lcb_quality_score(test), stable_seed(task_id + normalize_test(test))),
    )
    selected = [ordered[0]]
    remaining = ordered[1:]
    while remaining:
        idx = max(
            range(len(remaining)),
            key=lambda item_idx: (
                min(test_distance(remaining[item_idx], chosen) for chosen in selected),
                lcb_quality_score(remaining[item_idx]),
            ),
        )
        selected.append(remaining.pop(idx))
    return selected


def test_distance(left: dict, right: dict) -> float:
    left_tokens = set(re.findall(r"[A-Za-z_]+|-?\d+|true|false|null", normalize_test(left).lower()))
    right_tokens = set(re.findall(r"[A-Za-z_]+|-?\d+|true|false|null", normalize_test(right).lower()))
    if not left_tokens and not right_tokens:
        return abs(len(normalize_test(left)) - len(normalize_test(right)))
    union = left_tokens | right_tokens
    return 1.0 - len(left_tokens & right_tokens) / len(union)


def fill_with_public_tests(selected: list[dict], public_tests: list[dict], count: int) -> list[dict]:
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


def parse_lcb_public_tests(row: dict) -> list[dict]:
    value = row.get("public_test_cases")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
    else:
        parsed = value
    if not isinstance(parsed, list):
        return []
    tests = []
    for test in parsed:
        if isinstance(test, dict):
            tests.append(clean_test(test, stringify_test_field(test.get("testtype", "stdin")) or "stdin"))
    return tests


def expected_testtype(row: dict, public_tests: list[dict]) -> str:
    if public_tests:
        testtype = str(public_tests[0].get("testtype", "")).strip()
        if testtype:
            return testtype
    metadata = parse_json_maybe(row.get("metadata")) or {}
    return "functional" if metadata.get("func_name") else "stdin"


def parse_json_maybe(value: object) -> object | None:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def task_id_for_row(row: dict) -> str:
    return str(row.get("question_id") or row.get("task_id") or row.get("question_title"))


def normalize_test(test: dict) -> str:
    payload = {
        "input": str(test.get("input", "")).strip(),
        "output": str(test.get("output", "")).strip(),
        "testtype": str(test.get("testtype", "")).strip(),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def load_existing_results(path: Path, keep_failed: bool) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for row in read_jsonl(path):
        if not keep_failed and int(row.get("valid_count", 0)) == 0:
            continue
        rows[str(row["task_id"])] = row
    return rows


def reprocess_existing_results(
    existing_results: dict[str, dict],
    source_rows: list[dict],
    args: argparse.Namespace,
) -> dict[str, dict]:
    source_by_id = {task_id_for_row(row): row for row in source_rows}
    reprocessed: dict[str, dict] = {}
    for task_id, old_result in existing_results.items():
        row = source_by_id.get(task_id)
        raw_response = str(old_result.get("raw_response") or "")
        if row is None or not raw_response or raw_response == "<dry-run>":
            reprocessed[task_id] = old_result
            continue
        original_public_tests = parse_lcb_public_tests(row)
        generated_tests = parse_tests_from_response(raw_response)
        result = validate_and_score_tests(
            generated_tests=generated_tests,
            original_public_tests=original_public_tests,
            expected_testtype=expected_testtype(row, original_public_tests),
            max_input_chars=args.max_input_chars,
            max_output_chars=args.max_output_chars,
            include_original_duplicates=args.include_original_duplicates,
        )
        result.update(
            {
                "task_id": task_id,
                "question_id": str(row.get("question_id", task_id)),
                "platform": str(row.get("platform", old_result.get("platform", ""))),
                "difficulty": str(row.get("difficulty", old_result.get("difficulty", ""))),
                "model": old_result.get("model", args.model),
                "raw_response": raw_response,
                "api_error": old_result.get("api_error", ""),
                "num_requested": old_result.get("num_requested", args.num_candidates),
            }
        )
        reprocessed[task_id] = result
    return reprocessed


def rewrite_candidates(path: Path, results: dict[str, dict]) -> None:
    if path.exists():
        path.unlink()
    for task_id in sorted(results):
        append_jsonl(path, results[task_id])


def write_report(path: Path, results: dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("task_id,platform,difficulty,generated_count,valid_count,invalid_count,api_error\n")
        for task_id, result in sorted(results.items()):
            api_error = str(result.get("api_error", "")).replace("\n", " ")[:200]
            handle.write(
                f"{task_id},{result.get('platform', '')},{result.get('difficulty', '')},"
                f"{result.get('generated_count', 0)},{result.get('valid_count', 0)},"
                f"{result.get('invalid_count', 0)},{json.dumps(api_error, ensure_ascii=False)}\n"
            )


def maybe_log_warning(result: dict) -> None:
    if result.get("api_error"):
        tqdm.write(f"[warn] task_id={result.get('task_id')} failed: {result.get('api_error')}")


if __name__ == "__main__":
    main()
