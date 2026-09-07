#!/usr/bin/env python
"""Build matched correct-output, wrong-output, and input-only prompts."""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tddexp.controlled import PROTOCOL_VERSION, filter_evalplus_test, input_key, prompt_with_tests, target_call, wrong_value, strip_problem_examples
from tddexp.data import load_tasks, format_lcb_test_case
from tddexp.execution import execute
from tddexp.io import read_jsonl, write_jsonl


def prepare_function(task, row):
    code = task.imports + "\n" + task.canonical_code
    calls, boolean_outputs = [], []
    for t in task.tests:
        try:
            call = target_call(t, task.entry_point)
            # A complete input must be available without hidden evaluator state.
            for arg in call.args:
                ast.literal_eval(arg)
            calls.append(call)
            predicate = ast.parse(t).body[0].test
            boolean_outputs.append(isinstance(predicate, (ast.Call, ast.UnaryOp)))
        except (ValueError, SyntaxError):
            continue
        if len(calls) == 3:
            break
    if not calls:
        raise ValueError("No public inputs.")
    # Canonicalize arguments, including literal arithmetic and keyword arguments.
    call_texts = [('bool(' + ast.unparse(c) + ')') if boolean else ast.unparse(c)
                  for c, boolean in zip(calls, boolean_outputs)]
    queries = call_texts + ["(" + ", ".join(ast.unparse(a) for a in c.args) + ("," if c.args else "") + ")" for c in calls]
    if any(c.keywords for c in calls):
        raise ValueError("Public input uses keyword arguments.")
    values = execute(code, expressions=queries, timeout=15)["values"]
    if len(values) != 2 * len(calls) or any(v is None for v in values):
        raise ValueError("Reference execution failed on public inputs.")
    outputs = []
    for call_text, value in zip(call_texts, values[:len(calls)]):
        try:
            outputs.append(ast.literal_eval(value))
        except ValueError:
            if value.startswith('Counter('):
                outputs.append(ast.literal_eval(value[len('Counter('):-1]))
            else:
                raise
    args = [ast.literal_eval(v) for v in values[len(calls):]]
    correct = [f"assert {call} == {value!r}" for call, value in zip(call_texts, outputs)]
    wrong = []
    expected = []
    for call, value in zip(call_texts, outputs):
        try:
            changed = wrong_value(value)
            expected.append(False)
        except ValueError:
            changed = value
            expected.append(True)
        wrong.append(f"assert {call} == {changed!r}")
    if all(expected):
        raise ValueError("No type-preserving incorrect public output.")
    check = execute(code, tests=correct + wrong, timeout=15)
    if check["passed"] != [True] * len(correct) + expected:
        raise ValueError("Correct/incorrect oracle validation failed.")
    heldout, counts = filter_evalplus_test(row["test"], {input_key(a) for a in args})
    official = row["test"]
    if "canonical_solution" in row:
        heldout += f"\ncheck({task.entry_point})\n"
        official += f"\ncheck({task.entry_point})\n"
    return correct, wrong, call_texts, {"official_test": official, "heldout_test": heldout,
                                       "heldout_counts": counts, "reference_code": code,
                                       "oracle_validation": "reference_execution",
                                       "changed_outputs": expected.count(False)}


def wrong_lcb_output(output: str) -> str:
    try:
        value = json.loads(output)
        return json.dumps(wrong_value(value), ensure_ascii=False)
    except (json.JSONDecodeError, ValueError, TypeError):
        if output.strip().upper() in {"YES", "NO"}:
            return "NO" if output.strip().upper() == "YES" else "YES"
        match = re.search(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", output)
        if match:
            token = match.group()
            changed = str(float(token) + 1) if "." in token else str(int(token) + 1)
            return output[:match.start()] + changed + output[match.end():]
        if output.strip():
            # Keep a plain-text stdout value plain text (e.g. cardinal directions).
            i = next(i for i, c in enumerate(output) if not c.isspace())
            replacement = "B" if output[i] == "A" else "A"
            return output[:i] + replacement + output[i + 1:]
        raise ValueError("No type-preserving incorrect public output.")


def prepare_one(item):
    task, source, row = item
    task = replace(task, prompt=strip_problem_examples(task.prompt))
    result = {"task_id": source + "::" + task.task_id, "source_task_id": task.task_id,
              "source_dataset": source, "protocol_version": PROTOCOL_VERSION,
              "prompt": task.prompt, "signature": task.signature, "entry_point": task.entry_point}
    try:
        if source == "livecodebench":
            public = task.raw_public_tests
            correct = [format_lcb_test_case(t) for t in public]
            wrong_raw = [{**t, "output": wrong_lcb_output(t["output"])} for t in public]
            wrong = [format_lcb_test_case(t) for t in wrong_raw]
            inputs = ["input: " + t["input"] for t in public]
            meta = {"oracle_validation": "different_from_official_public_output",
                    "official_dataset": "data/livecodebench_release_v6_minus_v5.jsonl"}
        else:
            correct, wrong, inputs, meta = prepare_function(task, row)
        result.update(meta)
        if result['task_id'] == 'mbppplus::599':
            result['evaluation_timeout_seconds'] = 60
        result["condition_tests"] = {"nl_only": [], "nl_tests": correct, "wrong_tests": wrong, "inputs_only": inputs}
        result["condition_prompts"] = {c: prompt_with_tests(task.prompt, tests, c == "inputs_only")
                                       for c, tests in result["condition_tests"].items()}
        return result, None
    except (ValueError, SyntaxError, TypeError, KeyError) as exc:
        return None, {"task_id": result["task_id"], "reason": str(exc)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/controlled/main")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    jobs = []
    for source, name in [("humanevalplus", "humanevalplus"), ("mbppplus", "mbppplus"),
                         ("livecodebench", "livecodebench_release_v6_minus_v5")]:
        path = ROOT / "data" / (name + ".jsonl")
        rows = read_jsonl(path)
        by_id = {str(r.get("task_id", r.get("question_id"))): r for r in rows}
        jobs.extend((task, source, by_id[task.task_id]) for task in load_tasks(path, source))
    records, excluded = [], []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for row, error in pool.map(prepare_one, jobs):
            if row is not None:
                records.append(row)
            else:
                excluded.append(error)
    out = Path(a.output_dir)
    write_jsonl(out / "tasks.jsonl", records)
    write_jsonl(out / "excluded.jsonl", excluded)
    summary = {"source_tasks": len(jobs), "eligible_tasks": len(records), "excluded_tasks": len(excluded),
               "by_dataset": {s: sum(r["source_dataset"] == s for r in records)
                              for s in ["humanevalplus", "mbppplus", "livecodebench"]}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
