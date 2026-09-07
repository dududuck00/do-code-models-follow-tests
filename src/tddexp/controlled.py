"""Task definitions and test operations for the controlled prompting protocol."""
from __future__ import annotations

import ast
from collections import Counter
import copy
import itertools
import json
import random
import re

PROTOCOL_VERSION = "semantic-tests-1"
MAIN_CONDITIONS = ["nl_only", "nl_tests", "wrong_tests", "inputs_only"]
SEMANTIC_CONDITIONS = ["nl_only", "inputs_only", "tests_a", "tests_b", "explicit_a", "explicit_b"]
QUALITY_CONDITIONS = ["nl_only", "quality_random", "quality_low", "quality_high"]


def strip_problem_examples(prompt: str) -> str:
    lines, kept, in_example = prompt.splitlines(), [], False
    for line in lines:
        text = line.strip()
        if re.match(r"(?i)^(?:examples?(?:\s+\d+)?\s*:|for example\b|for examples\b)", text):
            in_example = True
        if in_example and re.match(r"(?i)^(?:### |constraints\b|notes?\s*:|assumptions\b|you (?:may|can|should|must)\b)", text):
            in_example = False
        if not in_example:
            kept.append(line)
    return '\n'.join(kept) + '\n'


def prompt_with_tests(prompt: str, tests: list[str], inputs_only: bool = False) -> str:
    if not tests:
        return prompt
    title = "Example Inputs" if inputs_only else "Visible Test Cases"
    block = "### " + title + "\n" + "\n\n".join(f"Test {i + 1}\n{t}" for i, t in enumerate(tests)) + "\n\n"
    before, sep, after = prompt.partition("### Answer\n")
    return before.rstrip() + "\n\n" + block + sep + after


def saved_prompt_tests(row: dict) -> list[str]:
    marker = "### Visible Test Cases\n"
    if marker in row.get("prompt", ""):
        block = row["prompt"].split(marker, 1)[1].split("### Answer\n", 1)[0].strip()
        return [t.strip() for t in re.split(r"(?:^|\n\n)Test \d+\n", block) if t.strip()]
    return row.get("prompt_tests") or []


def normalized_test(test: str) -> str:
    try:
        return ast.dump(ast.parse(test))
    except SyntaxError:
        return test.strip()


def same_test_set(left: list[str], right: list[str]) -> bool:
    return Counter(map(normalized_test, left)) == Counter(map(normalized_test, right))


def target_call(test: str, entry_point: str) -> ast.Call:
    tree = ast.parse(test)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == entry_point]
    if len(calls) != 1:
        raise ValueError("Expected one call to the target function.")
    return calls[0]


def wrong_value(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + max(1.0, abs(value) * 0.1)
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, (tuple, list)):
        result = list(value)
        if result:
            result[0] = wrong_value(result[0])
        else:
            result.append(0)
        return tuple(result) if isinstance(value, tuple) else result
    if isinstance(value, dict):
        result = copy.deepcopy(value)
        if result:
            key = next(iter(result))
            result[key] = wrong_value(result[key])
        else:
            result["x"] = 0
        return result
    if isinstance(value, set):
        return value - {next(iter(value))} if value else {0}
    if value is None:
        raise ValueError("None has no different value of the same type.")
    raise ValueError(f"Unsupported oracle value: {type(value).__name__}")


def input_key(args) -> str:
    # Dataset arguments are positional; preserve nested list/tuple distinctions.
    return ast.dump(ast.parse(repr(tuple(args)), mode="eval"))


def literal_call_inputs(test: str, entry_point: str) -> tuple:
    call = target_call(test, entry_point)
    if call.keywords:
        raise ValueError("Keyword calls need binding before held-out filtering.")
    return tuple(ast.literal_eval(a) for a in call.args)


def filter_evalplus_test(test: str, excluded: set[str]) -> tuple[str, dict]:
    """Filter paired inputs/results arrays, retaining the benchmark's checker."""
    tree = ast.parse(test)
    counts = {"original": 0, "removed": 0, "remaining": 0}

    def visit_body(body):
        assignments = {}
        for node in body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"inputs", "results"}:
                        assignments[target.id] = node
        if "inputs" in assignments:
            class Constants(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id in {'inf', 'nan'}:
                        return ast.copy_location(ast.Constant(float(node.id)), node)
                    return node
            ins = ast.literal_eval(Constants().visit(copy.deepcopy(assignments["inputs"].value)))
            outs_node = assignments["results"].value if "results" in assignments else None
            if outs_node is not None and (not isinstance(outs_node, (ast.List, ast.Tuple)) or len(ins) != len(outs_node.elts)):
                raise ValueError("EvalPlus inputs/results are not aligned literal arrays.")
            keep = [i for i, args in enumerate(ins) if input_key(args) not in excluded]
            counts["original"] += len(ins)
            counts["remaining"] += len(keep)
            counts["removed"] += len(ins) - len(keep)
            assignments["inputs"].value = ast.List(elts=[assignments["inputs"].value.elts[i] for i in keep], ctx=ast.Load())
            if outs_node is not None:
                assignments["results"].value = ast.List(elts=[outs_node.elts[i] for i in keep], ctx=ast.Load())
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit_body(node.body)

    visit_body(tree.body)
    if not counts["remaining"]:
        raise ValueError("No held-out inputs remain in the evaluator.")
    return ast.unparse(ast.fix_missing_locations(tree)), counts


def choose_test_suites(tests: list[str], killed: list[set[str]], count: int = 3,
                      seed: int = 0, length_tolerance: float = 0.15) -> dict:
    """Optimize suite union coverage within a shared character-length band."""
    if len(tests) < count or len(tests) != len(killed):
        raise ValueError("Insufficient aligned candidate tests.")
    options = list(itertools.combinations(range(len(tests)), count))
    if not options:
        raise ValueError("No candidate suites.")
    lengths = [sum(len(tests[i]) for i in indices) for indices in options]
    center = sorted(lengths)[len(lengths) // 2]
    options = [ids for ids, length in zip(options, lengths)
               if abs(length - center) <= max(1, center * length_tolerance)]
    if not options:
        raise ValueError("No suites in the specified length band.")
    def score(ids):
        return len(set().union(*(killed[i] for i in ids)))
    rng = random.Random(seed)
    rng.shuffle(options)
    return {"high": list(max(options, key=score)), "low": list(min(options, key=score)),
            "random": list(rng.choice(options)), "length_center": center,
            "length_tolerance": length_tolerance}
