from __future__ import annotations

import random
import re
import ast
from dataclasses import dataclass

from .data import CodeTask


CONDITIONS = {
    "nl_only",
    "nl_tests",
    "shuffled_tests",
    "irrelevant_tests",
    "asserts_only",
    "test_names_only",
}


@dataclass(frozen=True)
class ConditionSpec:
    base: str
    count: int | None = None
    quality: str = "original"


def format_tests(tests: list[str]) -> str:
    return "\n".join(tests).strip()


def shuffle_test_outputs(tests: list[str], seed: int = 0) -> list[str]:
    """Shuffle expected values, rejecting suites without a semantic permutation."""
    parsed: list[tuple[str, str]] = []
    for test in tests:
        try:
            node = ast.parse(test).body[0]
        except (SyntaxError, IndexError):
            return shuffle_lcb_expected_outputs(tests, seed=seed)
        if not (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)):
            raise ValueError("Output shuffling requires equality assertions; use prepared wrong_tests for other predicates.")
        parsed.append(("assert " + ast.unparse(node.test.left) + " == ", ast.unparse(node.test.comparators[0])))
    return _shuffle_parsed_outputs(parsed, seed)


def _shuffle_parsed_outputs(parsed: list[tuple[str, str]], seed: int) -> list[str]:
    outputs = [rhs for _, rhs in parsed]
    if len(set(outputs)) < 2:
        raise ValueError("No output permutation changes this test suite.")
    rng = random.Random(seed)
    rng.shuffle(outputs)
    if outputs == [rhs for _, rhs in parsed]:
        outputs = outputs[1:] + outputs[:1]
    return [lhs + rhs for (lhs, _), rhs in zip(parsed, outputs)]


def shuffle_lcb_expected_outputs(tests: list[str], seed: int = 0) -> list[str]:
    parsed: list[tuple[str, str]] = []
    for test in tests:
        if "\nexpected output: " in test:
            lhs, rhs = test.split("\nexpected output: ", 1)
            parsed.append((lhs + "\nexpected output: ", rhs))
        elif "\nexpected stdout: " in test:
            lhs, rhs = test.split("\nexpected stdout: ", 1)
            parsed.append((lhs + "\nexpected stdout: ", rhs))
        else:
            raise ValueError("Cannot locate an expected output in this test.")
    return _shuffle_parsed_outputs(parsed, seed)


def build_prompt(
    task: CodeTask,
    condition: str,
    visible_tests: list[str],
    irrelevant_tests: list[str] | None = None,
) -> str:
    prepared = (task.metadata or {}).get("condition_prompts", {})
    if condition in prepared:
        return prepared[condition]
    spec = parse_condition(condition)
    if spec is None:
        raise ValueError(f"Unknown condition: {condition}")

    if spec.base == "nl_only":
        return task.prompt

    tests = select_condition_tests(task, condition, visible_tests, irrelevant_tests=irrelevant_tests)

    if spec.base == "asserts_only":
        return _tests_only_prompt(task, tests)

    if spec.base == "test_names_only":
        test_names = [f"test_{idx + 1}: {task.entry_point}" for idx, _ in enumerate(tests)]
        if is_structured_prompt_dataset(task.dataset_name):
            return insert_test_names_before_answer(task.prompt, test_names)
        return task.prompt + "\n# Tests:\n" + "\n".join(f"# {name}" for name in test_names) + "\n\n"

    if not tests:
        return task.prompt

    if is_structured_prompt_dataset(task.dataset_name):
        return insert_tests_before_answer(task.prompt, tests)

    return (
        task.prompt.rstrip()
        + "\n\n"
        + "# The implementation should satisfy these tests:\n"
        + "\n".join(f"# {line}" for line in tests)
        + "\n\n"
    )


def parse_condition(condition: str) -> ConditionSpec | None:
    if condition in CONDITIONS:
        return ConditionSpec(base=condition)

    match = re.match(
        r"^(nl_tests|asserts_only|test_names_only|shuffled_tests|irrelevant_tests)_k?(\d+|all)$",
        condition,
    )
    if match:
        return ConditionSpec(base=match.group(1), count=parse_count(match.group(2)))

    match = re.match(
        r"^(nl_tests|asserts_only|test_names_only)_(high|low|diverse)(\d+|all)$",
        condition,
    )
    if match:
        return ConditionSpec(
            base=match.group(1),
            quality=match.group(2),
            count=parse_count(match.group(3)),
        )
    return None


def parse_count(value: str) -> int | None:
    if value == "all":
        return None
    return max(0, int(value))


def select_condition_tests(
    task: CodeTask,
    condition: str,
    visible_tests: list[str],
    irrelevant_tests: list[str] | None = None,
) -> list[str]:
    prepared = (task.metadata or {}).get("condition_tests", {})
    if condition in prepared:
        return list(prepared[condition])
    spec = parse_condition(condition)
    if spec is None:
        raise ValueError(f"Unknown condition: {condition}")

    if spec.base == "nl_only":
        return []
    if spec.base == "irrelevant_tests":
        tests = irrelevant_tests or []
    elif spec.base == "shuffled_tests":
        tests = shuffle_test_outputs(visible_tests, seed=stable_seed(task.task_id))
    else:
        tests = select_quality_tests(visible_tests, task_id=task.task_id, quality=spec.quality)
    return tests if spec.count is None else tests[: spec.count]


def select_quality_tests(tests: list[str], task_id: str, quality: str) -> list[str]:
    if quality == "original":
        return list(tests)
    scored = [(test_quality_score(test), stable_seed(task_id + test), test) for test in tests]
    if quality == "high":
        return [test for _, _, test in sorted(scored, key=lambda item: (-item[0], item[1]))]
    if quality == "low":
        return [test for _, _, test in sorted(scored, key=lambda item: (item[0], item[1]))]
    if quality == "diverse":
        return select_diverse_tests(scored)
    raise ValueError(f"Unknown test quality selector: {quality}")


def select_diverse_tests(scored: list[tuple[float, int, str]]) -> list[str]:
    if not scored:
        return []
    ordered = sorted(scored, key=lambda item: (-item[0], item[1]))
    selected = [ordered[0]]
    remaining = ordered[1:]
    while remaining:
        best_idx = max(
            range(len(remaining)),
            key=lambda idx: (
                min(test_distance(remaining[idx][2], chosen[2]) for chosen in selected),
                remaining[idx][0],
                -remaining[idx][1],
            ),
        )
        selected.append(remaining.pop(best_idx))
    return [test for _, _, test in selected]


def test_distance(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[A-Za-z_]+|-?\d+|True|False|None", left))
    right_tokens = set(re.findall(r"[A-Za-z_]+|-?\d+|True|False|None", right))
    if not left_tokens and not right_tokens:
        return abs(len(left) - len(right))
    union = left_tokens | right_tokens
    intersection = left_tokens & right_tokens
    return 1.0 - (len(intersection) / len(union))


def test_quality_score(test: str) -> float:
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return fallback_quality_score(test)
    score = 0.0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            score += constant_score(node.value)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            score += 1.0 + 0.25 * len(node.elts)
            if len(node.elts) == 0:
                score += 2.0
        elif isinstance(node, ast.Dict):
            score += 1.0 + 0.25 * len(node.keys)
            if len(node.keys) == 0:
                score += 2.0
        elif isinstance(node, ast.Call):
            score += 0.5 + 0.2 * len(node.args)
    return score + min(len(test), 300) / 300.0


def constant_score(value: object) -> float:
    if value in (None, True, False):
        return 1.5
    if isinstance(value, (int, float)):
        score = 1.0
        if value in {-1, 0, 1}:
            score += 1.0
        if isinstance(value, (int, float)) and abs(value) > 100:
            score += 1.0
        if isinstance(value, (int, float)) and value < 0:
            score += 1.0
        return score
    if isinstance(value, str):
        score = 1.0 + min(len(value), 50) / 25.0
        if value == "":
            score += 2.0
        if any(ch.isspace() for ch in value):
            score += 0.5
        return score
    return 0.5


def fallback_quality_score(test: str) -> float:
    score = min(len(test), 300) / 100.0
    score += len(re.findall(r"-?\d+", test)) * 0.25
    score += sum(token in test for token in ("[]", "()", "{}", "''", '""', "None", "True", "False"))
    return score


def _tests_only_prompt(task: CodeTask, tests: list[str]) -> str:
    signature = task.signature.strip()
    if not signature:
        signature = _guess_signature(task.prompt)
    if is_structured_prompt_dataset(task.dataset_name):
        return build_structured_tests_only_prompt(task.imports, signature, tests)

    header = task.imports.strip()
    if header:
        header += "\n\n"
    return (
        header
        + "# Implement the function so that the assertions pass.\n"
        + "\n".join(f"# {line}" for line in tests)
        + "\n\n"
        + signature
        + "\n"
    )


def _guess_signature(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.strip().startswith("def "):
            return line.strip()
    return prompt.rstrip()


def is_structured_prompt_dataset(dataset_name: str) -> bool:
    return dataset_name.lower().startswith(("livecodebench", "mbppplus", "humanevalplus"))


def insert_tests_before_answer(prompt: str, tests: list[str]) -> str:
    test_block = (
        "### Visible Test Cases\n"
        + "\n\n".join(f"Test {idx + 1}\n{test}" for idx, test in enumerate(tests))
        + "\n\n"
    )
    marker = "### Answer\n"
    if marker in prompt:
        before, after = prompt.split(marker, 1)
        return before.rstrip() + "\n\n" + test_block + marker + after
    return prompt.rstrip() + "\n\n" + test_block


def insert_test_names_before_answer(prompt: str, test_names: list[str]) -> str:
    test_block = (
        "### Visible Test Names\n"
        + "\n".join(f"- {name}" for name in test_names)
        + "\n\n"
    )
    marker = "### Answer\n"
    if marker in prompt:
        before, after = prompt.split(marker, 1)
        return before.rstrip() + "\n\n" + test_block + marker + after
    return prompt.rstrip() + "\n\n" + test_block


def build_structured_tests_only_prompt(imports: str, signature: str, tests: list[str]) -> str:
    prompt = "### Function Signature\n```python\n"
    if imports.strip():
        prompt += imports.strip() + "\n\n"
    prompt += signature.strip() + "\n```\n\n"
    prompt += (
        "### Visible Test Cases\n"
        + "\n\n".join(f"Test {idx + 1}\n{test}" for idx, test in enumerate(tests))
        + "\n\n"
    )
    prompt += "### Answer\nWrite a complete Python solution. Return only the code inside a Python markdown block.\n```python\n"
    return prompt


def stable_seed(text: str) -> int:
    total = 0
    for char in text:
        total = (total * 131 + ord(char)) % (2**31 - 1)
    return total
