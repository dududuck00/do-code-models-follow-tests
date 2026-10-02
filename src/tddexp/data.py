from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import ast
from typing import Iterable


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    prompt: str
    tests: list[str]
    entry_point: str
    canonical_code: str
    imports: str = ""
    signature: str = ""
    dataset_name: str = ""
    raw_public_tests: list[dict] | None = None
    raw_private_tests: list[dict] | None = None
    platform: str = ""
    difficulty: str = ""
    contest_date: str = ""
    starter_code: str = ""
    metadata: dict | None = None

    def split_tests(self, visible_count: int | None) -> tuple[list[str], list[str]]:
        if visible_count is None:
            visible_count = self.default_visible_count()
        visible = self.tests[:visible_count]
        hidden = self.tests[visible_count:]
        if not hidden:
            hidden = self.tests[:]
        return visible, hidden

    def default_visible_count(self) -> int:
        if self.raw_public_tests is not None:
            return len(self.raw_public_tests)
        return 3


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_assertions(test_text: str) -> list[str]:
    if isinstance(test_text, list):
        return [str(test).strip() for test in test_text if str(test).strip().startswith("assert ")]
    tests: list[str] = []
    for line in test_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("assert "):
            tests.append(stripped)
    return tests


def load_tasks(path: str | Path, dataset_name: str = "") -> list[CodeTask]:
    tasks: list[CodeTask] = []
    public_tests_by_id = load_public_tests_by_id(path, dataset_name)
    for row in read_jsonl(path):
        if row.get("protocol_version") == "semantic-tests-1":
            tasks.append(CodeTask(
                task_id=row["task_id"], prompt=row["prompt"],
                tests=row["condition_tests"].get("nl_tests", row["condition_tests"].get("tests_a", row["condition_tests"].get("unrelated_tests", []))),
                entry_point=row.get("entry_point", ""), canonical_code=row.get("reference_code", ""),
                signature=row.get("signature", ""), dataset_name="controlled_" + row["source_dataset"],
                metadata={"condition_prompts": row["condition_prompts"], "condition_tests": row["condition_tests"]},
            ))
            continue
        if is_livecodebench_row(row):
            tasks.append(load_livecodebench_row(row, dataset_name=dataset_name or "livecodebench"))
            continue
        if is_evalplus_row(row, dataset_name):
            tasks.append(
                load_evalplus_row(
                    row,
                    dataset_name=dataset_name,
                    public_tests_override=public_tests_by_id.get(str(row["task_id"])),
                )
            )
            continue

        test_text = row.get("test_list") or row.get("test") or ""
        tests = split_assertions(test_text)
        if not tests:
            continue
        task_id = str(row["task_id"])
        tasks.append(
            CodeTask(
                task_id=task_id,
                prompt=row["prompt"],
                tests=tests,
                entry_point=row.get("entry_point", ""),
                canonical_code=row.get("code", ""),
                imports=row.get("imports", ""),
                signature=row.get("signatures", ""),
                dataset_name=dataset_name,
            )
        )
    return tasks


def load_public_tests_by_id(path: str | Path, dataset_name: str) -> dict[str, list[str]]:
    if dataset_name.lower() != "humanevalplus":
        return {}
    base_path = Path(path).with_name("humaneval.jsonl")
    if not base_path.exists():
        return {}
    public_tests: dict[str, list[str]] = {}
    for row in read_jsonl(base_path):
        public_tests[str(row["task_id"])] = split_assertions(row.get("test", ""))
    return public_tests


def is_evalplus_row(row: dict, dataset_name: str = "") -> bool:
    name = dataset_name.lower()
    return (
        name in {"humanevalplus", "mbppplus"}
        or "canonical_solution" in row
        or isinstance(row.get("test_list"), list)
    ) and "test" in row


def load_evalplus_row(
    row: dict,
    dataset_name: str,
    public_tests_override: list[str] | None = None,
) -> CodeTask:
    name = dataset_name.lower()
    task_id = str(row["task_id"])
    entry_point = str(row.get("entry_point") or infer_entry_point(row.get("code", "")))
    metadata = {
        "evalplus_test": row.get("test", ""),
        "evalplus_style": "humanevalplus" if "canonical_solution" in row or name == "humanevalplus" else "mbppplus",
    }

    if metadata["evalplus_style"] == "humanevalplus":
        synthetic_public_tests = normalize_public_tests(row.get("synthetic_public_tests"))
        public_tests = (
            synthetic_public_tests
            or public_tests_override
            or parse_evalplus_io_assertions(row.get("test", ""), entry_point, max_tests=3)
            or parse_doctest_assertions(row.get("prompt", ""))
        )
        source_prompt = str(row.get("prompt", ""))
        imports = imports_before_first_def(source_prompt)
        source_tree = ast.parse(source_prompt)
        target = next(n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name == entry_point)
        signature = ast.unparse(target).splitlines()[0]
        description = strip_doctest_examples(ast.get_docstring(target, clean=True) or "")
        prompt = build_humanevalplus_prompt(
            description,
            signature,
            imports,
        )
        helpers = [n for n in source_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n is not target]
        if helpers:
            context = "### Provided Helpers\n```python\n" + "\n\n".join(ast.unparse(n) for n in helpers) + "\n```\n\n"
            prompt = prompt.replace("### Answer\n", context + "### Answer\n")
        canonical_code = source_prompt + str(row.get("canonical_solution", ""))
    else:
        raw_tests = normalize_public_tests(row.get("synthetic_public_tests")) or row.get("test_list") or []
        public_tests = [str(test) for test in raw_tests]
        code_tree = ast.parse(row.get("code", ""))
        functions = {n.name: n for n in code_tree.body if isinstance(n, ast.FunctionDef)}
        calls = []
        for test in row.get("test_list", []):
            calls.extend(n.func.id for n in ast.walk(ast.parse(test)) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id in functions)
        if calls:
            entry_point = max(dict.fromkeys(calls), key=calls.count)
        signature = ast.unparse(functions[entry_point]).splitlines()[0]
        prompt = build_mbppplus_prompt(str(row.get("prompt", "")), signature)
        canonical_code = str(row.get("code", ""))

    return CodeTask(
        task_id=task_id,
        prompt=prompt,
        tests=public_tests,
        entry_point=entry_point,
        canonical_code=canonical_code,
        imports=(
            imports
            if metadata["evalplus_style"] == "humanevalplus"
            else "\n".join(row.get("test_imports") or [])
        ),
        signature=signature,
        dataset_name=dataset_name,
        metadata=metadata,
    )


def is_livecodebench_row(row: dict) -> bool:
    return "question_content" in row and "public_test_cases" in row and "private_test_cases" in row


def normalize_public_tests(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(test).strip() for test in value if str(test).strip()]


def load_livecodebench_row(row: dict, dataset_name: str) -> CodeTask:
    public_tests = parse_lcb_tests(row.get("public_test_cases"))
    private_tests = parse_lcb_tests(row.get("private_test_cases"))
    task_id = str(row.get("question_id") or row.get("task_id") or row.get("question_title"))
    prompt = build_lcb_base_prompt(row)
    visible = [format_lcb_test_case(test) for test in public_tests]
    hidden = [format_lcb_test_case(test) for test in private_tests]
    return CodeTask(
        task_id=task_id,
        prompt=prompt,
        tests=visible + hidden,
        entry_point=guess_lcb_entry_point(row.get("starter_code", "")),
        canonical_code=row.get("code", ""),
        imports="",
        signature=row.get("starter_code", ""),
        dataset_name=dataset_name,
        raw_public_tests=public_tests,
        raw_private_tests=private_tests,
        platform=str(row.get("platform", "")),
        difficulty=str(row.get("difficulty", "")),
        contest_date=str(row.get("contest_date", "")),
        starter_code=str(row.get("starter_code", "")),
        metadata=parse_json_maybe(row.get("metadata")) or {},
    )


def parse_lcb_tests(value: object) -> list[dict]:
    parsed = parse_json_maybe(value)
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        if "test_cases" in parsed and isinstance(parsed["test_cases"], list):
            parsed = parsed["test_cases"]
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def parse_json_maybe(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def format_lcb_test_case(test: dict) -> str:
    if "input" in test and "output" in test:
        return f"input: {test['input']}\nexpected output: {test['output']}"
    if "stdin" in test and "stdout" in test:
        return f"stdin: {test['stdin']}\nexpected stdout: {test['stdout']}"
    return json.dumps(test, ensure_ascii=False)


def build_lcb_base_prompt(row: dict) -> str:
    title = row.get("question_title") or row.get("question_id") or "Programming problem"
    content = row.get("question_content_no_public_tests") or row.get("question_content", "")
    starter = row.get("starter_code", "")
    prompt = f"### Question: {title}\n\n{content.strip()}\n\n"
    if starter:
        prompt += "### Starter Code\n```python\n" + starter.rstrip() + "\n```\n\n"
        prompt += "### Answer\nComplete the starter code in Python. Return only the code inside a Python markdown block.\n```python\n"
    else:
        prompt += "### Answer\nWrite a complete Python solution. Return only the code inside a Python markdown block.\n```python\n"
    return prompt


def guess_lcb_entry_point(starter_code: str) -> str:
    for line in starter_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and "(" in stripped:
            return stripped.split("def ", 1)[1].split("(", 1)[0].strip()
    return ""


def infer_entry_point(code: str) -> str:
    match = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code, flags=re.MULTILINE)
    return match.group(1) if match else ""


def first_def_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped
    return ""


def parse_doctest_assertions(prompt: str) -> list[str]:
    tests: list[str] = []
    lines = prompt.splitlines()
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped.startswith(">>> "):
            idx += 1
            continue
        expr = stripped[4:].strip()
        idx += 1
        expected_lines: list[str] = []
        while idx < len(lines):
            current = lines[idx].strip()
            if current.startswith(">>> ") or current.endswith('"""') or current.endswith("'''"):
                break
            if current:
                expected_lines.append(current)
            idx += 1
        if expr and expected_lines:
            tests.append(f"assert {expr} == {' '.join(expected_lines)}")
    return tests


def parse_evalplus_io_assertions(test: str, entry_point: str, max_tests: int = 3) -> list[str]:
    inputs, results = extract_evalplus_inputs_results(test)
    if not inputs or not results:
        return []
    tests: list[str] = []
    for inp, expected in zip(inputs[:max_tests], results[:max_tests]):
        args = ", ".join(repr(arg) for arg in ensure_arg_list(inp))
        tests.append(f"assert {entry_point}({args}) == {repr(expected)}")
    return tests


def extract_evalplus_inputs_results(test: str) -> tuple[list | None, list | None]:
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return None, None

    values: dict[str, list] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"inputs", "results"}:
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(value, list):
                    values[target.id] = value
    return values.get("inputs"), values.get("results")


def ensure_arg_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def strip_doctests_from_prompt(prompt: str) -> str:
    lines = prompt.splitlines()
    kept: list[str] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped.startswith(">>> "):
            kept.append(lines[idx])
            idx += 1
            continue
        idx += 1
        while idx < len(lines):
            current = lines[idx].strip()
            if current.startswith(">>> ") or current.endswith('"""') or current.endswith("'''"):
                break
            idx += 1
    return "\n".join(kept).rstrip() + "\n"


def imports_before_first_def(prompt: str) -> str:
    imports: list[str] = []
    for line in prompt.splitlines():
        if line.strip().startswith("def "):
            break
        if line.strip():
            imports.append(line.rstrip())
    return "\n".join(imports).strip()


def extract_docstring_description(prompt: str) -> str:
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return strip_doctests_from_prompt(prompt).strip()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node, clean=True) or ""
            return strip_doctest_examples(docstring).strip()
    return strip_doctests_from_prompt(prompt).strip()


def strip_doctest_examples(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped.startswith(">>> "):
            kept.append(lines[idx])
            idx += 1
            continue
        idx += 1
        while idx < len(lines):
            current = lines[idx].strip()
            if current.startswith(">>> "):
                break
            if not current:
                idx += 1
                break
            idx += 1
    return "\n".join(kept).strip()


def build_humanevalplus_prompt(description: str, signature: str, imports: str = "") -> str:
    prompt = "### Problem\n" + description.strip() + "\n\n"
    if signature:
        code_lines = []
        if imports.strip():
            code_lines.append(imports.strip())
            code_lines.append("")
        code_lines.append(signature)
        prompt += "### Function Signature\n```python\n" + "\n".join(code_lines) + "\n```\n\n"
    prompt += "### Answer\nWrite a complete Python solution. Return only the code inside a Python markdown block.\n```python\n"
    return prompt


def build_humanevalplus_canonical_code(imports: str, signature: str, body: str) -> str:
    parts: list[str] = []
    if imports.strip():
        parts.append(imports.strip())
    if signature.strip():
        parts.append(signature.strip() + body.rstrip() + "\n")
    else:
        parts.append(body.rstrip() + "\n")
    return "\n\n".join(parts).rstrip() + "\n"


def build_mbppplus_prompt(description: str, signature: str) -> str:
    prompt = "### Problem\n" + description.strip() + "\n\n"
    if signature:
        prompt += "### Function Signature\n```python\n" + signature + "\n```\n\n"
    prompt += "### Answer\nWrite a complete Python solution. Return only the code inside a Python markdown block.\n```python\n"
    return prompt


def make_irrelevant_tests(tasks: list[CodeTask], visible_count: int | None) -> dict[str, list[str]]:
    """Assign each task visible tests from the next task as irrelevant tests."""
    irrelevant: dict[str, list[str]] = {}
    if not tasks:
        return irrelevant
    for idx, task in enumerate(tasks):
        donor = tasks[(idx + 1) % len(tasks)]
        visible, _ = donor.split_tests(visible_count)
        irrelevant[task.task_id] = visible
    return irrelevant
