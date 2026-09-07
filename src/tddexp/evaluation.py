from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import queue
import tempfile
import traceback
from typing import Any


@dataclass
class ExecutionResult:
    passed: bool
    error: str = ""


def evaluate_code(
    code: str,
    tests: list[str],
    timeout: float = 5.0,
    tmp_root: str | Path = ".tmp",
) -> ExecutionResult:
    test_block = "\n".join(tests)
    program = code.rstrip() + "\n\n" + test_block + "\n"
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue[Any] = ctx.Queue()
    process = ctx.Process(target=_run_program, args=(program, str(tmp_root), result_queue))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join(1)
        return ExecutionResult(False, f"timeout after {timeout}s")

    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        return ExecutionResult(False, f"no result, exitcode={process.exitcode}")

    if payload["passed"]:
        return ExecutionResult(True, "")
    return ExecutionResult(False, payload["error"])


def _run_program(program: str, tmp_root: str, result_queue: mp.Queue) -> None:
    namespace: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="exec_", dir=tmp_root) as workdir:
            namespace["__name__"] = "__main__"
            namespace["__file__"] = str(Path(workdir) / "candidate.py")
            exec(compile(program, namespace["__file__"], "exec"), namespace)
        result_queue.put({"passed": True, "error": ""})
    except BaseException:
        result_queue.put({"passed": False, "error": traceback.format_exc(limit=8)})
