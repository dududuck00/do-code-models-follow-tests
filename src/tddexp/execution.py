"""Execute a program and a group of assertions in one timed child process."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys


def execute(code: str, tests: list[str] | None = None, expressions: list[str] | None = None,
            timeout: float = 10.0) -> dict:
    root = Path(__file__).resolve().parents[2]
    tmp = root / ".tmp"
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    payload = {"code": code, "tests": tests or [], "expressions": expressions or []}
    try:
        p = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker"],
                           input=json.dumps(payload), text=True, capture_output=True,
                           timeout=timeout, cwd=tmp, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "passed": [False] * len(tests or []), "values": []}
    try:
        return json.loads(p.stdout)
    except (ValueError, TypeError):
        return {"status": "execution_error", "passed": [False] * len(tests or []),
                "values": [], "error": p.stderr[-1000:]}


def _worker() -> None:
    job = json.load(sys.stdin)
    scope = {"__name__": "__main__"}
    result = {"status": "ok", "passed": [], "values": [], "errors": []}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            exec(compile(job["code"], "<candidate>", "exec"), scope)
        except BaseException as exc:
            result.update(status="initialization_error", error=f"{type(exc).__name__}: {exc}",
                          passed=[False] * len(job["tests"]))
        else:
            for test in job["tests"]:
                try:
                    exec(compile(test, "<test>", "exec"), scope)
                    result["passed"].append(True)
                    result["errors"].append("")
                except BaseException as exc:
                    result["passed"].append(False)
                    result["errors"].append(f"{type(exc).__name__}: {exc}"[:300])
            for expression in job["expressions"]:
                try:
                    result["values"].append(repr(eval(expression, scope)))
                except BaseException:
                    result["values"].append(None)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    _worker()
