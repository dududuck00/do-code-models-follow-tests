"""Run the official LiveCodeBench checker in a timed, pipe-connected child.

The caller supplies the execution sandbox. Pipes avoid the manager socket used
by the upstream parallel scheduler; grading still uses upstream run_test.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

MEMORY_LIMIT_BYTES = 4 * 1024**3


def grade(sample: dict, code: str, root: str, timeout: int = 15) -> dict:
    tmp = Path(__file__).resolve().parents[2] / '.tmp'
    env = dict(os.environ, TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    count = len(json.loads(sample['input_output'])['inputs'])
    payload = dict(sample=sample, code=code, root=str(Path(root).resolve()), timeout=timeout)
    try:
        proc = subprocess.run([sys.executable, '-B', __file__, '--worker'],
                              input=json.dumps(payload), text=True, capture_output=True,
                              cwd=tmp, env=env, timeout=(timeout + 1) * count + 5)
    except subprocess.TimeoutExpired:
        return dict(passed=False, status='timeout')
    try:
        result = json.loads(proc.stdout)
    except ValueError as exc:
        if proc.returncode != 0 and 'LCB_CHECKER_STARTED' in proc.stderr:
            return dict(passed=False, status='process_exit', returncode=proc.returncode)
        raise RuntimeError(f'LiveCodeBench worker failed: {proc.stderr[-1000:]}') from exc
    if result.get('status') == 'runner_error':
        raise RuntimeError(result['error'])
    return result


def worker():
    job = json.load(sys.stdin)
    sys.path.insert(0, job['root'])
    from lcb_runner.evaluation.testing_util import run_test
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    sys.stderr.write('LCB_CHECKER_STARTED\n')
    sys.stderr.flush()
    sys.set_int_max_str_digits(50000)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            scores, metadata = run_test(job['sample'], test=job['code'], timeout=job['timeout'])
            result = dict(passed=bool(scores) and all(bool(x > 0) for x in scores),
                          status='ok', scores=[int(x) for x in scores], metadata=metadata)
        except BaseException as exc:
            result = dict(passed=False, status='execution_error', error=f'{type(exc).__name__}: {exc}')
    json.dump(result, sys.stdout, default=str)


if __name__ == '__main__':
    worker()
