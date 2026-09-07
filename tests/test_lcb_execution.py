import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tddexp.lcb_execution import grade


@unittest.skipUnless((ROOT / '.tmp/LiveCodeBench/lcb_runner').exists(), 'Official LiveCodeBench checkout required')
class LiveCodeBenchExecutionTests(unittest.TestCase):
    def test_official_checker_accepts_correct_and_rejects_wrong_code(self):
        cases = [(None, 'print(sum(map(int, input().split())))', 'print(0)', ['2 3\n', '4 -1\n'], ['5\n', '3\n']),
                 ('add', 'class Solution:\n def add(self,a,b): return a+b',
                  'class Solution:\n def add(self,a,b): return a-b', ['2\n3', '4\n-1'], ['5', '3'])]
        for fn, correct, wrong, inputs, outputs in cases:
            with self.subTest(interface=fn or 'stdin'):
                sample = {'input_output': json.dumps(dict(inputs=inputs, outputs=outputs, fn_name=fn))}
                self.assertTrue(grade(sample, correct, ROOT / '.tmp/LiveCodeBench', 2)['passed'])
                self.assertFalse(grade(sample, wrong, ROOT / '.tmp/LiveCodeBench', 2)['passed'])

    def test_resource_exhaustion_and_early_exit_fail_without_aborting_evaluation(self):
        sample = {'input_output': json.dumps(dict(inputs=['1'], outputs=['1'], fn_name='f'))}
        for code in ['class Solution:\n def f(self,x): return len(bytearray(5 * 1024**3))',
                     'class Solution:\n def f(self,x): raise SystemExit(1)']:
            self.assertFalse(grade(sample, code, ROOT / '.tmp/LiveCodeBench', 2)['passed'])


if __name__ == '__main__': unittest.main()
