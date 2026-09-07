"""Regression coverage for resuming partially completed experiment artifacts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from merge_controlled_generations import merge


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


class ControlledResumeTests(unittest.TestCase):
    def test_resume_reuses_grades_and_rechecks_changed_programs_and_budgets(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as folder:
            folder = Path(folder)
            dataset, generations, output, checkpoint = [folder / f'{n}.jsonl' for n in ['tasks', 'generations', 'results', 'checkpoint']]
            tasks = [dict(task_id=str(i), source_dataset='mbppplus', condition_prompts={'nl_only': 'Implement f'},
                          official_test='assert f() == 6', heldout_test='assert f() == 6') for i in range(2)]
            rows = [dict(task_id=str(i), condition='nl_only', repeat=0,
                         candidate_code=f'def f(): return {6 if i == 0 else 0}') for i in range(2)]
            write_rows(dataset, tasks)
            command = [sys.executable, '-B', str(ROOT / 'scripts/evaluate_controlled.py'),
                       '--dataset', str(dataset), '--generations', str(generations), '--output', str(output),
                       '--checkpoint', str(checkpoint), '--workers', '1', '--timeout', '5']
            def run():
                write_rows(generations, rows)
                subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
                return [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(sum(r['passed'] for r in run()), 1)
            rows[1]['candidate_code'] = 'def f(): return 6'
            self.assertTrue(all(r['passed'] for r in run()))
            self.assertEqual(len(checkpoint.read_text().splitlines()), 3)
            tasks[0]['evaluation_timeout_seconds'] = 6
            write_rows(dataset, tasks)
            self.assertTrue(all(r['passed'] for r in run()))
            self.assertEqual(len(checkpoint.read_text().splitlines()), 4)

    def test_merging_live_files_ignores_unfinished_line_and_rejects_overlap(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as folder:
            folder = Path(folder)
            dataset, part = folder / 'tasks.jsonl', folder / 'part.jsonl'
            write_rows(dataset, [dict(task_id='x', condition_prompts={'a': '', 'b': ''})])
            row = dict(task_id='x', condition='a', repeat=0, model_path='model')
            write_rows(part, [row])
            with part.open('a') as handle:
                handle.write('{"task_id":')
            rows, expected = merge([part], dataset, 1)
            self.assertEqual(rows, [row])
            self.assertEqual(expected, 2)
            with self.assertRaises(ValueError):
                merge([part, part], dataset, 1)


if __name__ == '__main__':
    unittest.main()
