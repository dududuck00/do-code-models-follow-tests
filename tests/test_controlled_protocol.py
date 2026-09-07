import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tddexp.code_utils import assemble_candidate
from tddexp.controlled import choose_test_suites, filter_evalplus_test, input_key, same_test_set
from tddexp.data import load_tasks
from tddexp.prompts import build_prompt, select_condition_tests, shuffle_test_outputs
from tddexp.statistics import paired_effect


class ControlledProtocolTests(unittest.TestCase):
    def test_whitespace_and_assertion_message_do_not_disable_shuffling(self):
        tests = ['assert f(1)==2, "message"', 'assert f(2) == 3']
        shuffled = shuffle_test_outputs(tests)
        self.assertFalse(same_test_set(tests, shuffled))
        scope = {'f': lambda x: x + 1}
        for test in shuffled:
            with self.assertRaises(AssertionError): exec(test, scope)

    def test_impossible_shuffle_is_explicit(self):
        for tests in [['assert f(1) == True'], ['assert f(1)==True', 'assert f(2)==True'], ['assert f(1)']]:
            with self.assertRaises(ValueError): shuffle_test_outputs(tests)

    def test_evalplus_filter_preserves_input_output_alignment(self):
        code = 'def check(f):\n inputs = [[1], [2], [3]]\n results = [4, 5, 6]\n for inp, out in zip(inputs, results):\n  assert f(*inp) == out\n'
        filtered, counts = filter_evalplus_test(code, {input_key((2,))})
        seen = []
        scope = {}; exec(filtered, scope)
        scope['check'](lambda x: seen.append(x) or x+3)
        self.assertEqual(seen, [1, 3]); self.assertEqual(counts['removed'], 1)

    def test_evalplus_filter_supports_computed_reference_outputs(self):
        filtered, counts = filter_evalplus_test('inputs = [[1], [2]]\nfor args in inputs:\n assert sum(args) > 0\n', {input_key((1,))})
        self.assertEqual(counts['remaining'], 1)
        scope = {}; exec(filtered, scope); self.assertEqual(scope['inputs'], [[2]])

    def test_complete_solution_keeps_helpers_and_main(self):
        source = 'def solve():\n return helper()\n\ndef helper():\n return 7\n\nif __name__ == "__main__":\n print(solve())\n'
        self.assertEqual(assemble_candidate('', '```python\n'+source+'```', 'solve', False), source)

    def test_union_coverage_and_length_matching(self):
        tests = ['a'*10, 'b'*10, 'c'*10, 'd'*10]
        kills = [{'p1'}, {'p1'}, {'p2'}, {'p3'}]
        selected = choose_test_suites(tests, kills)
        union = lambda key: len(set().union(*(kills[i] for i in selected[key])))
        self.assertEqual(union('high'), 3); self.assertEqual(union('low'), 2)

    def test_repetitions_are_not_new_tasks(self):
        rows = [dict(task_id=str(i), repeat=r, condition=c, passed=(c=='target' and i==0))
                for i in range(10) for r in range(3) for c in ['base', 'target']]
        result = paired_effect(rows, 'base', 'target', bootstrap_samples=100)
        self.assertEqual(result['num_tasks'], 10)
        self.assertAlmostEqual(result['paired_difference'], .1)
        self.assertEqual(result['ci95'], paired_effect(list(reversed(rows)), 'base', 'target', bootstrap_samples=100)['ci95'])
        with self.assertRaises(ValueError): paired_effect(rows + [rows[0]], 'base', 'target')

    def test_target_signature_and_helpers(self):
        tasks = {t.task_id: t for t in load_tasks('data/humanevalplus.jsonl', 'humanevalplus')}
        task = tasks['HumanEval/10']
        self.assertIn('def make_palindrome(', task.signature)
        scope = {}; exec(task.canonical_code, scope)
        self.assertEqual(scope['make_palindrome']('cat'), 'catac')
        mbpp = {t.task_id: t for t in load_tasks('data/mbppplus.jsonl', 'mbppplus')}
        self.assertEqual(mbpp['6'].entry_point, 'differ_At_One_Bit_Pos')

    def test_semantic_family_controls_inference(self):
        rows = [dict(task_id=str(i), family_id=str(i//6), repeat=0, condition=c,
                     passed=(c=='target' and i<6)) for i in range(12) for c in ['base','target']]
        result = paired_effect(rows,'base','target',cluster_field='family_id',bootstrap_samples=100)
        self.assertEqual(result['num_clusters'],2)
        self.assertIsNone(result['per_repeat']['0']['exact_mcnemar_p'])

    def test_deepseek_tokenizer_preserves_python_whitespace(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ROOT / 'data/model_tokenizers/deepseek-coder-6.7b-instruct')
        code = 'def f(x):\n    return x + 1\n'
        self.assertEqual(tokenizer.decode(tokenizer(code, add_special_tokens=False)['input_ids']), code)


if __name__ == '__main__': unittest.main()
