"""Repeated semantic pairs must remain paired before family aggregation."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_controlled import analyze
from analyze_semantic_families import summarize_models


class SemanticRepeatTests(unittest.TestCase):
    def test_switch_requires_both_rules_in_the_same_run(self):
        rows=[]
        for task,family in [('one','f1'),('two','f2')]:
            for repeat in range(3):
                for condition in ['nl_only','inputs_only','tests_a','tests_b','explicit_a','explicit_b']:
                    a=(condition=='explicit_a' or (condition=='tests_a' and repeat in [0,2]))
                    b=(condition=='explicit_b' or (condition=='tests_b' and repeat in [1,2]))
                    rows.append(dict(task_id=task,family_id=family,repeat=repeat,condition=condition,
                                     passed=a or b,rule_a_passed=a,rule_b_passed=b))
        result=analyze(rows)['paired_rule_switch']
        self.assertAlmostEqual(result['paired_difference'],1/3)
        self.assertEqual((result['num_tasks'],result['num_clusters'],result['repeats_per_task']),(2,2,3))
        self.assertEqual([result['per_repeat'][str(r)]['target_passed'] for r in range(3)],[0,0,2])
        models={'qwen25':rows,'qwen36':deepcopy(rows),'deepseek':deepcopy(rows)}
        families=summarize_models(models)
        self.assertAlmostEqual(families['families']['f1']['qwen25']['switch'],1/3)
        self.assertEqual(len(families['model_comparisons']),3)
        for comparison in families['model_comparisons']:
            self.assertEqual(comparison['paired_difference'],0)
            self.assertEqual(comparison['repeats_per_task'],3)


if __name__=='__main__':unittest.main()
