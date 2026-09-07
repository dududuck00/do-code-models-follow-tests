#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tddexp.controlled import PROTOCOL_VERSION, prompt_with_tests
from tddexp.io import write_jsonl
from tddexp.semantic_tasks import families, sample_input


def build_tasks():
    rows = []
    for fi, family in enumerate(families()):
        scope_a, scope_b = {}, {}
        exec(family['code_a'], scope_a)
        exec(family['code_b'], scope_b)
        used = set()
        for variant in range(6):
            rng = random.Random(17000 + fi * 100 + variant)
            cases, common = [], []
            for _ in range(30000):
                x = sample_input(family['kind'], rng, variant)
                key = repr(x)
                if key in used: continue
                a = scope_a['solve'](copy.deepcopy(x))
                b = scope_b['solve'](copy.deepcopy(x))
                if a != b and len(cases) < 33:
                    cases.append((x, a, b)); used.add(key)
                elif a == b and len(common) < 10:
                    common.append((x, a)); used.add(key)
                if len(cases) == 33 and len(common) == 10: break
                if len(cases) == 33 and family['family_id'] == 'search_index': break
            if len(cases) < 33: raise ValueError(f"Insufficient distinguishing cases: {family['family_id']}")
            visible, hidden = cases[:3], cases[3:]
            desc = family['description']
            interfaces = {
                'threshold': 'The argument data is the two-element Python list [values, threshold], where values is a list of integers.',
                'interval': 'The argument data is the three-element Python list [values, lo, hi], where values is a list of integers.',
                'division': 'The argument data is a two-element Python list [a, b].',
                'substring': 'The argument data is a two-element Python list [text, substring], both elements being strings.',
                'case': 'The argument data is a two-element Python list [text, substring], both elements being strings.',
                'chunks': 'The argument data is the two-element Python list [values, size], where values is a list of integers.',
                'graph': 'The argument data is the two-element Python list [edges, start]. Vertices are integers and edges is a list of two-element lists.',
                'search': 'The argument data is the two-element Python list [values, target], where values is a list of integers.',
                'topk': 'The argument data is the two-element Python list [values, k], where values is a list of integers.',
            }
            if family['kind'] in interfaces:
                desc += '\n' + interfaces[family['kind']]
            prompt = f"### Problem\n{desc}\n\n### Function Signature\n```python\ndef solve(data):\n```\n\n### Answer\nReturn a complete Python implementation inside a Python markdown block.\n```python\n"
            tests = {"nl_only": [], "inputs_only": [f"solve({x!r})" for x, _, _ in visible],
                     "tests_a": [f"assert solve({x!r}) == {a!r}" for x, a, _ in visible],
                     "tests_b": [f"assert solve({x!r}) == {b!r}" for x, _, b in visible],
                     "explicit_a": [], "explicit_b": []}
            prompts = {c: prompt_with_tests(prompt, ts, c == 'inputs_only') for c, ts in tests.items()}
            for side in ['a', 'b']:
                prompts['explicit_' + side] = prompt.replace(desc, desc + "\n" + family['rule_' + side], 1)
            rows.append({"protocol_version": PROTOCOL_VERSION, "task_id": f"semantic::{family['family_id']}::{variant}",
                         "family_id": family['family_id'], "instance": variant, "source_dataset": "semantic_pairs",
                         "entry_point": "solve", "signature": "def solve(data):", "prompt": prompt,
                         "condition_prompts": prompts, "condition_tests": tests,
                         "reference_code": family['code_a'], "reference_a": family['code_a'], "reference_b": family['code_b'],
                         "rule_a": family['rule_a'], "rule_b": family['rule_b'],
                         "heldout_a": [f"assert solve({x!r}) == {a!r}" for x, a, _ in hidden],
                         "heldout_b": [f"assert solve({x!r}) == {b!r}" for x, _, b in hidden],
                         "heldout_common": [f"assert solve({x!r}) == {a!r}" for x, a in common]})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default='data/controlled/semantic')
    a = p.parse_args()
    rows = build_tasks()
    out = Path(a.output_dir)
    write_jsonl(out / 'tasks.jsonl', rows)
    summary = {'task_pairs': len(rows), 'semantic_families': len(families()), 'instances_per_family': 6,
               'visible_tests': 3, 'discriminating_hidden_inputs_per_pair': 30,
               'analysis_cluster': 'family_id', 'source': 'authored paired specifications'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
