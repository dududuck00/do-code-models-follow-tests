"""Paired specifications: 20 semantic families, six input instances per family."""
from __future__ import annotations

import random


def families() -> list[dict]:
    # A and B share the task domain, interface and underspecified description.
    definitions = [
        ("dedup_order", "Return a list containing each distinct integer in data once.",
         "Preserve the order of first occurrence.", "Return the distinct integers in ascending order.",
         "return list(dict.fromkeys(data))", "return sorted(set(data))", "ints"),
        ("duplicate_policy", "Remove duplicate integers from data and return the remaining values in their original order.",
         "Keep the first occurrence of each value.", "Keep only values that occur exactly once in the input.",
         "return list(dict.fromkeys(data))", "return [x for x in data if data.count(x) == 1]", "ints"),
        ("argmax_tie", "Return the zero-based index of a maximum value in the nonempty integer list data.",
         "Choose the first such index.", "Choose the last such index.",
         "return data.index(max(data))", "return len(data) - 1 - data[::-1].index(max(data))", "ints"),
        ("threshold_boundary", "data contains an integer list and a threshold. Return the values meeting the threshold in input order.",
         "Include values equal to the threshold.", "Include only values strictly greater than the threshold.",
         "xs, t = data\nreturn [x for x in xs if x >= t]", "xs, t = data\nreturn [x for x in xs if x > t]", "threshold"),
        ("interval_boundary", "data contains an integer list and two bounds lo and hi with lo < hi. Return the values in that interval in input order.",
         "Include both bounds.", "Include the lower bound and exclude the upper bound.",
         "xs, lo, hi = data\nreturn [x for x in xs if lo <= x <= hi]", "xs, lo, hi = data\nreturn [x for x in xs if lo <= x < hi]", "interval"),
        ("median_choice", "Return an integer median from the nonempty integer list data.",
         "For even-length lists choose the lower middle value.", "For even-length lists choose the upper middle value.",
         "return sorted(data)[(len(data)-1)//2]", "return sorted(data)[len(data)//2]", "ints"),
        ("integer_rounding", "data is a pair of integers a and b, with b positive. Divide a by b and convert the result to an integer.",
         "Round the quotient down toward negative infinity.", "Truncate the quotient toward zero.",
         "a, b = data\nreturn a // b", "a, b = data\nreturn (abs(a)//b) * (-1 if a < 0 else 1)", "division"),
        ("remainder_sign", "data is a pair of integers a and b, with b positive. Return the remainder when dividing a by b.",
         "Use the nonnegative remainder associated with floor division.", "Use the signed remainder associated with truncation toward zero.",
         "a, b = data\nreturn a % b", "a, b = data\nq = (abs(a)//b) * (-1 if a < 0 else 1)\nreturn a - q*b", "division"),
        ("substring_overlap", "data contains a string and a nonempty substring. Count occurrences of the substring in the string.",
         "Count overlapping occurrences.", "Count non-overlapping occurrences from left to right.",
         "s, t = data\nreturn sum(s.startswith(t, i) for i in range(len(s)))", "s, t = data\nreturn s.count(t)", "substring"),
        ("substring_case", "data contains an ASCII string and a nonempty substring. Count non-overlapping occurrences of the substring.",
         "Match letters case-sensitively.", "Match letters case-insensitively.",
         "s, t = data\nreturn s.count(t)", "s, t = data\nreturn s.lower().count(t.lower())", "case"),
        ("space_fields", "Split the string data into fields using spaces and return a list of strings.",
         "Consecutive spaces separate empty fields; preserve leading and trailing empty fields.",
         "Treat consecutive spaces as one separator and omit empty fields.",
         "return data.split(' ')", "return [x for x in data.split(' ') if x]", "spaces"),
        ("normalization", "Normalize the ASCII string data by retaining textual content and removing punctuation and spacing.",
         "Keep decimal digits as part of the normalized text.", "Keep alphabetic letters only.",
         "return ''.join(c for c in data if c.isalnum())", "return ''.join(c for c in data if c.isalpha())", "text"),
        ("sort_ties", "data contains pairs [integer score, string name]. Return the pairs sorted by ascending score.",
         "Preserve input order among equal scores.", "Sort equal scores by ascending name.",
         "return sorted(data, key=lambda x: x[0])", "return sorted(data, key=lambda x: (x[0], x[1]))", "records"),
        ("partial_chunk", "data contains an integer list and a positive chunk size. Split the list into consecutive chunks of that size.",
         "Keep the final chunk even when it is shorter than the requested size.", "Return only chunks of the full requested size.",
         "xs, n = data\nreturn [xs[i:i+n] for i in range(0, len(xs), n)]", "xs, n = data\nreturn [xs[i:i+n] for i in range(0, len(xs)-n+1, n)]", "chunks"),
        ("group_count", "Summarize repeated integers in data as pairs [value, count], following the order in which groups first appear.",
         "Group consecutive equal values into runs.", "Group all occurrences of each value together.",
         "out = []\nfor x in data:\n    if out and out[-1][0] == x:\n        out[-1][1] += 1\n    else:\n        out.append([x, 1])\nreturn out",
         "return [[x, data.count(x)] for x in dict.fromkeys(data)]", "ints"),
        ("graph_direction", "data contains a list of edges [u, v] and a starting vertex. Return the sorted vertices reachable from the start, including the start.",
         "Each edge can be traversed from u to v only.", "Each edge can be traversed in either direction.",
         "edges, start = data\nseen = {start}\nwhile True:\n    new = seen | {v for u, v in edges if u in seen}\n    if new == seen: return sorted(seen)\n    seen = new",
         "edges, start = data\nseen = {start}\nwhile True:\n    new = seen | {v for u, v in edges if u in seen} | {u for u, v in edges if v in seen}\n    if new == seen: return sorted(seen)\n    seen = new", "graph"),
        ("merge_touching", "Return the sorted merged intervals from data. Each interval is [start, end] with start < end; overlapping intervals are combined.",
         "Also merge intervals whose endpoints touch.", "Merge intervals only when their interiors overlap.",
         "out = []\nfor a, b in sorted(data):\n    if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)\n    else: out.append([a, b])\nreturn out",
         "out = []\nfor a, b in sorted(data):\n    if out and a < out[-1][1]: out[-1][1] = max(out[-1][1], b)\n    else: out.append([a, b])\nreturn out", "intervals"),
        ("search_index", "data contains a list of integers and a target that occurs in it. Return the position of its first occurrence.",
         "The first position is numbered zero.", "The first position is numbered one.",
         "xs, target = data\nreturn xs.index(target)", "xs, target = data\nreturn xs.index(target) + 1", "search"),
        ("topk_ties", "data contains a nonempty integer list and k with 1 <= k <= its length. Return the largest values in descending order, using the kth-largest value as the cutoff.",
         "Return exactly k values.", "Include every value equal to the cutoff, even when the result has more than k elements.",
         "xs, k = data\nreturn sorted(xs, reverse=True)[:k]", "xs, k = data\nt = sorted(xs, reverse=True)[k-1]\nreturn sorted((x for x in xs if x >= t), reverse=True)", "topk"),
        ("empty_segments", "data is a string containing comma-separated entries. Trim spaces around each entry and return the entries as a list.",
         "Keep entries that are empty after trimming.", "Discard entries that are empty after trimming.",
         "return [x.strip() for x in data.split(',')]", "return [x.strip() for x in data.split(',') if x.strip()]", "csv"),
    ]
    return [dict(family_id=f, description=d, rule_a=a, rule_b=b,
                 code_a="def solve(data):\n" + "\n".join("    " + l for l in ca.splitlines()) + "\n",
                 code_b="def solve(data):\n" + "\n".join("    " + l for l in cb.splitlines()) + "\n", kind=k)
            for f, d, a, b, ca, cb, k in definitions]


def sample_input(kind: str, rng: random.Random, variant: int):
    scale = 3 + 2 * variant
    xs = [rng.randint(-scale, scale) for _ in range(rng.randint(3, 15 + variant))]
    if kind == "ints": return xs
    if kind == "threshold": return [xs, rng.choice(xs)]
    if kind == "interval":
        lo, hi = sorted(rng.sample(range(-scale-1, scale+2), 2))
        return [xs + [lo, hi], lo, hi]
    if kind == "division": return [rng.randint(-100 * scale, 100 * scale), rng.randint(2, 20)]
    if kind == "substring":
        token = ''.join(rng.choices('abcxy01', k=rng.randint(1, 4)))
        return [token * rng.randint(3, 30), token * rng.randint(2, 3)]
    if kind == "case":
        token = rng.choice(["cat", "Ab", "xy", "Hello"])
        return ["".join(rng.choice([token.lower(), token.upper(), token]) for _ in range(rng.randint(3, 10))), token]
    if kind == "spaces": return " " * rng.randint(0, 2) + (" " * rng.randint(1, 4)).join(str(x) for x in xs) + " " * rng.randint(0, 2)
    if kind == "text": return "".join(rng.choice("abcXYZ0123456789!?,_ ") for _ in range(rng.randint(8, 30)))
    if kind == "records": return [[rng.randint(0, 3), ''.join(rng.choices('abcdef', k=3))] for _ in xs]
    if kind == "chunks": return [xs, rng.randint(2, 7)]
    if kind == "graph":
        vertices = rng.sample(range(30 + scale), rng.randint(4, 10))
        return [[[rng.choice(vertices), rng.choice(vertices)] for _ in range(rng.randint(3, 15))], rng.choice(vertices)]
    if kind == "intervals":
        return [[a, a + rng.randint(1, 4)] for a in xs]
    if kind == "search": return [xs, rng.choice(xs)]
    if kind == "topk": return [xs, rng.randint(1, len(xs)-1)]
    if kind == "csv": return ','.join(rng.choice(['', ' ', str(x), '  '+str(x)+' ']) for x in xs)
    raise ValueError(kind)
