import json
import re
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.data_generation.generalized_pairs.generate import MAPPINGS, generate_pairs, load_catalog, plan_contrasts, write_pairs
from src.data_generation.generalized_pairs.sql_types import Distribution, SQL_TYPES, apportion, subtype_blocks, subtype_names


PREDICATE = re.compile(r"<TABLE1\.COL\d+, \w+> (>=|<=|!=|=|<|>|IN|LIKE|BETWEEN) ")


def emitted_type(sql):
    """Identify a subtype from the emitted SQL, independently of the generator's slots."""
    if "(SELECT " in sql:
        start = sql.index("(SELECT ")
        depth = 0
        for end in range(start, len(sql)):
            depth += (sql[end] == "(") - (sql[end] == ")")
            if not depth:
                break
        inner = len(PREDICATE.findall(sql[start:end]))
        outer = len(PREDICATE.findall(sql[end + 1:]))
        op, fn = re.search(r" (>=|<=|!=|=|<|>) \(SELECT (AVG|MIN|MAX)\(", sql).groups()
        return "scalar", f"{fn}/{op}/inner{inner}/outer{outer}"
    mode = "_".join(k.lower() for k in ("ORDER", "LIMIT") if f" {k} " in sql) or "plain"
    where = re.split(r" GROUP BY | ORDER BY | LIMIT ", sql.partition(" WHERE ")[2])[0]
    ops = PREDICATE.findall(where)
    n = len(ops)
    agg = re.search(r"\{(COUNT|AVG|SUM|MIN|MAX)\(", sql)
    if " GROUP BY " in sql:
        having = re.search(r" HAVING .+?\) (>=|<=|!=|=|<|>) ", sql)
        return "group", f"{agg[1]}/where{n}/having:{having[1] if having else 'none'}/{mode}"
    if agg:
        return "aggregate", f"{agg[1]}/where{n}"
    suffix = ""
    if n == 1:
        suffix = ":" + ops[0]
    elif n > 1:
        suffix = ":OR" if " OR " in where else ":AND"
    return "rows", f"where{n}{suffix}/{mode}"


class TypeDistributionTest(unittest.TestCase):
    def path(self, data):
        with tempfile.NamedTemporaryFile(dir=Path(__file__).parent, delete=False) as file:
            file.write(json.dumps(data).encode())
        path = Path(file.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_cli_takes_run_options_and_reads_proportions_from_json(self):
        profile = {"family_percent": {"rows": 100}, "example_weights": {"ordinary": 1},
                   "subtype_percent": {"rows": {"where1:>/plain": 100}}}
        path, output = self.path(profile), self.path([])
        command = [sys.executable, str(SQL_TYPES.with_name("generate.py")), "--count", "9",
                   "--max-questions-per-sql", "1", "--seed", "42", "--output", str(output),
                   "--sql-types", str(path)]
        subprocess.run(command, check=True, capture_output=True)
        rows = json.loads(output.read_bytes())
        self.assertEqual(len(rows), 9)
        self.assertEqual(len({row["output"] for row in rows}), 9)
        self.assertEqual({emitted_type(row["output"]) for row in rows}, {("rows", "where1:>/plain")})
        self.assertEqual(rows, list(generate_pairs(9, 42, max_questions_per_sql=1, sql_types=path)))

        for flag in ("--family-percent", "--subtype-percent", "--example-weights"):
            with self.subTest(flag=flag):
                result = subprocess.run([*command, flag, "{}"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("unrecognized arguments", result.stderr)
        for key, value in (("count", 9), ("max_questions_per_sql", 1), ("seed", 42), ("output", "sample.json")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"unknown configuration key: {key}"):
                Distribution.load(self.path({**profile, key: value}))

    def test_all_653_keys_reach_their_sql(self):
        keys = subtype_names()
        bundled = json.loads(SQL_TYPES.read_bytes())["subtype_percent"]
        self.assertEqual({f: set(v) for f, v in bundled.items()}, {f: set(v) for f, v in keys.items()})
        def percentages(names):
            # Allocate 10,000 basis points equally with at most a one-point difference.
            each, extra = divmod(10_000, len(names))
            return {name: (each + (i < extra)) / 100 for i, name in enumerate(names)}
        path = self.path({
            "family_percent": {"rows": 8.58, "aggregate": 2.30, "group": 64.32, "scalar": 24.80},
            "example_weights": {"ordinary": 1, "contrast": 0},
            "subtype_percent": {f: percentages(v) for f, v in keys.items()},
        })
        rows = list(generate_pairs(653, 42, sql_types=path, max_questions_per_sql=1))
        self.assertEqual(Counter(emitted_type(r["output"]) for r in rows),
                         {(f, key): 1 for f, ids in keys.items() for key in ids})

    def test_bundled_preview_guarantees_simple_queries(self):
        rows = list(generate_pairs(1000, 42, max_questions_per_sql=3))
        types = Counter(emitted_type(row["output"]) for row in rows)
        profile = Distribution.load(SQL_TYPES)
        targets = apportion(500, profile.targets["rows"])
        self.assertEqual(len(targets), 56)
        self.assertTrue(all(n > 0 for n in targets.values()))
        self.assertEqual({key: n for (family, key), n in types.items() if family == "rows"}, targets)
        self.assertEqual(sum(n for key, n in targets.items() if key.startswith("where1:") and key.endswith("/plain")), 180)
        self.assertEqual(targets["where1:=/plain"], 40)
        self.assertEqual(targets["where1:>/plain"], 25)
        self.assertLessEqual(max(Counter(row["output"] for row in rows).values()), 3)

    def test_large_profile_allocations_without_generating_a_dataset(self):
        profile = Distribution.load(SQL_TYPES.with_name("sql_types_large.json"))
        catalog = load_catalog(MAPPINGS)
        families = apportion(500_000, profile.families)
        self.assertEqual(families, {"rows": 250_000, "aggregate": 125_000, "group": 62_500, "scalar": 62_500})
        quotas = {f: apportion(n, profile.targets[f]) for f, n in families.items()}
        self.assertEqual(sum(n for k, n in quotas["rows"].items() if k.startswith("where1:")), 177_500)
        plans, _, reserved_count = plan_contrasts(profile, catalog, quotas, 42, None)
        paired = sum(2 * n for group in plans.values() for _, n in group)
        self.assertLess(abs(paired / 500_000 - 1 / 9), .001)
        sizes = Counter()
        for block in catalog.rows():
            for key, part in subtype_blocks("rows", block):
                sizes[key] += part.size
        for key, n in quotas["rows"].items():
            self.assertLessEqual(n, (sizes[key] - reserved_count["rows", key]) * 64, key)

    def test_percentages_apply_to_records_including_contrasts(self):
        path = self.path({"family_percent": {"rows": 40, "aggregate": 60},
                          "example_weights": {"ordinary": 8, "contrast": 1},
                          "subtype_percent": {"rows": {"where1:>/plain": 10, "where0/order_limit": 0}}})
        rows = list(generate_pairs(1000, 42, sql_types=path))
        self.assertEqual(rows, list(generate_pairs(1000, 42, sql_types=path)))
        counts = Counter(emitted_type(r["output"]) for r in rows)
        self.assertEqual(sum(n for (f, _), n in counts.items() if f == "rows"), 400)
        self.assertEqual(sum(n for (f, _), n in counts.items() if f == "aggregate"), 600)
        self.assertEqual(counts["rows", "where1:>/plain"], 40)
        self.assertNotIn(("rows", "where0/order_limit"), counts)
        self.assertTrue(any("contrast" in r for r in rows))
        parents = {}
        for r in rows:
            parent = r["id"].split(".")[0]
            self.assertEqual(parents.setdefault(r["output"], parent), parent)

    def test_explicit_equal_shares_and_partial_family_defaults(self):
        for shares, expected in [
            (dict.fromkeys(("rows", "aggregate", "group", "scalar"), 25),
             {"rows": 25, "aggregate": 25, "group": 25, "scalar": 25}),
            ({"rows": 40, "aggregate": None, "scalar": None}, {"rows": 40, "aggregate": 30, "scalar": 30}),
        ]:
            with self.subTest(shares=shares):
                path = self.path({"family_percent": shares, "example_weights": {"ordinary": 8, "contrast": 1}})
                rows = generate_pairs(100, 42, sql_types=path)
                self.assertEqual(Counter(emitted_type(r["output"])[0] for r in rows), expected)

    def test_cross_subtype_pairs_charge_both_parents(self):
        types = {"AVG/>/inner0/outer1": 50, "AVG/>/inner1/outer1": 50}
        profile = {"family_percent": {"scalar": 100}, "example_weights": {"contrast": 1},
                   "subtype_percent": {"scalar": types}}
        rows = list(generate_pairs(30, 42, sql_types=self.path(profile)))
        self.assertEqual(Counter(emitted_type(r["output"])[1] for r in rows), {k: 15 for k in types})
        by_id = {r["id"]: r for r in rows}
        partners = [r for r in rows if "contrast" in r]
        self.assertEqual(len(partners), 15)
        for r in partners:
            peer = by_id[r["contrast"]["paired_id"]]
            self.assertNotEqual(emitted_type(peer["output"]), emitted_type(r["output"]))
        types.update({"AVG/>/inner0/outer1": 100, "AVG/>/inner1/outer1": 0})
        with self.assertRaisesRegex(ValueError, "complete pairs|contrast records"):
            list(generate_pairs(30, 42, sql_types=self.path(profile)))
        profile["example_weights"]["ordinary"] = 1
        only = list(generate_pairs(30, 42, sql_types=self.path(profile)))
        self.assertTrue(all(emitted_type(r["output"])[1] == "AVG/>/inner0/outer1" for r in only))
        self.assertFalse(any("contrast" in r for r in only))

    def test_scarce_type_uses_all_available_wording_in_any_source_order(self):
        path = self.path({"family_percent": {"rows": 100}, "example_weights": {"ordinary": 1},
                          "subtype_percent": {"rows": {"where0/plain": 100}}})
        for seed in (0, 2):
            rows = list(generate_pairs(74, seed, sql_types=path))
            self.assertEqual(len({(r["output"], r["input"]) for r in rows}), 74)
            with self.assertRaisesRegex(ValueError, "at most 74 available"):
                list(generate_pairs(75, seed, sql_types=path))

    def test_config_errors_and_insufficient_capacity_preserve_output(self):
        profile = {"family_percent": {"rows": 100}, "example_weights": {"ordinary": 1}}
        for change in ({"family_percent": {"rows": 40}}, {"family_percent": {"typo": 100}},
                       {"example_weights": {"ordinary": 0}}, {"subtype_percent": {"rows": {"typo": 0}}},
                       {"subtype_percent": {"rows": {"where0/plain": -2}}},
                       {"subtype_percent": {"rows": {"where0/plain": True}}},
                       {"subtype_percent": {"rows": {"where0/plain": 60, "where0/limit": 50}}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Distribution.load(self.path({**profile, **change}))
        profile["subtype_percent"] = {"rows": {"where0/plain": 100}}
        path = self.path(profile)
        output = self.path(["existing"])
        with self.assertRaisesRegex(ValueError, "rows/where0/plain"):
            write_pairs(output, generate_pairs(5, 42, sql_types=path, max_questions_per_sql=1))
        self.assertEqual(json.loads(output.read_bytes()), ["existing"])
