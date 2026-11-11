import json
import re
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.data_generation.build_back import split_pairs
from src.data_generation.generalized_pairs.catalog import Block, Phrase
from src.data_generation.generalized_pairs.contrasts import ContrastPair
from src.data_generation.generalized_pairs.generate import (
    MAPPINGS, check_ordinary_capacity, generate_pairs, load_catalog, prepare_quota_streams, write_pairs,
)
from src.data_generation.generalized_pairs.sql_types import Distribution, apportion, subtype_names


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
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
        command = [sys.executable, str(MAPPINGS.with_name("generate.py")), "--count", "9",
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

    def test_cli_requires_seed_and_sql_types_before_writing_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            output = Path(folder) / "pairs.json"
            for missing, supplied in (("--seed", ["--sql-types", "sql_types.json"]), ("--sql-types", ["--seed", "0"])):
                with self.subTest(missing=missing):
                    result = subprocess.run(
                        [sys.executable, "-m", "src.data_generation.generalized_pairs.generate",
                         "--count", "1", "--output", str(output), *supplied],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(f"the following arguments are required: {missing}", result.stderr)
                    self.assertFalse(output.exists())

    def test_description_is_optional_string_metadata(self):
        profile = {"family_percent": {"rows": 100}, "example_weights": {"ordinary": 1}}
        expected = Distribution.load(self.path(profile))
        self.assertEqual(Distribution.load(self.path({**profile, "description": "단순 조회"})), expected)
        for value in (None, 1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "description must be a string"):
                Distribution.load(self.path({**profile, "description": value}))
        with self.assertRaisesRegex(ValueError, "unknown family_percent key: description"):
            Distribution.load(self.path({**profile, "family_percent": {"rows": 100, "description": "설명"}}))

    def test_empty_template_lists_every_option_and_uses_automatic_percentages(self):
        path = DATA_DIR / "example" / "sql_types.json"
        types = subtype_names()
        families = dict.fromkeys(types)
        self.assertEqual(json.loads(path.read_bytes()), {
            "description": "",
            "family_percent": families,
            "example_weights": {"ordinary": 1, "contrast": None},
            "subtype_percent": {family: dict.fromkeys(names) for family, names in types.items()},
        })
        minimal = self.path({"family_percent": families, "example_weights": {"ordinary": 1}})
        self.assertEqual(Distribution.load(path), Distribution.load(minimal))
        rows = list(generate_pairs(8, 42, sql_types=path))
        self.assertEqual(rows, list(generate_pairs(8, 42, sql_types=minimal)))
        self.assertEqual(Counter(emitted_type(row["output"])[0] for row in rows), dict.fromkeys(types, 2))

    def test_nullable_weights_still_require_a_positive_number(self):
        profile = {"family_percent": {"rows": 100}}
        contrast_only = Distribution.load(self.path({**profile, "example_weights": {"contrast": 1}}))
        self.assertEqual(Distribution.load(self.path({
            **profile, "example_weights": {"ordinary": None, "contrast": 1},
        })), contrast_only)
        for weights in ({"ordinary": None, "contrast": None}, {"ordinary": 1, "contrast": False},
                        {"ordinary": -1, "contrast": 1}, {"ordinary": "1"}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                Distribution.load(self.path({**profile, "example_weights": weights}))

    def test_all_653_keys_reach_their_sql(self):
        keys = subtype_names()
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

    def test_simple_rows_profile_builds_100_training_pairs(self):
        options = DATA_DIR / "example_for_100" / "sql_types_simple_rows_for_100_test.json"
        rows = list(generate_pairs(103, 42, sql_types=options))
        training, validation = split_pairs(rows, seed=42, validation_ratio=0.05)
        self.assertEqual((len(training), len(validation)), (100, 3))
        self.assertEqual(Counter(emitted_type(row["output"]) for row in rows), {
            ("rows", "where0/plain"): 72, ("rows", "where0/limit"): 31,
        })
        self.assertTrue({row["output"] for row in training}.isdisjoint(row["output"] for row in validation))

    def test_large_quotas_and_capacity_without_generating_a_dataset(self):
        raw = {
            "family_percent": {"rows": 50, "aggregate": 25, "group": 12.5, "scalar": 12.5},
            "example_weights": {"ordinary": 99, "contrast": 1},
            "subtype_percent": {"rows": {"where0/plain": 0.01, "where1:>/order_limit": 2}},
        }
        profile = Distribution.load(self.path(raw))
        catalog = load_catalog(MAPPINGS)
        families = apportion(500_000, profile.families)
        self.assertEqual(families, {"rows": 250_000, "aggregate": 125_000, "group": 62_500, "scalar": 62_500})
        quotas = {f: apportion(n, profile.targets[f]) for f, n in families.items()}
        self.assertEqual(quotas["rows"], {"where0/plain": 25, "where1:>/order_limit": 5_000, None: 244_975})
        streams, _ = prepare_quota_streams(500_000, 42, catalog, None, profile)
        self.assertEqual(sum(s.quota for s in streams), 500_000)
        paired = sum(2 * n for s in streams if s.paired for _, n in s.stream)
        # Each family rounds its contrast quota down to a whole number of pairs.
        self.assertEqual(paired, sum(count // 100 // 2 * 2 for count in families.values()))

        # At the same scale, 0.03% of the rows family requires 75 different bare SELECTs/wordings.
        raw["subtype_percent"]["rows"]["where0/plain"] = 0.03
        with self.assertRaisesRegex(ValueError, "rows/where0/plain.*75 ordinary records.*at most 74"):
            next(generate_pairs(500_000, 42, sql_types=self.path(raw)))

    def test_partially_matched_sqls_fill_their_remaining_ordinary_slots(self):
        for key, count, expected_pairs in (("SUM/where1", 102, 5), ("COUNT/where1", 75, 4)):
            path = self.path({"family_percent": {"aggregate": 100}, "example_weights": {"ordinary": 8, "contrast": 1},
                              "subtype_percent": {"aggregate": {key: 100}}})
            for seed in (2, 42):
                with self.subTest(key=key, seed=seed):
                    rows = list(generate_pairs(count, seed, sql_types=path, max_questions_per_sql=3))
                    self.assertEqual(len(rows), count)
                    self.assertEqual(len({(r["output"], r["input"]) for r in rows}), count)
                    self.assertEqual({emitted_type(r["output"]) for r in rows}, {("aggregate", key)})
                    parents = [rows[i:i + 3] for i in range(0, count, 3)]
                    self.assertEqual(len({p[0]["output"] for p in parents}), count // 3)
                    for i, p in enumerate(parents, 1):
                        self.assertEqual(len({r["output"] for r in p}), 1)
                        self.assertEqual([r["id"] for r in p], [f"{i}.{child}" for child in (1, 2, 3)])
                    peers = {r["id"]: r for r in rows}
                    paired = [r for r in rows if "contrast" in r]
                    self.assertEqual(len(paired), expected_pairs)
                    for r in paired:
                        parent, child = r["id"].split(".")
                        peer_id = r["contrast"]["paired_id"]
                        self.assertEqual(peer_id, f"{int(parent)-1}.{child}")
                        self.assertNotEqual(peers[peer_id]["output"], r["output"])
                    self.assertTrue(any("contrast" in p[0] and "contrast" not in p[-1] for p in parents))

    def test_only_the_scarce_side_of_a_cross_subtype_pair_takes_fillers(self):
        having = "SUM/where0/having:>/plain"
        where = "SUM/where1/having:none/plain"
        path = self.path({"family_percent": {"group": 100}, "example_weights": {"ordinary": 8, "contrast": 1},
                          "subtype_percent": {"group": {having: 30, where: 70}}})
        rows = list(generate_pairs(30, 42, sql_types=path, max_questions_per_sql=3))
        self.assertEqual(Counter(emitted_type(r["output"])[1] for r in rows), {having: 9, where: 21})
        right, = [r for r in rows if "contrast" in r]
        left = next(r for r in rows if r["id"] == right["contrast"]["paired_id"])
        self.assertEqual(sum(r["output"] == left["output"] for r in rows), 1)
        siblings = [r for r in rows if r["output"] == right["output"]]
        self.assertEqual([r["id"].split(".")[1] for r in siblings], ["1", "2", "3"])
        self.assertTrue(all("contrast" not in r for r in siblings[1:]))

    def test_matched_wording_is_excluded_from_actual_ordinary_capacity(self):
        left = Phrase("SELECT 1", ("shared wording",))
        pair = ContrastPair("boundary", left, Phrase("SELECT 2", ("other side",)))
        block = Block(Phrase(left.sql, ("shared wording", "ordinary wording")), {}, (0, 0, False, False))
        held = {left.sql: (pair, 1)}
        self.assertEqual(check_ordinary_capacity([block], held, 1, 3, 42, "rows", None),
                         {left.sql: ("ordinary wording",)})
        with self.assertRaisesRegex(ValueError, "2 ordinary records.*at most 1 available"):
            check_ordinary_capacity([block], held, 2, 3, 42, "rows", None)

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
                next(generate_pairs(75, seed, sql_types=path))

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
