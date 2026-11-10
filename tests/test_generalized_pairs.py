import json
from collections import Counter
from fractions import Fraction
from functools import partial
from itertools import groupby, islice, repeat
from math import prod
from pathlib import Path
import random
import re
import sqlite3
from tempfile import NamedTemporaryFile
import unittest

from src.data_generation.generalized_pairs.catalog import Block, Catalog, Column, FIELD, Phrase
from src.data_generation.generalized_pairs.generate import (
    MAPPINGS, balanced_batches, contrast_catalog, generate_pairs, question_budget, write_pairs,
)
from src.data_generation.generalized_pairs.contrasts import (
    average_scope_pairs, boundary_pairs, filter_level_pairs, rank_pairs,
)
from src.data_generation.generalized_pairs.sql_types import Distribution


COLUMN = re.compile(r"TABLE1\.COL(\d+), (numeric|string|date)")
VALUE = re.compile(r"\[(?:TABLE1\.COL\d+|AGG_VALUE), (numeric|string|date), #([123])\]")
QUESTION_TAG = re.compile(r"<[^>]+>|\{[^}]+\}|\[[^\]]+\]")


def sqlite_sql(sql):
    """Substitute test data into the tags, retaining the generator's quoting."""
    values = {"numeric": "2", "string": "ab", "date": "2025-01-02"}
    sql = VALUE.sub(lambda m: values[m[1]], sql)
    sql = COLUMN.sub(lambda m: "c" + m[1], sql)
    sql = re.sub(r"<(c\d+)>", r"\1", sql.replace("<TABLE1>", "sample"))
    return sql.replace("{", "").replace("}", "")


class GeneralizedPairsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=Path(__file__).parent, delete=False) as file:
            cls.automatic_profile = Path(file.name)
            cls.addClassCleanup(cls.automatic_profile.unlink, missing_ok=True)
            json.dump({"family_percent": {"rows": None, "aggregate": None, "group": None, "scalar": None},
                       "example_weights": {"ordinary": 8, "contrast": 1}}, file)
        cls.automatic_pairs = staticmethod(partial(generate_pairs, sql_types=cls.automatic_profile))

    def setUp(self):
        self.mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
        self.catalog = Catalog(self.mappings)

    def temp_file(self):
        with NamedTemporaryFile(dir=Path(__file__).parent, delete=False) as handle:
            path = Path(handle.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_seed_prefix_and_wording_independent_sql_ids(self):
        records = list(self.automatic_pairs(37, 18))
        self.assertEqual(records[:17], list(self.automatic_pairs(17, 18)))
        self.assertNotEqual(records, list(self.automatic_pairs(37, 19)))
        for start in range(0, 36, 3):
            group = records[start:start + 3]
            parent = start // 3 + 1
            self.assertEqual([r["id"] for r in group], [f"{parent}.{i}" for i in (1, 2, 3)])
            self.assertEqual(len({r["output"] for r in group}), 1)
            self.assertEqual(len({r["input"] for r in group}), 3)
        path = self.temp_file()
        frames_to_edit = list(self.mappings["questions"].values()) + [
            self.mappings["intent"][key] for key in ("count", "group_count", "average_comparison")
        ]
        for frames in frames_to_edit:
            frames[:] = ["please answer this question. " + frame for frame in frames]
        path.write_text(json.dumps(self.mappings), encoding="utf-8")
        revised = list(self.automatic_pairs(37, 18, path))
        for original, changed in zip(records, revised):
            self.assertEqual((original["id"], original["output"]), (changed["id"], changed["output"]))
            self.assertEqual(changed["input"], "please answer this question. " + original["input"])

    def test_each_available_structure_gets_a_turn(self):
        for seed in (0, 42):
            sample = list(self.automatic_pairs(72, seed, max_questions_per_sql=1))
            matched = {key for r in sample if "contrast" in r for key in (r["id"], r["contrast"]["paired_id"])}
            sample = [r for r in sample if r["id"] not in matched]
            for offset, count, turns in ((0, 4, 16), (2, 3, 12)):
                shapes = set()
                for r in sample[offset::4][:turns]:
                    sql = r["output"]
                    # One column comparison per row predicate, even for BETWEEN and IN.
                    where = sql.split(" WHERE ")[1] if " WHERE " in sql else ""
                    where = re.split(" GROUP BY | ORDER BY | LIMIT ", where)[0]
                    filters = len(re.findall(r"<TABLE1\.COL\d+, \w+> ", where))
                    shapes.add((filters, " ORDER BY " in sql, " LIMIT " in sql))
                self.assertEqual(shapes, {
                    (n, order, limit) for n in range(count)
                    for order in (False, True) for limit in (False, True)
                })

    def test_seed_selects_frames_and_nested_expressions_independently(self):
        frame = Phrase("example sql", ("one {clause}", "two {clause}", "three {clause}"))
        clause = Phrase("", ("with {op}", "when {op}")).fill(op=Phrase("", ("above", "over")))
        block = Block(frame.fill(clause=clause), {}, (0, 0, False, False))
        def questions(seed, limit):
            return list(block.source(0, limit, seed=seed)[1])

        all_forms = questions(42, 100)
        self.assertEqual(set(all_forms), {
            f"{frame} {clause} {op}" for frame in ("one", "two", "three")
            for clause in ("with", "when") for op in ("above", "over")
        })
        self.assertEqual(len(all_forms), 12)
        self.assertEqual(len({q.split()[0] for q in all_forms[:3]}), 3)
        self.assertEqual(all_forms[:3], questions(42, 3))
        self.assertNotEqual(all_forms, questions(43, 100))

    def test_per_sql_limit_and_sequential_ids(self):
        first_sources = [r["output"] for r in self.automatic_pairs(17, 42, max_questions_per_sql=1)]
        for limit in (1, 2, 5, 8):
            records = list(self.automatic_pairs(17, 42, max_questions_per_sql=limit))
            self.assertEqual(len(records), 17)
            for parent, (sql, rows) in enumerate(groupby(records, key=lambda r: r["output"]), 1):
                group = list(rows)
                self.assertEqual(sql, first_sources[parent - 1])
                self.assertLessEqual(len(group), limit)
                self.assertEqual(len({r["input"] for r in group}), len(group))
                self.assertEqual([r["id"] for r in group], [f"{parent}.{i+1}" for i in range(len(group))])
            self.assertEqual(records[limit - 1]["id"], f"1.{limit}")
        for invalid in (0, -1):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                list(self.automatic_pairs(0, 42, max_questions_per_sql=invalid))

    def test_exhausted_question_forms_do_not_repeat(self):
        self.mappings["questions"]["all_rows"] = [
            "list {*} from <TABLE1>{where}.{order}{limit}",
            "list {*} from <TABLE1>{where}.{order}{limit}",
        ]
        block = next(Catalog(self.mappings).rows())
        sql, questions = block.source(0, max_questions_per_sql=10, seed=0)
        self.assertEqual(sql, "SELECT {*} FROM <TABLE1>")
        self.assertEqual(list(questions), ["list {*} from <TABLE1>."])

    def test_json_count_and_failed_write(self):
        path = self.temp_file()
        for count in (0, 1, 4):
            records = list(self.automatic_pairs(count, 2))
            write_pairs(path, iter(records))
            self.assertEqual(json.loads(path.read_bytes()), records)
        before = path.read_bytes()
        for bad_count in (-1, 10**30):
            with self.assertRaises(ValueError):
                write_pairs(path, self.automatic_pairs(bad_count, 42))
            self.assertEqual(path.read_bytes(), before)

    def test_thousand_pairs_compile_and_tags_match(self):
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.execute("CREATE TABLE sample (c1, c2, c3, c4, c5, c6)")
        sample = list(self.automatic_pairs(1000, 42, max_questions_per_sql=3))
        self.assertEqual(len({record["id"] for record in sample}), 1000)
        parents = {}
        for record in sample:
            sql, question = record["output"], record["input"]
            parent = record["id"].split(".")[0]
            self.assertEqual(parents.setdefault(sql, parent), parent)
            db.execute(sqlite_sql(sql))
            prose = QUESTION_TAG.sub("", question)
            self.assertEqual(prose, prose.lower())
            for tag in QUESTION_TAG.findall(question):
                valid = tag in ("<TABLE1>", "{*}") or COLUMN.fullmatch(tag[1:-1]) or VALUE.fullmatch(tag)
                self.assertTrue(valid, tag)
            sql_tokens = COLUMN.sub("", VALUE.sub("", sql))
            self.assertEqual(sql_tokens, sql_tokens.upper())
            self.assertIsNone(FIELD.search(sql + question))
            self.assertEqual(set(COLUMN.findall(sql)), set(COLUMN.findall(question)))
            ids = list(dict.fromkeys(int(n) for n, _ in COLUMN.findall(question)))
            self.assertEqual(ids, list(range(1, len(ids) + 1)))
            self.assertLessEqual(len(ids), 6)
            self.assertEqual({v[0] for v in VALUE.finditer(question)},
                             {v[0] for v in VALUE.finditer(sql)})
        self.assertGreaterEqual(len(parents), 334)

    def test_scalar_population_is_independent_of_returned_rows(self):
        target, measure, category = Column(1, "string"), Column(2, "numeric"), Column(3, "string")
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.execute("CREATE TABLE sample (c1 TEXT, c2 INTEGER, c3 TEXT)")
        db.executemany("INSERT INTO sample VALUES (?, ?, ?)",
                       [("a", 1, "ab"), ("b", 5, "ab"), ("c", 12, "z")])
        for inner, outer, shared, expected in (
            ((), (category,), False, []),          # outer subset vs global mean of 6
            ((category,), (), False, ["b", "c"]),  # all rows vs subset mean of 3
            ((category,), (category,), True, ["b"]),
        ):
            block = self.catalog.scalar_block(target, measure, inner, outer, shared)
            for i in range(block.size):
                sql, questions = block.source(i, seed=0)
                if "> (SELECT AVG" in sql and f"{category.ref} = " in sql:
                    break
            else:
                self.fail("Missing scalar AVG with an equality filter")
            self.assertEqual([row[0] for row in db.execute(sqlite_sql(sql))], expected)
            for question in questions:
                self.assertEqual(question.count(category.value(1).words[0]), 1)

    def test_value_order_and_like_wildcards(self):
        col = Column(1, "string")
        predicates = self.catalog.predicates(col)
        for key, pattern in (("starts with", "{a}%"), ("ends with", "%{a}"), ("contains", "%{a}%")):
            pred = next(p for p in predicates if key in p.words[0])
            self.assertEqual(pred.sql, f"{col.ref} LIKE '" + pattern.format(a=col.value(1).words[0]) + "'")
        values = next(p for p in predicates if " IN " in p.sql and "#3" in p.sql)
        for text in (values.sql, *values.distinct_questions({}, 3, random.Random(42))):
            self.assertEqual([v[2] for v in VALUE.finditer(text)], ["1", "2", "3"])
        for kind in ("numeric", "date"):
            col = Column(1, kind)
            pred = next(p for p in self.catalog.predicates(col) if " BETWEEN " in p.sql)
            self.assertEqual(pred.sql, f"{col.ref} BETWEEN {col.value(1).sql} AND {col.value(2).sql}")

    def test_all_authored_forms_preserve_slots_and_tag_order(self):
        for section, entries in self.mappings.items():
            for key, forms in entries.items():
                optional = {("clauses", "limit"): {"result_unit"},
                            ("contrast_questions", "rank_direction"): {"unit"}}.get((section, key), set())
                required = set(FIELD.findall(forms[0])) - optional
                self.assertEqual(len(forms), len(set(forms)), (section, key))
                for form in forms:
                    with self.subTest(section=section, key=key, form=form):
                        fields = set(FIELD.findall(form))
                        if (section, key) == ("intent", "ranked"):
                            if "explicit_order" in fields:
                                self.assertEqual(fields, {"explicit_order", "explicit_limit"})
                            else:
                                self.assertEqual(fields - {"edge", "result_unit"}, {"n", "rank", "key", "tie"})
                        else:
                            self.assertEqual(fields - optional, required)
                        self.assertNotRegex(form.replace("<TABLE1>", ""), "[A-Z]")
                        if section == "questions":
                            if key == "all_rows":
                                self.assertIn("{*}", form)
                            if key == "aggregate":
                                self.assertIn("<TABLE1>", form)  # COUNT may have no column tags.
                            if key == "group":
                                self.assertLess(min(form.index("{key}"), form.index("{group_col}")),
                                                form.index("{agg}"))
                            ordered = {
                                "rows": ("target", "where", "order"),
                                "all_rows": ("where", "order"),
                                "aggregate": ("agg", "where"),
                                "group": ("agg", "where", "having", "order"),
                                "scalar": ("target", "measure", "population", "outer"),
                            }[key]
                            offsets = [form.index("{" + field + "}") for field in ordered]
                            self.assertEqual(offsets, sorted(offsets))
                        if section == "predicates" and key == "between":
                            self.assertLess(form.index("{a}"), form.index("{b}"))

    def test_limit_wording_for_one_and_many_results(self):
        for n, phrase in zip((1, 2), self.catalog.limits[1:3]):
            self.assertEqual(phrase.sql, f" LIMIT {n}")
            forms = list(phrase.distinct_questions({}, 100, random.Random(42)))
            self.assertEqual(len(forms), len(self.mappings["clauses"]["limit"]))
            for form in forms:
                self.assertIn(str(n), form)
                self.assertNotRegex(form, r"\b1 results\b|\b2 result\b|\{")

    def test_date_order_and_limit_range(self):
        self.assertEqual([p.sql for p in self.catalog.limits[1:]], [f" LIMIT {n}" for n in range(1, 101)])
        col = Column(1, "date")
        for direction, cue in (("ASC", r"oldest first|earliest.*first|earliest to latest|chronological order"),
                               ("DESC", r"newest first|latest.*first|most recent to oldest|newest to oldest|reverse chronological")):
            option = next(p for p in self.catalog.row_sorting((col,)) if p.sql == f" ORDER BY {col.ref} {direction}")
            cap = self.catalog.limits[7]
            ranked = option.ranked.fill(explicit_order=option, explicit_limit=cap, **dict(cap.bindings))
            for phrase in (option, ranked):
                for wording in phrase.distinct_questions({}, 10, random.Random(42)):
                    self.assertIn(col.ref, wording)
                    self.assertRegex(wording, cue)
                    if phrase is ranked:
                        self.assertIn("7", wording)

    def test_half_million_mix_from_batch_sizes(self):
        # Exercise the production scheduler without generating a training dataset.
        preview = list(self.automatic_pairs(1000, 42))
        self.assertEqual(preview, list(islice(self.automatic_pairs(500_000, 42), 1000)))
        self.assertEqual(preview, list(self.automatic_pairs(1000, 42, max_questions_per_sql=3)))
        self.assertEqual(len({r["output"] for r in preview}), 334)
        def form_count(phrase, scope):
            if isinstance(phrase, int):
                return phrase
            scope = {**scope, **dict(phrase.bindings)}
            return sum(prod(form_count(scope[k], scope) for k in dict.fromkeys(FIELD.findall(f)))
                       for f in phrase.words)

        pairs, reserved = contrast_catalog(self.catalog, 42)
        available = [sum(b.size for b in blocks) for blocks in (
            self.catalog.rows(), self.catalog.aggregates(False), self.catalog.aggregates(True), self.catalog.scalars()
        )]
        for sql in reserved:
            if "(SELECT " in sql:
                available[3] -= 1
            elif " GROUP BY " in sql:
                available[2] -= 1
            elif re.match(r"SELECT \{(COUNT|AVG|SUM|MIN|MAX)\(", sql):
                available[1] -= 1
            else:
                available[0] -= 1
        weights = Distribution.load(self.automatic_profile).weights
        ratio = Fraction(weights["ordinary"]) / Fraction(weights["contrast"])
        schedule = balanced_batches([repeat(1, n) for n in available], pairs, 3, ratio)
        for block in self.catalog.aggregates(False):
            if not block.choices:  # Even unfiltered aggregates have enough distinct forms.
                self.assertEqual(len(list(block.source(0, 64, seed=42)[1])), 64)
        names = {"rank_direction": 0, "boundary": 1, "filter_level": 2, "average_scope": 3}
        counts, matched = Counter(), Counter()
        checked = set()
        emitted = 0
        index, paired, item = next(schedule)
        while emitted < 500_000:
            budget = question_budget(emitted, paired or index == 1, None)
            size = budget
            if paired:
                left, right = item.left, dict(item.right.bindings)
                shared = {k: p if p == right[k] else 1 for k, p in left.bindings}
                capacity = min(64, form_count(Phrase(left.sql, left.words), shared))
                if (item.kind, capacity) not in checked:
                    for _, questions in item.sources(42, 64):
                        self.assertEqual(len(list(questions)), capacity)
                    checked.add((item.kind, capacity))
                size = 2 * min(budget, capacity)
            size = min(size, 500_000 - emitted)
            counts[names[item.kind] if paired else index] += size
            if paired:
                matched[item.kind] += size
            emitted += size
            if emitted < 500_000:
                index, paired, item = schedule.send(size)
        for n in counts.values():
            self.assertLess(abs(n - 125_000), 200)
        for kind in names:
            self.assertLess(abs(matched[kind] - 500_000 / 36), 200)
        self.assertEqual(question_budget(499_000, True, 3), 3)

    def test_matched_training_questions_differ_only_in_the_cue(self):
        records = list(self.automatic_pairs(1000, 42))
        previous = {}
        kinds = set()
        for r in records:
            if "contrast" in r:
                kind, peer = r["contrast"]["kind"], previous[r["contrast"]["paired_id"]]
                parent, child = r["id"].split(".")
                self.assertEqual(peer["id"], f"{int(parent)-1}.{child}")
                self.assertNotEqual(peer["output"], r["output"])
                left, right = peer["input"], r["input"]
                if kind == "rank_direction":
                    variants = {left.replace("highest", "lowest")}
                elif kind == "average_scope":
                    variants = {left.replace("all rows in the table", "only rows in that same subset")}
                elif kind == "filter_level":
                    expected = re.sub(r"rows whose <TABLE1.COL2, numeric>", "groups whose aggregate", left)
                    variants = {expected.replace("[TABLE1.COL2, numeric, #1]", "[AGG_VALUE, numeric, #1]")}
                else:
                    self.assertEqual(kind, "boundary")
                    section = "date_comparisons" if "COL1, date" in peer["output"] else "comparisons"
                    op = re.search(r"WHERE <[^>]+> ([<>]) ", peer["output"])[1]
                    variants = {
                        left.replace(strict, inclusive, 1)
                        for strict in self.mappings[section][op] if strict in left
                        for inclusive in self.mappings[section][op + "="]
                    }
                self.assertNotEqual(left, right)
                self.assertIn(right, variants)
                kinds.add(kind)
            previous[r["id"]] = r
        self.assertEqual(kinds, {"boundary", "rank_direction", "average_scope", "filter_level"})
        # An output ending halfway through the right parent only refers back to existing rows.
        self.assertEqual(list(islice(self.automatic_pairs(1000, 42), 53)), list(self.automatic_pairs(53, 42)))

    def test_contrast_answers_on_sqlite(self):
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.execute("CREATE TABLE sample (c1 INTEGER, c2 INTEGER)")
        db.executemany("INSERT INTO sample VALUES (?, ?)", [(1, 2), (2, 2), (5, 2), (12, 1), (12, 5)])
        cases = [
            (next(boundary_pairs(self.catalog)), [(3,)], [(4,)]),
            (next(rank_pairs(self.catalog)), [12], [1]),
            (next(p for p in average_scope_pairs(self.catalog)
                  if "SELECT {*} " in p.left.sql and "> (SELECT AVG" in p.left.sql
                  and "COL2, numeric> =" in p.left.sql), [], [(5, 2)]),
            (next(filter_level_pairs(self.catalog)), [(12, 5.0)], [(12, 3.0)]),
        ]
        for pair, expected_left, expected_right in cases:
            with self.subTest(kind=pair.kind):
                left, right = (db.execute(sqlite_sql(s.sql)).fetchall() for s in (pair.left, pair.right))
                if pair.kind == "rank_direction":  # Ties at the limit may return either row with c1 = 12.
                    left, right = [r[0] for r in left], [r[0] for r in right]
                self.assertEqual(left, expected_left)
                self.assertEqual(right, expected_right)

    def test_intent_frames_match_the_sql_clauses(self):
        records = list(self.automatic_pairs(1000, 42))
        seen = set()
        for r in records:
            sql, q = r["output"], r["input"]
            if "how many" in q:
                self.assertIn("{COUNT(*)}", sql)
                seen.add("count")
            count = re.search(r"\b(top|bottom) (\d+) by ", q)
            if count:
                order = sql.split(" ORDER BY ")[1]
                direction = "DESC" if count[1] == "top" else "ASC"
                self.assertEqual(re.search(r" (ASC|DESC)\b", order)[1], direction)
                self.assertTrue(sql.endswith(" LIMIT " + count[2]))
                if re.search(r"(?:ASC|DESC), ", order):
                    self.assertRegex(q, r"tie|equal values")
                    seen.add("tie")
                seen.add(count[1])
            if "its average across" in q:
                self.assertIn("(SELECT AVG(", sql)
                seen.add("average")
        self.assertEqual(seen, {"count", "top", "bottom", "tie", "average"})


if __name__ == "__main__":
    unittest.main()
