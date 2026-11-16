"""팀원 4 · natural_questions/generate.py 검증

    python -m unittest discover tests
"""

import importlib.util
import json
import random
import re
import sqlite3
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATE_PATH = ROOT / "src" / "data_generation" / "natural_questions" / "generate.py"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "schema_mapping_sample.json"
DB_PATH = ROOT / "data" / "v1" / "shop.db"

spec = importlib.util.spec_from_file_location("natural_questions_generate", GENERATE_PATH)
nq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nq)

MAPPINGS = json.loads(nq.MAPPINGS_PATH.read_text(encoding="utf-8"))
RECORDS = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class ParseTagsTest(unittest.TestCase):
    def test_kinds_in_order(self):
        tags = nq.parse_tags(RECORDS[0]["input"])
        self.assertEqual([t.kind for t in tags], ["select", "table", "column", "value"])
        self.assertEqual([t.key for t in tags], ["items.item_name", "items", "items.price", "items.price"])

    def test_star_and_agg_value(self):
        self.assertEqual(nq.parse_tags("Show {*} of <items>.")[0].kind, "star")
        self.assertEqual(nq.parse_tags("at least [AGG_VALUE, numeric, #1].")[0].key, "AGG_VALUE")

    def test_unknown_tag_rejected(self):
        with self.assertRaises(ValueError):
            nq.parse_tags("Show the {items.item_name} of <items>.")  # 타입 누락


class GrammarTest(unittest.TestCase):
    def realize(self, sentence, seed=0):
        return nq.realize(sentence, nq.parse_tags(sentence), MAPPINGS, random.Random(seed))[0]

    def test_each_makes_singular(self):
        for seed in range(10):
            q = self.realize(RECORDS[2]["input"], seed)
            self.assertRegex(q, r"^List the (name|item name|product name) of each (item|product) with ")

    def test_plural_select_of_table(self):
        for seed in range(10):
            q = self.realize(RECORDS[0]["input"], seed)
            self.assertRegex(q, r"^Show the (names|item names|product names) of (items|products) where the "
                                r"(price|cost|unit price) is greater than .+\.$")

    def test_sentence_start_capitalized(self):
        self.assertRegex(self.realize("<items> with the lowest <items.price, numeric>."), r"^[A-Z]")

    def test_article_agreement(self):
        for seed in range(20):
            q = self.realize("Show {*} of a <items>.", seed)
            self.assertNotRegex(q, r"\ba item\b|\ban product\b")


class LikeModeTest(unittest.TestCase):
    # 팀원 2 generalized_pairs/mappings.json의 predicates 문구 일부
    LIKE_FORMS = {
        "starts": ["{col} begins with the text {a}", "the first part of {col} matches {a}"],
        "ends": ["{col} has {a} at the end", "{a} is the suffix of {col}", "{col} ends with the text {a}"],
        "contains": ["{a} is part of the text in {col}", "{col} has {a} somewhere in it"],
    }
    COL = "<items.item_name, string>"
    VALUE = "[items.item_name, string, #1]"

    def mode(self, predicate, like_forms=LIKE_FORMS):
        sentence = f"Show the {{items.category, string}} of <items> where {predicate}."
        tag = next(t for t in nq.parse_tags(sentence) if t.kind == "value")
        return nq.like_mode(sentence, tag, like_forms)

    def test_team2_forms(self):
        cases = {
            f"{self.COL} begins with the text {self.VALUE}": "starts",
            f"the first part of {self.COL} matches {self.VALUE}": "starts",
            f"{self.COL} has {self.VALUE} at the end": "ends",
            f"{self.VALUE} is the suffix of {self.COL}": "ends",
            f"{self.COL} ends with the text {self.VALUE}": "ends",
            f"{self.VALUE} is part of the text in {self.COL}": "contains",
            f"{self.COL} has {self.VALUE} somewhere in it": "contains",
        }
        for predicate, expected in cases.items():
            self.assertEqual(self.mode(predicate), expected, predicate)

    def test_fallback_without_team2_forms(self):
        self.assertEqual(self.mode(f"the {self.COL} ends with {self.VALUE}", {}), "ends")
        self.assertIsNone(self.mode(f"{self.COL} has {self.VALUE} at the end", {}))

    def test_plain_condition_is_not_like(self):
        self.assertIsNone(self.mode(f"the {self.COL} is {self.VALUE}"))

    def test_other_column_does_not_match(self):
        # 문구의 {col}은 값 태그와 같은 컬럼이어야 한다.
        self.assertIsNone(self.mode(f"<items.category, string> has {self.VALUE} at the end"))

    def test_realize_uses_fragment(self):
        sentence = f"Show the {{items.category, string}} of <items> where {self.COL} has {self.VALUE} at the end."
        fragments = MAPPINGS["values"]["items.item_name"]["like_fragments"]["ends"]
        for seed in range(10):
            _, values = nq.realize(sentence, nq.parse_tags(sentence), MAPPINGS, random.Random(seed), self.LIKE_FORMS)
            self.assertIn(values[self.VALUE]["db_value"], fragments)


class GenerateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = nq.generate(RECORDS, MAPPINGS)
        cls.parents = {r["id"]: r for r in RECORDS}

    def test_output_is_parent_tag_sentence(self):
        for ex in self.examples:
            parent_id = ex["id"].rsplit(".", 1)[0]
            self.assertEqual(ex["output"], self.parents[parent_id]["input"])

    def test_no_tag_chars_in_question(self):
        for ex in self.examples:
            self.assertNotRegex(ex["input"], r"[{}\[\]<>]", ex["id"])

    def test_ids_and_count_per_parent(self):
        limit = MAPPINGS["generation"]["max_per_parent"]
        for parent_id in self.parents:
            ids = [ex["id"] for ex in self.examples if ex["id"].rsplit(".", 1)[0] == parent_id]
            self.assertTrue(1 <= len(ids) <= limit, parent_id)
            self.assertEqual(ids, [f"{parent_id}.{i}" for i in range(1, len(ids) + 1)])

    def test_questions_unique_per_parent(self):
        for parent_id in self.parents:
            qs = [ex["input"] for ex in self.examples if ex["id"].rsplit(".", 1)[0] == parent_id]
            self.assertEqual(len(qs), len(set(qs)), parent_id)

    def test_values_cover_value_tags(self):
        for ex in self.examples:
            marks = {t.mark for t in nq.parse_tags(ex["output"]) if t.kind == "value"}
            self.assertEqual(set(ex["values"]), marks, ex["id"])
            for v in ex["values"].values():
                self.assertIn(v["text"], ex["input"], ex["id"])

    def test_same_column_values_distinct(self):
        for ex in self.examples:
            by_column = {}
            for mark, v in ex["values"].items():
                by_column.setdefault(mark.rsplit(",", 1)[0], []).append(v["db_value"])
            for column, vals in by_column.items():
                self.assertEqual(len(vals), len(set(vals)), f"{ex['id']} {column}")

    def test_deterministic(self):
        self.assertEqual(nq.generate(RECORDS, MAPPINGS), self.examples)

    def test_like_direction(self):
        ex = next(ex for ex in self.examples if ex["id"].startswith("4.1.1."))
        fragment = next(iter(ex["values"].values()))["db_value"]
        self.assertIn(fragment, MAPPINGS["values"]["items.item_name"]["like_fragments"]["ends"])


@unittest.skipUnless(DB_PATH.exists(), "data/v1/shop.db 없음")
class ShopDbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = sqlite3.connect(DB_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_mapping_keys_match_schema(self):
        columns = {f"{t}.{row[1]}" for t in MAPPINGS["tables"] for row in self.db.execute(f"PRAGMA table_info({t})")}
        self.assertEqual(columns, set(MAPPINGS["columns"]))
        self.assertEqual(columns, set(MAPPINGS["values"]))

    def test_string_candidates_exist(self):
        for key, spec in MAPPINGS["values"].items():
            if spec["type"] != "string":
                continue
            table, column = key.split(".")
            for value in spec["candidates"]:
                row = self.db.execute(f"SELECT 1 FROM {table} WHERE {column} = ?", (value,)).fetchone()
                self.assertIsNotNone(row, f"{key} = {value}")

    def test_like_fragments_match_rows(self):
        patterns = {"starts": "{}%", "ends": "%{}", "contains": "%{}%"}
        for key, spec in MAPPINGS["values"].items():
            table, column = key.split(".")
            for mode, fragments in spec.get("like_fragments", {}).items():
                for f in fragments:
                    row = self.db.execute(f"SELECT 1 FROM {table} WHERE {column} LIKE ?",
                                          (patterns[mode].format(f),)).fetchone()
                    self.assertIsNotNone(row, f"{key} {mode} {f}")


if __name__ == "__main__":
    unittest.main()
