from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.front.tokenizer import FrontTokenizer, schema_names

NAMES = ["customers", "orders", "customer_id", "membership", "quantity", "order_date"]
QUESTION = "Show the membership level of each buyer."
OUTPUT = "Show the {customers.membership, string} of each <customers>."


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tok = FrontTokenizer.fit([QUESTION, OUTPUT] * 3, vocab_size=512, names=NAMES)

    def test_tagged_output_round_trip_and_shared_words(self):
        output = ("List the {orders.quantity, numeric} of the top 10 <orders> whose "
                  "<orders.order_date, date> is later than [orders.order_date, date, #1] "
                  "and [AGG_VALUE, numeric, #2], or {*}?")
        ids = self.tok.encode_output(output)
        self.assertEqual(self.tok.decode_output(ids), output)
        self.assertEqual(self.tok.encode_output("Show the")[:1], self.tok.encode_question("Show")[:1])
        # Inside a tag a schema name is one seed, unlike the same word in the sentence.
        tag = self.tok.encode_output("<customers>")
        self.assertEqual(tag, [ord("<"), self.tok.seed_base + 7, ord(">")])
        self.assertNotEqual(tag[1:2], self.tok.encode_output("customers"))
        self.assertNotEqual(self.tok.encode_question("Show"), self.tok.encode_question("show"))

    def test_rejects_unknown_names_and_stray_tag_characters(self):
        with self.assertRaisesRegex(ValueError, "unknown schema name 'items'"):
            self.tok.encode_output("Show the <items>.")
        for output in ("Show the <customers.", "Show $3.", "Show café."):
            with self.assertRaisesRegex(ValueError, "unsupported output character"):
                self.tok.encode_output(output)
        # Questions fall back to UTF-8 bytes for anything outside ASCII words.
        ids = self.tok.encode_question("café $3?")
        self.assertEqual(ids[-6:], list("é $3?".encode("utf-8")))

    def test_reload_and_output_whitelist(self):
        restored = FrontTokenizer.from_dict(self.tok.to_dict())
        self.assertEqual(restored.encode_question(QUESTION), self.tok.encode_question(QUESTION))
        self.assertEqual(restored.encode_output(OUTPUT), self.tok.encode_output(OUTPUT))
        self.assertEqual(tuple(restored.sql_ids), tuple(self.tok.output_ids))
        self.assertEqual(self.tok.output_ids[0], self.tok.eos_id)
        self.assertIn(self.tok.merge_base, self.tok.output_ids)
        for token in (ord("$"), 0xC3, self.tok.pad_id, self.tok.bos_id, self.tok.sep_id):
            self.assertNotIn(token, self.tok.output_ids)
        with self.assertRaisesRegex(ValueError, "invalid output token ID"):
            self.tok.decode_output([self.tok.sep_id])
        changed = self.tok.to_dict() | {"output_ids": list(self.tok.output_ids[:-1])}
        with self.assertRaisesRegex(ValueError, "output whitelist"):
            FrontTokenizer.from_dict(changed)

    def test_schema_names_from_database(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "shop.db"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT)")
                db.execute("CREATE TABLE orders (order_id INTEGER, customer_id INTEGER, date TEXT)")
            db.close()
            names = schema_names(path)
        self.assertEqual(names, ["customers", "orders", "customer_id", "name", "order_id", "date"])
        tok = FrontTokenizer(512, names, [])
        self.assertEqual(tok.names, ("customers", "orders", "customer_id", "name", "order_id"))
        self.assertEqual(tok.decode_output(tok.encode_output("<orders.date, date>")), "<orders.date, date>")
        with self.assertRaisesRegex(ValueError, "database not found"):
            schema_names(Path(folder) / "missing.db")

    def test_fit_reads_the_fixed_target_database_by_default(self):
        tok = FrontTokenizer.fit([QUESTION, OUTPUT], vocab_size=512)
        self.assertEqual(tok.names, tuple(schema_names()))
        self.assertIn("order_date", tok.names)
        self.assertEqual(tok.decode_output(tok.encode_output(OUTPUT)), OUTPUT)


if __name__ == "__main__":
    unittest.main()
