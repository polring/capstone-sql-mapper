import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from inference.mapper import main
from src.mapper.mapper import map_question


class MapperTests(unittest.TestCase):
    def test_documented_front_output_produces_the_back_question(self):
        output, mapping = map_question(
            "Show the names of items where the price is greater than 10 dollars.",
            "Show the {items.item_name, string} of <items> where the <items.price, numeric> "
            "is greater than [items.price, numeric, #1].",
        )
        self.assertEqual(output, "Show the {TABLE1.COL1, string} of <TABLE1> where the "
                                 "<TABLE1.COL2, numeric> is greater than [TABLE1.COL2, numeric, #1].")
        self.assertEqual(mapping, {
            "tables": {"TABLE1": "items"},
            "columns": {"TABLE1.COL1": "items.item_name", "TABLE1.COL2": "items.price"},
            "values": {"[TABLE1.COL2, numeric, #1]": "10 dollars"},
        })

    def test_value_first_and_repeated_column_reuse_number(self):
        output, mapping = map_question(
            "Between $10.00 and 20 dollars, list the prices of products by price.",
            "Between [items.price, numeric, #1] and [items.price, numeric, #2], "
            "list the {items.price, numeric} of <items> by <items.price, numeric>.",
        )
        self.assertEqual(output, "Between [TABLE1.COL1, numeric, #1] and [TABLE1.COL1, numeric, #2], "
                                 "list the {TABLE1.COL1, numeric} of <TABLE1> by <TABLE1.COL1, numeric>.")
        self.assertEqual(mapping["values"], {"[TABLE1.COL1, numeric, #1]": "$10.00",
                                             "[TABLE1.COL1, numeric, #2]": "20 dollars"})

    def test_wildcard_and_aggregate_comparison(self):
        self.assertEqual(map_question("Show all orders.", "Show {*} <orders>.")[0],
                         "Show {*} <TABLE1>.")
        output, mapping = map_question(
            "Count the orders in each status with at least 10.",
            "Count the <orders> in each <orders.status, string> with at least [AGG_VALUE, numeric, #1].",
        )
        self.assertTrue(output.endswith("[AGG_VALUE, numeric, #1]."))
        self.assertEqual(mapping["values"], {"[AGG_VALUE, numeric, #1]": "10"})

    def test_case_article_and_exact_value_text(self):
        _, mapping = map_question(
            "show an item named O’Brien  Jr. after March 5, 2025.",
            "Show a <items> named [items.item_name, string, #1] after [orders.order_date, date, #1].",
        )
        self.assertEqual(mapping["values"], {
            "[TABLE1.COL1, string, #1]": "O’Brien  Jr.",
            "[TABLE2.COL1, date, #1]": "March 5, 2025",
        })

    def test_invalid_or_ambiguous_front_output(self):
        examples = [
            ("Show the items.", "Find <items>.", "cannot be aligned"),
            ("Show New York items.", "Show [customers.city, string, #1] <items>.", "ambiguous"),
            ("Show 10.", "Show [items.price, numeric, #4].", "malformed"),
            ("Show 10.", "Show [items.price, numeric, #2].", "out-of-order"),
            ("Show price above 10.", "Show {items.price, string} above [items.price, numeric, #1].",
             "conflicting types"),
        ]
        for question, tagged, reason in examples:
            with self.subTest(tagged=tagged), self.assertRaisesRegex(ValueError, reason):
                map_question(question, tagged)

    def test_cli_output_is_readable_by_the_back_model_file_interface(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as folder:
            source = Path(folder) / "front.json"
            destination = Path(folder) / "back.json"
            source.write_text(json.dumps([{"id": "1.1", "input": "Show items.", "output": "Show <items>."}]),
                              encoding="utf-8")
            main(["--input", str(source), "--output", str(destination)])
            record, = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(record["input"], "Show <TABLE1>.")
            self.assertEqual(record["mapping"]["tables"], {"TABLE1": "items"})
            self.assertEqual(record["id"], "1.1")


if __name__ == "__main__":
    unittest.main()
