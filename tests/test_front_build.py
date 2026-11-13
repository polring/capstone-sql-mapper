import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from src.data_generation.build_front import build, split_paths, split_questions


def record(root, child=1):
    return {"id": f"{root}.1.1.{child}", "input": f"question {root} {child}",
            "output": f"tag sentence {root}", "values": {}}


def roots(rows):
    return {row["id"].split(".")[0] for row in rows}


class FrontBuildTest(unittest.TestCase):
    records = [record(root, child) for root in range(1, 21) for child in (1, 2, 3)]

    def test_split_groups_by_source_sql(self):
        splits = split_questions(self.records, seed=42, validation_ratio=0.1, test_ratio=0.1)
        self.assertEqual([len(roots(splits[name])) for name in ("train", "validation", "test")], [16, 2, 2])
        self.assertEqual(sum(len(rows) for rows in splits.values()), len(self.records))
        self.assertFalse(roots(splits["train"]) & roots(splits["validation"]))
        self.assertFalse(roots(splits["train"]) & roots(splits["test"]))
        self.assertFalse(roots(splits["validation"]) & roots(splits["test"]))

    def test_order_independent_and_seeded(self):
        forward = split_questions(self.records, seed=7)
        backward = split_questions(list(reversed(self.records)), seed=7)
        self.assertEqual(roots(forward["validation"]), roots(backward["validation"]))
        self.assertEqual(forward["test"], [])

    def test_rejects_bad_input(self):
        cases = {
            "invalid or duplicate": [record(1), record(1)],
            "conflicting outputs": [record(1), {**record(2), "input": "question 1 1"}],
            "at least 2": [record(1), record(1, 2)],
        }
        for message, rows in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                split_questions(rows, seed=0)
        with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
            split_questions([{**record(1), "id": "1.1.1"}, record(2)], seed=0)
        with self.assertRaisesRegex(ValueError, "ratios"):
            split_questions(self.records, seed=0, validation_ratio=0.5, test_ratio=0.5)

    def test_build_writes_beside_source_and_removes_stale_test(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            source = Path(tmp) / "natural_questions.json"
            source.write_text(json.dumps(self.records), encoding="utf-8")
            paths = split_paths(source)
            counts = build(source, seed=0, test_ratio=0.1)
            self.assertTrue(paths["test"].is_file())
            for name, path in paths.items():
                self.assertEqual(len(json.loads(path.read_text(encoding="utf-8"))), counts[name])
            build(source, seed=0)
            self.assertFalse(paths["test"].exists())

    def test_cli(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            source = Path(tmp) / "natural_questions.json"
            source.write_text(json.dumps(self.records), encoding="utf-8")
            command = [sys.executable, "-m", "src.data_generation.build_front", "--input", str(source)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--seed", result.stderr)
            result = subprocess.run([*command, "--seed", "0"], check=True, capture_output=True, text=True)
            self.assertIn("natural_questions_front_train.json", result.stdout)
            self.assertIn("natural_questions_front_validation.json", result.stdout)


if __name__ == "__main__":
    unittest.main()
