import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.data_generation import build_back
from src.data_generation.build_front import build, drop_conflicts, split_paths, split_questions


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
            counts, _ = build(source, seed=0, test_ratio=0.1)
            self.assertTrue(paths["test"].is_file())
            for name, path in paths.items():
                self.assertEqual(len(json.loads(path.read_text(encoding="utf-8"))), counts[name])
            build(source, seed=0)
            self.assertFalse(paths["test"].exists())

    def test_build_drops_conflicting_questions_and_continues(self):
        # 100건 프로필에서 실제로 나온 충돌: 같은 질문이 상품명과 고객명 두 정답으로 생성됐다.
        clash = "show me names for the records."
        rows = [*self.records,
                {**record(1, 4), "input": clash, "output": "show me {items.item_name, string} for <items>."},
                {**record(2, 4), "input": clash, "output": "show me {customers.name, string} for <customers>."}]
        kept, conflicts = drop_conflicts(rows)
        self.assertEqual((kept, conflicts), (self.records, {clash: 2}))
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            source = Path(tmp) / "natural_questions.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            counts, conflicts = build(source, seed=0)
            self.assertEqual((sum(counts.values()), conflicts), (len(self.records), {clash: 2}))
            result = subprocess.run([sys.executable, "-m", "src.data_generation.build_front", "--input", str(source),
                                     "--seed", "0"], check=True, capture_output=True, text=True)
            self.assertIn("dropped 2 rows of 1 questions", result.stderr)
            self.assertIn(clash, result.stderr)

    def test_shares_validation_sqls_with_back_split(self):
        back_rows = [{"id": f"{root}.1", "input": f"q {root}", "output": f"SELECT {{T{root}.C1}} FROM <T{root}>"}
                     for root in range(1, 21)]
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            back_source = Path(tmp) / "pairs.json"
            back_source.write_text(json.dumps(back_rows), encoding="utf-8")
            build_back.build(back_source, seed=3, validation_ratio=0.1)
            back_validation = json.loads(build_back.split_paths(back_source)[1].read_text(encoding="utf-8"))
            source = Path(tmp) / "natural_questions.json"
            source.write_text(json.dumps(self.records), encoding="utf-8")
            build(source, seed=99, back_input=back_source)
            validation = json.loads(split_paths(source)["validation"].read_text(encoding="utf-8"))
            self.assertEqual(roots(validation), roots(back_validation))
            with self.assertRaisesRegex(ValueError, "test split"):
                build(source, seed=0, test_ratio=0.1, back_input=back_source)
            source.write_text(json.dumps([*self.records, record(21)]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source SQL 21 is missing"):
                build(source, seed=0, back_input=back_source)

    def test_failed_publish_restores_every_previous_split(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            source = Path(tmp) / "natural_questions.json"
            source.write_text(json.dumps(self.records), encoding="utf-8")
            paths = split_paths(source)
            build(source, seed=0, test_ratio=0.1)
            before = {name: path.read_bytes() for name, path in paths.items()}
            original = Path.replace
            failed = []

            def fail_on_validation(path, target):
                if Path(target) == paths["validation"] and not failed:
                    failed.append(path)
                    raise OSError("disk full")
                return original(path, target)

            with patch.object(Path, "replace", fail_on_validation), self.assertRaisesRegex(OSError, "disk full"):
                build(source, seed=1)
            self.assertEqual(len(failed), 1)
            self.assertEqual({name: path.read_bytes() for name, path in paths.items()}, before)
            self.assertFalse(any(Path(tmp).glob(".front-build-*")))

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
