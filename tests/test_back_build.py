import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from src.data_generation.build_back import STAGING_DIR, build, read_back_splits, split_pairs, sql_family


def record(root, child, sql):
    return {"id": f"{root}.{child}", "input": f"question {root}.{child}", "output": sql}


class BackBuildTest(unittest.TestCase):
    def test_cli_requires_source_and_seed_and_uses_source_directory(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp) / "data directory"
            folder.mkdir()
            source = folder / "v1_dataset.json"
            source.write_text(json.dumps([record(i, 1, f"SELECT {i}") for i in (1, 2)]), encoding="utf-8")
            options = {"--input": str(source), "--seed": "0"}
            command = [sys.executable, "-m", "src.data_generation.build_back"]
            for missing in options:
                with self.subTest(missing=missing):
                    args = [part for flag, value in options.items() if flag != missing for part in (flag, value)]
                    result = subprocess.run([*command, *args], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(f"the following arguments are required: {missing}", result.stderr)
            args = [*command, "--input", str(source), "--seed", "0"]
            result = subprocess.run([*args, "--data-dir", tmp], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("unrecognized arguments: --data-dir", result.stderr)
            self.assertEqual(list(folder.iterdir()), [source])
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            for name in ("v1_dataset_back_train.json", "v1_dataset_back_validation.json"):
                self.assertTrue((folder / name).is_file())
                self.assertIn(name, result.stdout)

    def test_stratified_split_is_grouped_and_order_independent(self):
        patterns = [
            "SELECT {{*}} FROM <TABLE1> LIMIT {}",
            "SELECT {{SUM(<TABLE1.COL1, numeric>)}} FROM <TABLE1> LIMIT {}",
            "SELECT {{COUNT(*)}} FROM <TABLE1> GROUP BY <TABLE1.COL1, date> LIMIT {}",
            "SELECT {{*}} FROM <TABLE1> WHERE <TABLE1.COL1, numeric> "
            "> (SELECT AVG(<TABLE1.COL1, numeric>) FROM <TABLE1>) LIMIT {}",
        ]
        records = []
        for family, pattern in enumerate(patterns):
            for idx in range(20):
                sql = pattern.format(idx + 1)
                records.extend(record(20 * family + idx + 1, child, sql) for child in (1, 2))
        train, validation = split_pairs(records, seed=42)
        self.assertEqual(len(train), 152)
        self.assertEqual(len(validation), 8)
        self.assertEqual({sql_family(r["output"]) for r in validation}, {"rows", "aggregate", "group", "scalar"})
        self.assertEqual({r["output"] for r in train} & {r["output"] for r in validation}, set())
        self.assertEqual({r["id"].split(".")[0] for r in train}
                         & {r["id"].split(".")[0] for r in validation}, set())
        _, reverse_validation = split_pairs(list(reversed(records)), seed=42)
        self.assertEqual(reverse_validation, list(reversed(validation)))

    def test_identical_sql_across_parent_ids_stays_together(self):
        records = [record(i, 1, f"SELECT {{*}} FROM <TABLE1> LIMIT {i}") for i in range(1, 9)]
        records.append(record(10, 1, "select   {*}  FROM <TABLE1> LIMIT 1"))
        for seed in range(8):
            _, validation = split_pairs(records, seed=seed, validation_ratio=0.5)
            self.assertEqual(records[0] in validation, records[-1] in validation)

    def test_build_prefixes_both_splits_and_preserves_other_datasets(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            version = Path(tmp)
            rows = [record(i, 1, f"SELECT {{*}} FROM <TABLE1> LIMIT {i}") for i in range(1, 21)]
            source = version / "v1.dataset.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            test_bytes = b'[{"independent": true}]'
            unrelated = ("back_test.json", "e2e_test.json", "back_train.json", "back_validation.json")
            for name in unrelated:
                (version / name).write_bytes(test_bytes)
            self.assertEqual(build(source, seed=42), (19, 1))
            outputs = [version / f"v1.dataset_back_{part}.json" for part in ("train", "validation")]
            snapshots = [path.read_bytes() for path in outputs]
            self.assertEqual(read_back_splits(*outputs), tuple(json.loads(b) for b in snapshots))
            written = [r for snapshot in snapshots for r in json.loads(snapshot)]
            self.assertCountEqual([r["id"] for r in written], [r["id"] for r in rows])
            other = version / "other.json"
            other.write_text(json.dumps(rows[:4]), encoding="utf-8")
            self.assertEqual(build(other, seed=0), (3, 1))
            self.assertEqual([len(part) for part in read_back_splits(
                version / "other_back_train.json", version / "other_back_validation.json")], [3, 1])
            self.assertEqual(snapshots, [path.read_bytes() for path in outputs])
            self.assertEqual(build(source, seed=42), (19, 1))
            self.assertEqual(snapshots, [path.read_bytes() for path in outputs])
            for name in unrelated:
                self.assertEqual((version / name).read_bytes(), test_bytes)
            source.write_text(json.dumps(rows + rows[:1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                build(source, seed=42)
            self.assertEqual(snapshots, [path.read_bytes() for path in outputs])
            (version / STAGING_DIR).mkdir()
            with self.assertRaisesRegex(ValueError, "publication"):
                read_back_splits(*outputs)

    def test_failed_validation_publish_restores_previous_train(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            version = Path(tmp)
            rows = [record(i, 1, f"SELECT {{*}} FROM <TABLE1> LIMIT {i}") for i in (1, 2)]
            source = version / "v1_dataset.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            train = version / "v1_dataset_back_train.json"
            train.write_text("old training set", encoding="utf-8")
            (version / "v1_dataset_back_validation.json").mkdir()
            with self.assertRaises(OSError):
                build(source, seed=42)
            self.assertEqual(train.read_text(encoding="utf-8"), "old training set")
            self.assertFalse((version / ".back-build-in-progress").exists())

    def test_rejects_conflicting_parent_or_question(self):
        rows = [record(1, 1, "SELECT {*} FROM <TABLE1>"), record(1, 2, "SELECT {COUNT(*)} FROM <TABLE1>")]
        with self.assertRaisesRegex(ValueError, "source 1"):
            split_pairs(rows, seed=42)
        rows[1].update(id="2.1", input=rows[0]["input"])
        with self.assertRaisesRegex(ValueError, "same input"):
            split_pairs(rows, seed=42)


if __name__ == "__main__":
    unittest.main()
