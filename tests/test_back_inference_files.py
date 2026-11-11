import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference.back import main
from src.back.generate import Generation


class InferenceFileTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(folder.cleanup)
        self.model = Path(folder.name) / "model.gguf"
        self.input = Path(folder.name) / "questions.json"
        self.output = Path(folder.name) / "results.json"
        self.model.write_bytes(b"GGUF fixture")
        loader = patch("inference.back.load_gguf", return_value=(object(), object()))
        loader.start()
        self.addCleanup(loader.stop)

    def infer(self, *options):
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            main(["--model", str(self.model), "--device", "cpu", *map(str, options)])

    def test_requires_device_for_single_question_and_file_input(self):
        for inputs in (["--question", "show <TABLE1>"], ["--input", str(self.input), "--output", str(self.output)]):
            with self.subTest(inputs=inputs):
                with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as failure:
                    main(["--model", str(self.model), *inputs])
                self.assertEqual(failure.exception.code, 2)
                self.assertIn("required: --device", errors.getvalue())

    def test_batches_write_only_question_and_generated_sql_in_order(self):
        rows = [
            {"id": "1.1", "input": "retrieve {*} from <TABLE1>.", "output": "reference SQL"},
            {"input": "show {TABLE1.COL1, date}", "metadata": {"label": "날짜"}},
            {"input": "count <TABLE1>"},
        ]
        self.input.write_text(json.dumps(rows), encoding="utf-8")
        results = [
            Generation("first SQL", "eos"),
            Generation("partial SQL", "context"),
            Generation("last SQL", "eos"),
        ]
        expected = {row["input"]: result for row, result in zip(rows, results)}
        batches = []

        def generate(model, tokenizer, questions, **kwargs):
            batches.append(list(questions))
            return [expected[question] for question in questions]

        with patch("inference.back.generate_results", side_effect=generate):
            self.infer("--input", self.input, "--output", self.output, "--batch-size", 2)

        self.assertEqual(batches, [[rows[0]["input"], rows[1]["input"]], [rows[2]["input"]]])
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), [
            {"input": row["input"], "output": result.sql} for row, result in zip(rows, results)
        ])

    def test_empty_array_produces_empty_result(self):
        self.input.write_text("[]", encoding="utf-8")
        self.infer("--input", self.input, "--output", self.output)
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), [])

    def test_later_batch_failure_preserves_previous_result(self):
        self.input.write_text('[{"input":"first"},{"input":"second"}]', encoding="utf-8")
        self.output.write_bytes(b"[]\n")
        failed_batch = ValueError("could not encode second question")
        with patch("inference.back.generate_results", side_effect=[[Generation("first", "eos")], failed_batch]):
            with self.assertRaises(SystemExit) as failure:
                self.infer("--input", self.input, "--output", self.output, "--batch-size", 1)
        self.assertNotEqual(failure.exception.code, 0)
        self.assertEqual(self.output.read_bytes(), b"[]\n")

    def test_rejects_incompatible_options_and_protects_source_files(self):
        self.input.write_bytes(b"[]\n")
        options = [
            ("--input", self.input),
            ("--question", "show <TABLE1>", "--output", self.output),
            ("--question", "show <TABLE1>", "--input", self.input, "--output", self.output),
            ("--input", self.input, "--output", self.output, "--batch-size", 0),
            ("--input", self.input, "--output", self.input),
            ("--input", self.input, "--output", self.model),
        ]
        for args in options:
            with self.subTest(args=args):
                with self.assertRaises(SystemExit) as failure:
                    self.infer(*args)
                self.assertNotEqual(failure.exception.code, 0)
        self.assertEqual(self.input.read_bytes(), b"[]\n")
        self.assertEqual(self.model.read_bytes(), b"GGUF fixture")
        self.assertFalse(self.output.exists())

    def test_rejects_rows_without_a_nonempty_input(self):
        for records in ({"input": "not an array"}, ["just a string"], [{}], [{"input": "  "}], [{"input": 3}]):
            self.input.write_text(json.dumps(records), encoding="utf-8")
            with self.subTest(records=records):
                with self.assertRaises(SystemExit) as failure:
                    self.infer("--input", self.input, "--output", self.output)
                self.assertNotEqual(failure.exception.code, 0)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
