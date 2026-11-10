import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import torch

from src.back.generate import Generation
from src.back.tokenizer import BackTokenizer
from training.back.config import TrainConfig
from training.back.dataset import IGNORE, PairDataset, collate_pairs
from training.back.evaluate import main as evaluate
from training.back.evaluate import score_pairs, unmapped_tags
from training.back.train import train_epoch


class TrainingDataTests(unittest.TestCase):
    def test_evaluation_requires_explicit_dataset_and_device(self):
        for missing, provided in (("--data", ["--device", "cpu"]), ("--device", ["--data", "test.json"])):
            with self.subTest(missing=missing):
                with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as failure:
                    evaluate(["--checkpoint", "best.pt", *provided])
                self.assertEqual(failure.exception.code, 2)
                self.assertIn(f"required: {missing}", errors.getvalue())

    def test_shift_masks_question_and_padding_but_includes_first_sql_and_eos(self):
        rows = [
            {"input": "show <TABLE1>", "output": "SELECT {*} FROM <TABLE1>"},
            {"input": "how many <TABLE1> are there today?", "output": "SELECT {COUNT(*)} FROM <TABLE1>"},
        ]
        tokenizer = BackTokenizer.fit([r["input"] for r in rows], 512)
        ds = PairDataset(rows, tokenizer)
        inputs, targets = collate_pairs([ds[0], ds[1]], tokenizer.pad_id)
        for i, row in enumerate(rows):
            sep = inputs[i].tolist().index(tokenizer.sep_id)
            sql = tokenizer.encode_sql(row["output"])
            self.assertTrue(targets[i, :sep].eq(IGNORE).all())
            self.assertEqual(targets[i, sep:sep + len(sql) + 1].tolist(), [*sql, tokenizer.eos_id])
            self.assertTrue(targets[i, sep + len(sql) + 1:].eq(IGNORE).all())

    def test_validation_and_smoke_require_eos_and_only_pass_questions_to_generator(self):
        q = "the row count of <TABLE1>?"
        sql = "SELECT {COUNT(*)} FROM <TABLE1>"
        rows = [{"input": q, "output": sql}] * 3
        model = torch.nn.Identity()
        with patch("training.back.evaluate.generate_results", return_value=[
            Generation(sql, "eos"), Generation(sql, "context"), Generation("SELECT {COUNT(} FROM <TABLE1>", "eos"),
        ]) as generate:
            report, predictions = score_pairs(model, None, rows, device="cpu", batch_size=4, with_predictions=True)
        generate.assert_called_once_with(model, None, [q, q, q], device="cpu")
        self.assertEqual(report["correct"], 1)
        self.assertEqual(report["rows_with_tag_errors"], 1)
        self.assertEqual(report["without_eos"], 1)
        self.assertEqual(len(predictions), 3)

    def test_value_slot_uses_its_owner_table_column_and_type(self):
        q = "show {TABLE1.COL1, numeric} from <TABLE1> above [TABLE1.COL1, numeric, #1]"
        self.assertEqual(unmapped_tags(q, "SELECT {TABLE1.COL1, numeric} FROM <TABLE1>"), [])
        self.assertEqual(unmapped_tags(q, "[TABLE1.COL1, numeric, #2]"), ["[TABLE1.COL1, numeric, #2]"])

    def test_production_settings_and_cosine_endpoints(self):
        cfg = TrainConfig(seed=42)
        self.assertEqual(cfg.batch_size * cfg.accumulation_steps, 32)
        self.assertAlmostEqual(cfg.learning_rate(1, 112500), cfg.lr / 1125)
        self.assertAlmostEqual(cfg.learning_rate(1125, 112500), cfg.lr)
        self.assertAlmostEqual(cfg.learning_rate(112500, 112500), cfg.min_lr)

    def test_accumulation_normalizes_unequal_microbatches_by_sql_tokens(self):
        class Lookup(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2, 2))

            def forward(self, inputs, *, compact):
                return self.weight[inputs]

            def sql_targets(self, targets):
                return targets

        xs = torch.tensor([[0, 1, 0], [0, 0, 0], [1, 1, 0]])
        ys = torch.tensor([[1, 1, IGNORE], [0, IGNORE, IGNORE], [1, 0, 0]])
        config = TrainConfig(seed=42, accumulation_steps=4, warmup_ratio=0, lr=0.1, min_lr=0.1)
        accumulated, whole = Lookup(), Lookup()
        for model, batches in [
            (accumulated, [(xs[i:i + 1], ys[i:i + 1]) for i in range(3)]),
            (whole, [(xs, ys)]),
        ]:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            step, _, _ = train_epoch(model, batches, optimizer, config, "cpu", False, 0, 1)
            self.assertEqual(step, 1)
        torch.testing.assert_close(accumulated.weight, whole.weight)


if __name__ == "__main__":
    unittest.main()
