import argparse
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from src.back.generate import Generation
from src.front.tokenizer import FrontTokenizer
from training.front.dataset import IGNORE, PairDataset, collate_pairs, overfit_pairs
from training.front.evaluate import score_pairs
from training.front.train import training_split_paths

NAMES = ["customers", "orders", "membership", "quantity"]
ROWS = [
    {"id": "1.1.1.1", "input": "Show the tier of each client.",
     "output": "Show the {customers.membership, string} of each <customers>."},
    {"id": "1.1.1.2", "input": "Show the membership level of each buyer.",
     "output": "Show the {customers.membership, string} of each <customers>."},
    {"id": "2.1.1.1", "input": "How many purchases are there?", "output": "How many <orders> are there?"},
    {"id": "3.1.1.1", "input": "Show the amount ordered of each purchase.",
     "output": "Show the {orders.quantity, numeric} of each <orders>."},
]


class FrontTrainingTests(unittest.TestCase):
    def test_shift_masks_question_and_includes_tagged_sentence_and_eos(self):
        tokenizer = FrontTokenizer.fit([t for r in ROWS for t in (r["input"], r["output"])], 512, names=NAMES)
        ds = PairDataset(ROWS[:2], tokenizer)
        inputs, targets = collate_pairs([ds[0], ds[1]], tokenizer.pad_id)
        for i, row in enumerate(ROWS[:2]):
            sep = inputs[i].tolist().index(tokenizer.sep_id)
            output = tokenizer.encode_output(row["output"])
            self.assertTrue(targets[i, :sep].eq(IGNORE).all())
            self.assertEqual(targets[i, sep:sep + len(output) + 1].tolist(), [*output, tokenizer.eos_id])

    def test_overfit_sample_has_distinct_targets_across_source_sqls(self):
        selected = overfit_pairs(ROWS, 3, seed=0)
        self.assertEqual(len({r["output"] for r in selected}), 3)
        self.assertEqual({r["id"].split(".")[0] for r in selected}, {"1", "2", "3"})
        with self.assertRaisesRegex(ValueError, "need 4 distinct target sentences"):
            overfit_pairs(ROWS, 4, seed=0)

    def test_scores_separate_tags_from_sentence_frame(self):
        target = ROWS[0]["output"]
        rows = [{"input": "q", "output": target}] * 4
        with patch("training.front.evaluate.generate_results", return_value=[
            Generation(target, "eos"),
            Generation("List the {customers.membership, string} of each <customers>.", "eos"),
            Generation("Show the {orders.quantity, numeric} of each <customers>.", "eos"),
            Generation("Show the {customers.membership, string} of each <customers", "context"),
        ]):
            report, predictions = score_pairs(torch.nn.Identity(), None, rows, device="cpu", batch_size=4,
                                              with_predictions=True)
        self.assertEqual((report["correct"], report["without_eos"], report["rows_with_malformed_tags"]), (1, 1, 1))
        self.assertEqual((report["tag_em"], report["frame_em"]), (0.5, 0.5))
        self.assertEqual([p["tags_exact"] for p in predictions], [True, True, False, False])

    def test_train_file_selects_matching_front_validation_split(self):
        args = argparse.Namespace(train_file=Path("data/x/questions_front_train.json"), data_dir=None, overfit=None)
        self.assertEqual(training_split_paths(args)[1], Path("data/x/questions_front_validation.json"))
        args.train_file = Path("data/x/questions_back_train.json")
        with self.assertRaisesRegex(ValueError, "front_train.json"):
            training_split_paths(args)


if __name__ == "__main__":
    unittest.main()
