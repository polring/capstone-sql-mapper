import unittest

from src.back.config import ModelConfig
from src.back.tokenizer import BackTokenizer


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tok = BackTokenizer.fit(["date of {TABLE1.COL1, date}"] * 3, vocab_size=512)

    def test_shared_atoms_preserve_case_and_sql_round_trip(self):
        tag = "{TABLE1.COL1, date}"
        self.assertEqual(self.tok.encode_question(tag), self.tok.encode_sql(tag))
        self.assertNotEqual(self.tok.encode_question("Table"), self.tok.encode_question("table"))
        sql = ("SELECT {*} FROM <TABLE1> WHERE <TABLE1.COL2, numeric> >= "
               "[TABLE1.COL2, numeric, #1] LIMIT 27")
        ids = self.tok.encode_sql(sql)
        self.assertEqual(self.tok.decode_sql(ids), sql)
        self.assertEqual(ids[-2:], [ord("2"), ord("7")])
        self.assertEqual(len(self.tok.encode_sql(">=")), 1)
        with self.assertRaisesRegex(ValueError, "unsupported SQL token"):
            self.tok.encode_sql("SELECT made_up_identifier")

    def test_reload_and_whitelist(self):
        restored = BackTokenizer.from_dict(self.tok.to_dict())
        self.assertEqual(self.tok.encode_question("unseen DATE and <TABLE1>"),
                         restored.encode_question("unseen DATE and <TABLE1>"))
        self.assertEqual(tuple(restored.sql_ids), tuple(self.tok.sql_ids))
        self.assertNotIn(ord("x"), self.tok.sql_ids)
        self.assertIn(self.tok.eos_id, self.tok.sql_ids)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        import torch
        from src.back.model import BackModel
        self.tok = BackTokenizer.fit(["show names", "count all"], vocab_size=512)
        self.cfg = ModelConfig(vocab_size=512, d_model=16, num_layers=2, num_heads=2,
                               ffn_dim=48, context_length=64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            self.model = BackModel(self.cfg, self.tok.sql_ids)
        self.model.eval()

    def test_sql_projection_and_causal_mask(self):
        import torch
        ids = torch.tensor([[self.tok.bos_id, self.tok.sep_id, self.tok.eos_id]])
        with torch.no_grad():
            compact = self.model(ids, compact=True)
            dense = self.model(ids)
            prefix = self.model(ids[:, :2], compact=True)
        torch.testing.assert_close(compact, dense[:, :, self.model.allowed_ids])
        self.assertTrue(dense[:, :, ord("x")].isneginf().all())
        torch.testing.assert_close(prefix, compact[:, :2])
        mapped = self.model.sql_targets(torch.tensor([[-100, self.tok.eos_id]]))
        self.assertEqual(mapped.tolist(), [[-100, 0]])

    def test_cached_decode_ignores_padding_and_matches_full_recomputation(self):
        import torch
        from src.back.generate import generate_results
        questions = ["show <TABLE1>", "how many rows are present?"]
        result = generate_results(self.model, self.tok, questions, max_new_tokens=8)
        for q, actual in zip(questions, result):
            ids = [self.tok.bos_id, *self.tok.encode_question(q), self.tok.sep_id]
            answer = []
            with torch.no_grad():
                for _ in range(8):
                    logits = self.model(torch.tensor([ids]), compact=True)[0, -1]
                    chosen = self.tok.sql_ids[int(logits.argmax())]
                    if chosen == self.tok.eos_id:
                        break
                    ids.append(chosen)
                    answer.append(chosen)
            self.assertEqual(actual.sql, self.tok.decode_sql(answer))
            self.assertEqual(actual.stop_reason, "max_new_tokens" if len(answer) == 8 else "eos")

    def test_eos_is_preferred_on_exact_tie_and_limits_report_truncation(self):
        import torch
        from src.back.generate import generate_results
        with torch.no_grad():
            self.model.lm_head.weight.zero_()
        cases = [("hi", None, "eos"), ("hi", 0, "max_new_tokens"),
                 (" " * 62, None, "context")]
        for q, limit, reason in cases:
            result, = generate_results(self.model, self.tok, [q], max_new_tokens=limit)
            self.assertEqual((result.sql, result.stop_reason), ("", reason))
