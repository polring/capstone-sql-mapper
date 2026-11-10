import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import torch

from src.back.config import ModelConfig
from src.back.generate import generate_results
from src.back.gguf import load_gguf
from src.back.model import BackModel
from src.back.tokenizer import BackTokenizer
from training.back.export_gguf import write_gguf


class GgufInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def setUp(self):
        root = Path(__file__).resolve().parent
        folder = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "model.gguf"
        self.question = "show {TABLE1.COL1, date} in <TABLE1>"
        self.tokenizer = BackTokenizer.fit([self.question] * 3, vocab_size=512)
        config = ModelConfig(vocab_size=512, d_model=16, num_layers=1, num_heads=2,
                             ffn_dim=32, context_length=64, eps=2**-18, theta=4096.0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            self.model = BackModel(config, self.tokenizer.sql_ids).eval()

    @torch.inference_mode()
    def test_load_reproduces_tokenization_weights_and_greedy_decoding(self):
        write_gguf(self.model, self.tokenizer, self.path)
        model, tokenizer = load_gguf(self.path)
        self.assertFalse(model.training)
        self.assertEqual(model.config, self.model.config)
        self.assertEqual(tokenizer.to_dict(), self.tokenizer.to_dict())
        ids = [tokenizer.bos_id, *tokenizer.encode_question(self.question), tokenizer.sep_id]
        inputs = torch.tensor([ids])
        for name, weight in self.model.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], weight, rtol=0, atol=0)
        torch.testing.assert_close(model(inputs), self.model(inputs), rtol=0, atol=0)
        self.assertEqual(generate_results(model, tokenizer, [self.question], max_new_tokens=4),
                         generate_results(self.model, self.tokenizer, [self.question], max_new_tokens=4))

    def test_rejects_incompatible_contract_or_incomplete_tensor_data(self):
        write_gguf(self.model, self.tokenizer, self.path)
        raw = self.path.read_bytes()
        version_entry = b"capstone_sql.contract_version" + struct.pack("<II", 4, 3)
        self.assertEqual(raw.count(version_entry), 1)
        v2 = raw.replace(version_entry, version_entry[:-4] + struct.pack("<I", 2))
        for label, corrupted in (("contract v2", v2), ("truncated", raw[:len(raw) // 2])):
            with self.subTest(label=label):
                self.path.write_bytes(corrupted)
                with self.assertRaises(ValueError):
                    load_gguf(self.path)

    def test_inference_cli_runs_with_only_gguf(self):
        # Program a tiny deterministic decoder whose next tokens are SEP -> SELECT -> EOS.
        select_id, = self.tokenizer.encode_sql("SELECT")
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.final_norm.weight.fill_(1)
            embedding = self.model.embedding.embedding.weight
            embedding[self.tokenizer.sep_id, 0] = embedding[select_id, 1] = 1
            head = self.model.lm_head.weight
            head[select_id, 0] = head[self.tokenizer.eos_id, 1] = 1
        write_gguf(self.model, self.tokenizer, self.path)
        result = subprocess.run(
            [sys.executable, "-B", "-m", "inference.back", "--model", str(self.path),
             "--question", self.question, "--device", "cpu"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SELECT", result.stdout)


if __name__ == "__main__":
    unittest.main()
