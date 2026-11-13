from dataclasses import asdict
import json
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
from src.back.gguf import load_gguf as load_back_gguf
from src.back.model import BackModel
from src.front.gguf import load_gguf
from src.front.tokenizer import FrontTokenizer
from training.front.evaluate import load_model
from training.front.export_gguf import write_gguf

NAMES = ["customers", "orders", "customer_id", "membership", "order_date"]
QUESTION = "Show the membership level of each buyer."
OUTPUT = "Show the {customers.membership, string} of each <customers>."


class FrontGgufTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def setUp(self):
        root = Path(__file__).resolve().parent
        folder = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.path = self.folder / "front.gguf"
        self.tokenizer = FrontTokenizer.fit([QUESTION, OUTPUT] * 3, vocab_size=512, names=NAMES)
        config = ModelConfig(vocab_size=512, d_model=16, num_layers=1, num_heads=2,
                             ffn_dim=32, context_length=64, eps=2**-18, theta=4096.0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            self.model = BackModel(config, self.tokenizer.output_ids).eval()

    @torch.inference_mode()
    def test_load_reproduces_tokenization_weights_and_greedy_decoding(self):
        # Export through a saved checkpoint, the same path the export CLI takes.
        checkpoint = self.folder / "best.pt"
        torch.save({"format_version": 1, "model": "front", "model_config": asdict(self.model.config),
                    "tokenizer": self.tokenizer.to_dict(), "model_state": self.model.state_dict(),
                    "epoch": 1}, checkpoint)
        restored, restored_tokenizer, _ = load_model(checkpoint)
        write_gguf(restored, restored_tokenizer, self.path)
        model, tokenizer = load_gguf(self.path)
        self.assertFalse(model.training)
        self.assertEqual(model.config, self.model.config)
        self.assertEqual(tokenizer.to_dict(), self.tokenizer.to_dict())
        self.assertEqual(tokenizer.names, tuple(NAMES))
        self.assertEqual(tokenizer.encode_output(OUTPUT), self.tokenizer.encode_output(OUTPUT))
        ids = [tokenizer.bos_id, *tokenizer.encode_question(QUESTION), tokenizer.sep_id]
        inputs = torch.tensor([ids])
        for name, weight in self.model.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], weight, rtol=0, atol=0)
        torch.testing.assert_close(model(inputs), self.model(inputs), rtol=0, atol=0)
        self.assertEqual(generate_results(model, tokenizer, [QUESTION], max_new_tokens=6),
                         generate_results(self.model, self.tokenizer, [QUESTION], max_new_tokens=6))

    def test_front_and_back_loaders_reject_each_others_profile(self):
        write_gguf(self.model, self.tokenizer, self.path)
        with self.assertRaisesRegex(ValueError, "tagged_sql_v1"):
            load_back_gguf(self.path)
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
        # Program a tiny deterministic decoder whose next tokens are SEP -> <customers> -> EOS.
        table_id = self.tokenizer.encode_output("<customers>")[1]
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.final_norm.weight.fill_(1)
            embedding = self.model.embedding.embedding.weight
            embedding[self.tokenizer.sep_id, 0] = embedding[table_id, 1] = 1
            head = self.model.lm_head.weight
            head[table_id, 0] = head[self.tokenizer.eos_id, 1] = 1
        write_gguf(self.model, self.tokenizer, self.path)
        source = self.folder / "questions.json"
        target = self.folder / "predictions.json"
        source.write_text(json.dumps([{"input": QUESTION}, {"input": "How many orders?"}]), encoding="utf-8")
        root = Path(__file__).resolve().parents[1]
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        single = subprocess.run(
            [sys.executable, "-B", "-m", "inference.front", "--model", str(self.path),
             "--question", QUESTION, "--device", "cpu"],
            cwd=root, capture_output=True, text=True, env=env,
        )
        self.assertEqual(single.returncode, 0, single.stderr)
        self.assertEqual(single.stdout.strip(), "customers")
        batch = subprocess.run(
            [sys.executable, "-B", "-m", "inference.front", "--model", str(self.path),
             "--input", str(source), "--output", str(target), "--device", "cpu"],
            cwd=root, capture_output=True, text=True, env=env,
        )
        self.assertEqual(batch.returncode, 0, batch.stderr)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")),
                         [{"input": QUESTION, "output": "customers"},
                          {"input": "How many orders?", "output": "customers"}])


if __name__ == "__main__":
    unittest.main()
