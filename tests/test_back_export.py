import io
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import torch

from src.back.config import ModelConfig
from src.back.model import BackModel
from src.back.tokenizer import BackTokenizer
from training.back.export_gguf import write_gguf


class ExportTest(unittest.TestCase):
    def test_f32_weights_and_tokenizer_metadata_round_trip(self):
        tok = BackTokenizer.fit(["what is the maximum in <TABLE1>?"] * 2, vocab_size=512)
        model = BackModel(ModelConfig(vocab_size=512, d_model=16, num_layers=1,
                                      num_heads=2, ffn_dim=32), tok.sql_ids)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            target = Path(folder) / "back.gguf"
            write_gguf(model, tok, target)
            raw = target.read_bytes()
        reader = io.BytesIO(raw)

        def unpack(fmt):
            return struct.unpack(fmt, reader.read(struct.calcsize(fmt)))

        def string():
            return reader.read(unpack("<Q")[0]).decode("utf8")

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                inner, count = unpack("<IQ")
                return [value(inner) for _ in range(count)]
            return unpack({4: "<I", 6: "<f"}[kind])[0]

        magic, version, count, num_meta = unpack("<4sIQQ")
        self.assertEqual((magic, version), (b"GGUF", 3))
        metadata = {}
        for _ in range(num_meta):
            key = string()
            metadata[key] = value(unpack("<I")[0])
        prefix = "capstone_sql."
        self.assertNotIn(prefix + "contract_version", metadata)
        self.assertNotIn(prefix + "tokenizer.profile", metadata)
        self.assertEqual(metadata[prefix + "dim"], model.config.d_model)
        self.assertEqual(metadata[prefix + "tokenizer.sql_ids"], list(tok.sql_ids))
        self.assertEqual(metadata[prefix + "tokenizer.seeds"], tok.to_dict()["seeds"])
        triples = metadata[prefix + "tokenizer.merges"]
        self.assertEqual(list(zip(triples[::3], triples[1::3])), tok.merges)
        self.assertEqual(triples[2::3], list(range(tok.merge_base, tok.merge_base + len(tok.merges))))
        descriptions = {}
        for _ in range(count):
            name = string()
            dims = unpack(f"<{unpack('<I')[0]}Q")
            kind, offset = unpack("<IQ")
            self.assertEqual((kind, offset % 32), (0, 0))
            descriptions[name] = (tuple(reversed(dims)), offset)
        self.assertEqual(set(descriptions), set(model.state_dict()))
        start = (reader.tell() + 31) // 32 * 32
        for name, expected in model.state_dict().items():
            shape, offset = descriptions[name]
            actual = np.frombuffer(raw, dtype="<f4", count=expected.numel(), offset=start + offset)
            np.testing.assert_array_equal(actual.reshape(shape), expected.numpy())


if __name__ == "__main__":
    unittest.main()
