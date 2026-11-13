"""Export the front model's tagged-sentence decoder as an F32 GGUF v3 file (capstone_sql contract 3)."""

import argparse
import struct
from pathlib import Path

from src.front.gguf import PROFILE
from training.back.export_gguf import _string, _uints, write_gguf as _write_gguf
from training.front.evaluate import load_model


def _strings(values):
    return struct.pack("<IQ", 8, len(values)) + b"".join(map(_string, values))


def _metadata(config, tokenizer):
    prefix = "capstone_sql."
    layout = tokenizer.to_dict()
    triples = [
        token
        for i, (left, right) in enumerate(layout["merges"], tokenizer.merge_base)
        for token in (left, right, i)
    ]
    integers = {
        "contract_version": 3,
        "dim": config.d_model,
        "layers": config.num_layers,
        "heads": config.num_heads,
        "ffn_dim": config.ffn_dim,
        "context": config.context_length,
        "vocab_size": config.vocab_size,
    }
    for name in ("bos_id", "eos_id", "sep_id", "pad_id", "seed_base", "merge_base"):
        integers["tokenizer." + name] = getattr(tokenizer, name)

    entries = [
        ("general.architecture", 8, _string("capstone_sql")),
        ("general.alignment", 4, struct.pack("<I", 32)),
        (prefix + "tokenizer.profile", 8, _string(PROFILE)),
        (prefix + "tokenizer.seeds", 9, _strings(layout["seeds"])),
        # Schema names restore the tag seeds without opening the target DB at inference time.
        (prefix + "tokenizer.names", 9, _strings(layout["names"])),
        (prefix + "tokenizer.merges", 9, _uints(triples)),
        (prefix + "tokenizer.output_ids", 9, _uints(tokenizer.output_ids)),
    ]
    entries.extend((prefix + key, 4, struct.pack("<I", val)) for key, val in integers.items())
    entries.extend((prefix + key, 6, struct.pack("<f", getattr(config, key))) for key in ("eps", "theta"))
    return entries


def write_gguf(model, tokenizer, destination):
    _write_gguf(model, tokenizer, destination, metadata=_metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.checkpoint.resolve() == args.output.resolve():
        parser.error("--output must differ from --checkpoint")
    model, tokenizer, _ = load_model(args.checkpoint, device="cpu")
    write_gguf(model, tokenizer, args.output)
    print(f"saved {args.output} (capstone_sql contract 3, {PROFILE}, F32)")


if __name__ == "__main__":
    main()
