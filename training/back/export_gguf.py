"""Export the tagged-SQL decoder as an F32 GGUF v3 file (capstone_sql contract 3)."""

import argparse
import io
import struct
import tempfile
from pathlib import Path

import torch

from training.back.evaluate import load_model


def _string(text):
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _uints(values):
    return struct.pack("<IQ", 4, len(values)) + struct.pack(f"<{len(values)}I", *values)


def _metadata(config, tokenizer):
    prefix = "capstone_sql."
    layout = tokenizer.to_dict()
    seeds = layout["seeds"]
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
        (prefix + "tokenizer.profile", 8, _string("tagged_sql_v1")),
        (prefix + "tokenizer.seeds", 9, struct.pack("<IQ", 8, len(seeds)) + b"".join(map(_string, seeds))),
        (prefix + "tokenizer.merges", 9, _uints(triples)),
        (prefix + "tokenizer.sql_ids", 9, _uints(tokenizer.sql_ids)),
    ]
    entries.extend((prefix + key, 4, struct.pack("<I", val)) for key, val in integers.items())
    entries.extend((prefix + key, 6, struct.pack("<f", getattr(config, key))) for key in ("eps", "theta"))
    return entries


def write_gguf(model, tokenizer, destination, metadata=_metadata):
    """The front model passes its own tokenizer metadata; tensors are laid out the same way."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = metadata(model.config, tokenizer)
    # Keep both embedding and output descriptors even when training shares weights.
    tensors = list(model.named_parameters(remove_duplicate=False))
    header = io.BytesIO()
    header.write(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(entries)))
    for key, kind, payload in entries:
        header.write(_string(key) + struct.pack("<I", kind) + payload)
    offset = 0
    for name, tensor in tensors:
        dims = tuple(reversed(tensor.shape))
        header.write(_string(name) + struct.pack("<I", len(dims)))
        header.write(struct.pack(f"<{len(dims)}Q", *dims))
        header.write(struct.pack("<IQ", 0, offset))
        offset += tensor.numel() * 4
        offset += -offset % 32

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            tmp_path = Path(output.name)
            output.write(header.getbuffer())
            output.write(bytes(-output.tell() % 32))
            for name, tensor in tensors:
                data = tensor.detach().cpu().float().contiguous()
                if not torch.isfinite(data).all():
                    raise ValueError(f"non-finite weights: {name}")
                output.write(data.numpy().astype("<f4", copy=False).tobytes())
                output.write(bytes(-output.tell() % 32))
        tmp_path.replace(destination)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.checkpoint.resolve() == args.output.resolve():
        parser.error("--output must differ from --checkpoint")
    model, tokenizer, _ = load_model(args.checkpoint, device="cpu")
    write_gguf(model, tokenizer, args.output)
    print(f"saved {args.output} (capstone_sql contract 3, tagged_sql_v1, F32)")


if __name__ == "__main__":
    main()
