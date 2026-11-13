"""Read F32 GGUF v3 files exported for the capstone_sql tagged-SQL decoder."""

import math
import os
from pathlib import Path
import struct

import numpy as np
import torch

from .config import ModelConfig
from .model import BackModel
from .tokenizer import BackTokenizer


def _unpack(file, fmt):
    fmt = "<" + fmt
    size = struct.calcsize(fmt)
    raw = file.read(size)
    if len(raw) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, raw)


def _string(file, end):
    size, = _unpack(file, "Q")
    if size > end - file.tell():
        raise ValueError("truncated GGUF string")
    return file.read(size).decode("utf-8")


def _value(file, end):
    kind, = _unpack(file, "I")
    if kind == 4:  # UINT32
        return _unpack(file, "I")[0]
    if kind == 6:  # FLOAT32
        return _unpack(file, "f")[0]
    if kind == 8:  # STRING
        return _string(file, end)
    if kind == 9:  # ARRAY
        subtype, count = _unpack(file, "IQ")
        if subtype == 4 and count <= (end - file.tell()) // 4:
            return list(_unpack(file, f"{count}I"))
        if subtype == 8 and count <= (end - file.tell()) // 8:
            return [_string(file, end) for _ in range(count)]
    raise ValueError(f"unsupported or malformed GGUF metadata type {kind}")


def _settings(metadata):
    if metadata.get("general.architecture") != "capstone_sql":
        raise ValueError("expected a capstone_sql model")
    if metadata.get("general.alignment") != 32:
        raise ValueError("expected a 32-byte GGUF alignment")
    sql = {k.removeprefix("capstone_sql."): v for k, v in metadata.items() if k.startswith("capstone_sql.")}
    if sql.get("contract_version") != 3 or sql.get("tokenizer.profile") != "tagged_sql_v1":
        raise ValueError("expected capstone_sql contract 3 with the tagged_sql_v1 tokenizer")
    try:
        for field in ("eps", "theta"):
            if type(sql[field]) is not float:
                raise ValueError(f"expected FLOAT32 metadata for capstone_sql.{field}")
        config = ModelConfig(d_model=sql["dim"], num_layers=sql["layers"], num_heads=sql["heads"],
                             ffn_dim=sql["ffn_dim"], context_length=sql["context"], vocab_size=sql["vocab_size"],
                             eps=sql["eps"], theta=sql["theta"])
        for name in ("bos_id", "eos_id", "sep_id", "pad_id", "seed_base", "merge_base"):
            if sql["tokenizer." + name] != getattr(BackTokenizer, name):
                raise ValueError(f"unexpected tokenizer {name}")
        triples = sql["tokenizer.merges"]
        if not isinstance(triples, list) or len(triples) % 3:
            raise ValueError("expected tokenizer merge triples")
        merges = []
        for index in range(0, len(triples), 3):
            left, right, new_id = triples[index:index + 3]
            if new_id != BackTokenizer.merge_base + len(merges):
                raise ValueError("unexpected tokenizer merge ID")
            merges.append((left, right))
        tokenizer = BackTokenizer.from_dict({"version": 1, "vocab_size": config.vocab_size, "merges": merges,
                                             "seeds": sql["tokenizer.seeds"], "sql_ids": sql["tokenizer.sql_ids"]})
    except KeyError as exc:
        raise ValueError(f"missing GGUF metadata: capstone_sql.{exc.args[0]}") from exc
    return config, tokenizer


def load_gguf(path, device="cpu", settings=_settings):
    """Restore the model and tokenizer from an exported GGUF for greedy inference.

    The front model passes its own metadata reader; the tensor layout is shared.
    """
    with Path(path).open("rb") as file:
        file_size = os.fstat(file.fileno()).st_size
        magic, version, tensor_count, metadata_count = _unpack(file, "4sIQQ")
        if (magic, version) != (b"GGUF", 3):
            raise ValueError("expected a GGUF v3 file")
        metadata = {}
        for _ in range(metadata_count):
            key = _string(file, file_size)
            if key in metadata:
                raise ValueError(f"duplicate GGUF metadata: {key}")
            metadata[key] = _value(file, file_size)
        config, tokenizer = settings(metadata)

        tensors = {}
        for _ in range(tensor_count):
            name = _string(file, file_size)
            rank, = _unpack(file, "I")
            if rank not in (1, 2) or name in tensors:
                raise ValueError(f"invalid GGUF tensor descriptor: {name}")
            shape = _unpack(file, f"{rank}Q")[::-1]
            kind, offset = _unpack(file, "IQ")
            if kind != 0 or offset % 32 or not all(shape):
                raise ValueError(f"expected an aligned nonempty F32 tensor: {name}")
            tensors[name] = (offset, shape)
        data_start = (file.tell() + 31) // 32 * 32
        previous_end = 0
        for offset, shape in sorted(tensors.values()):
            tensor_end = offset + 4 * math.prod(shape)
            if offset < previous_end or data_start + tensor_end > file_size:
                raise ValueError("overlapping or truncated GGUF tensors")
            previous_end = tensor_end

        # Reject dimensions inconsistent with the file before allocating the model.
        dim = config.d_model
        per_block = 4 * dim * dim + 3 * dim * config.ffn_dim + 2 * dim
        expected_weights = 2 * config.vocab_size * dim + config.num_layers * per_block + dim
        if expected_weights != sum(math.prod(shape) for _, shape in tensors.values()):
            raise ValueError("GGUF tensor sizes do not match the model configuration")
        # The exporter stores weights, but not the two precomputed F32 RoPE tables.
        # A corrupted context must not turn a small file into a multi-gigabyte allocation.
        rope_bytes = config.context_length * (dim // config.num_heads) * 4
        if rope_bytes > max(256 * 1024**2, file_size):
            raise ValueError("GGUF context requires an excessively large rotary cache for this model")
        model = BackModel(config, tokenizer.sql_ids).float()
        weights = dict(model.named_parameters(remove_duplicate=False))
        if tensors.keys() != weights.keys():
            raise ValueError("GGUF tensor names do not match the model")
        with torch.no_grad():
            for name, parameter in weights.items():
                offset, shape = tensors[name]
                if shape != tuple(parameter.shape):
                    raise ValueError(f"unexpected GGUF tensor shape: {name}")
                file.seek(data_start + offset)
                values = np.fromfile(file, dtype="<f4", count=parameter.numel())
                if values.size != parameter.numel() or not np.isfinite(values).all():
                    raise ValueError(f"truncated or non-finite GGUF weights: {name}")
                parameter.copy_(torch.from_numpy(values.astype(np.float32, copy=False)).reshape(shape))
    return model.to(device).eval(), tokenizer
