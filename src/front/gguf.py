"""Read F32 GGUF v3 files exported for the front model's DB-tagged sentence decoder."""

from src.back.config import ModelConfig
from src.back.gguf import load_gguf as _load_gguf

from .tokenizer import FrontTokenizer

PROFILE = "front_tagged_v1"


def _settings(metadata):
    if metadata.get("general.architecture") != "capstone_sql":
        raise ValueError("expected a capstone_sql model")
    if metadata.get("general.alignment") != 32:
        raise ValueError("expected a 32-byte GGUF alignment")
    sql = {k.removeprefix("capstone_sql."): v for k, v in metadata.items() if k.startswith("capstone_sql.")}
    if sql.get("contract_version") != 3 or sql.get("tokenizer.profile") != PROFILE:
        raise ValueError(f"expected capstone_sql contract 3 with the {PROFILE} tokenizer")
    try:
        for field in ("eps", "theta"):
            if type(sql[field]) is not float:
                raise ValueError(f"expected FLOAT32 metadata for capstone_sql.{field}")
        config = ModelConfig(d_model=sql["dim"], num_layers=sql["layers"], num_heads=sql["heads"],
                             ffn_dim=sql["ffn_dim"], context_length=sql["context"], vocab_size=sql["vocab_size"],
                             eps=sql["eps"], theta=sql["theta"])
        for name in ("bos_id", "eos_id", "sep_id", "pad_id", "seed_base"):
            if sql["tokenizer." + name] != getattr(FrontTokenizer, name):
                raise ValueError(f"unexpected tokenizer {name}")
        names = sql["tokenizer.names"]
        if not isinstance(names, list):
            raise ValueError("expected tokenizer schema names")
        triples = sql["tokenizer.merges"]
        if not isinstance(triples, list) or len(triples) % 3:
            raise ValueError("expected tokenizer merge triples")
        # Merge IDs start after the schema names, so check them against the stored merge base.
        merge_base = sql["tokenizer.merge_base"]
        merges = []
        for index in range(0, len(triples), 3):
            left, right, new_id = triples[index:index + 3]
            if new_id != merge_base + len(merges):
                raise ValueError("unexpected tokenizer merge ID")
            merges.append((left, right))
        tokenizer = FrontTokenizer.from_dict({"version": 1, "vocab_size": config.vocab_size, "names": names,
                                              "merges": merges, "seeds": sql["tokenizer.seeds"],
                                              "output_ids": sql["tokenizer.output_ids"]})
        if tokenizer.merge_base != merge_base:
            raise ValueError("unexpected tokenizer merge_base")
    except KeyError as exc:
        raise ValueError(f"missing GGUF metadata: capstone_sql.{exc.args[0]}") from exc
    return config, tokenizer


def load_gguf(path, device="cpu"):
    """Restore the front model and its tokenizer; the DB is not needed after export."""
    return _load_gguf(path, device, settings=_settings)
