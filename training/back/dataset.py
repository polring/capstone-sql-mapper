"""Versioned input/output pairs and shifted, SQL-only training targets."""

from array import array
import json
from pathlib import Path
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data_generation.build_back import sql_family

IGNORE = -100
FAMILIES = ("rows", "aggregate", "group", "scalar")


def validate_pairs(rows, path) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty JSON array")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not all(
            isinstance(row.get(k), str) and row[k].strip() for k in ("input", "output")
        ):
            raise ValueError(f"{path}: row {index + 1} needs nonempty input/output strings")
    return rows


def read_pairs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as file:
        return validate_pairs(json.load(file), path)


def overfit_pairs(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Balanced unique SQLs. Prefer concise examples to keep the smoke test inexpensive."""
    if count < len(FAMILIES):
        raise ValueError("overfit count must cover all four SQL families")
    candidates = list(rows)
    random.Random(seed).shuffle(candidates)
    candidates.sort(key=lambda r: len(r["input"]) + len(r["output"]))
    by_family = {family: [] for family in FAMILIES}
    seen = set()
    for row in candidates:
        sql = row["output"]
        if sql not in seen:
            by_family[sql_family(sql)].append(row)
            seen.add(sql)
    selected = []
    for index, family in enumerate(FAMILIES):
        quota = count // len(FAMILIES) + int(index < count % len(FAMILIES))
        if len(by_family[family]) < quota:
            raise ValueError(f"need {quota} distinct {family} SQLs in the training split")
        selected.extend(by_family[family][:quota])
    return selected


class PairDataset(Dataset):
    """Packed int32 token buffer avoids Python-int storage for large datasets."""

    def __init__(self, rows, tokenizer):
        tokens = array("i")
        self.ends = array("Q", [0])
        self.prefix_lengths = array("I")
        self.max_length = 0
        for i, row in enumerate(rows):
            try:
                q = tokenizer.encode_question(row["input"])
                sql = tokenizer.encode_sql(row["output"])
            except ValueError as exc:
                raise ValueError(f"cannot encode row {row.get('id', i + 1)}: {exc}") from exc
            if tokenizer.decode_sql(sql) != row["output"]:
                raise ValueError(f"SQL round-trip failed on row {row.get('id', i + 1)}")
            tokens.append(tokenizer.bos_id)
            tokens.extend(q)
            tokens.append(tokenizer.sep_id)
            self.prefix_lengths.append(len(q) + 2)
            tokens.extend(sql)
            tokens.append(tokenizer.eos_id)
            self.max_length = max(self.max_length, len(tokens) - self.ends[-1])
            self.ends.append(len(tokens))
        self.tokens = torch.frombuffer(tokens, dtype=torch.int32)

    def __len__(self):
        return len(self.prefix_lengths)

    def __getitem__(self, index):
        return self.tokens[self.ends[index]:self.ends[index + 1]], self.prefix_lengths[index]


def collate_pairs(samples, pad_id):
    sequences, prefixes = zip(*samples)
    ids = pad_sequence(sequences, batch_first=True, padding_value=pad_id).long()
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    positions = torch.arange(targets.size(1))
    targets.masked_fill_(positions[None, :] < torch.tensor(prefixes)[:, None] - 1, IGNORE)
    targets.masked_fill_(targets == pad_id, IGNORE)
    return inputs, targets
