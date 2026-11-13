"""Front-model splits and memorization sample; packing and batching reuse the back model's."""

import random

from training.back.dataset import IGNORE, PairDataset, collate_pairs, read_pairs, validate_pairs  # noqa: F401

TRAIN_FILE = "front_train.json"
VALIDATION_FILE = "front_validation.json"


def source_sql(row) -> str:
    return row.get("id", "").split(".", 1)[0]


def overfit_pairs(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Distinct targets spread over source SQLs. Prefer concise examples to keep the smoke test inexpensive."""
    candidates = list(rows)
    random.Random(seed).shuffle(candidates)
    candidates.sort(key=lambda r: len(r["input"]) + len(r["output"]))
    by_source = {}
    seen = set()
    for row in candidates:
        if row["output"] not in seen:
            by_source.setdefault(source_sql(row), []).append(row)
            seen.add(row["output"])
    if len(seen) < count:
        raise ValueError(f"need {count} distinct target sentences in the training split")
    # Take one sentence per source SQL in turn so the sample covers as many SQL shapes as possible.
    queues = list(by_source.values())
    selected = []
    while len(selected) < count:
        for queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.pop(0))
    return selected
