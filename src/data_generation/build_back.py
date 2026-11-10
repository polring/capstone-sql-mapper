"""Write the back model's training and validation splits beside the source JSON."""

import argparse
from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import random
import re
import shutil

if __package__:
    from .generalized_pairs.generate import write_pairs
else:
    from generalized_pairs.generate import write_pairs


PAIR_ID = re.compile(r"[1-9][0-9]*\.[1-9][0-9]*")
SQL_TOKENS = re.compile(r"'(?:''|[^'])*'|\w+|!=|<=|>=|[^\s]")
TRAIN_FILE = "back_train.json"
VALIDATION_FILE = "back_validation.json"
STAGING_DIR = ".back-build-in-progress"


def sql_family(sql: str) -> str:
    """Classify a generated target into one of the four SQL families."""
    if re.search(r"\(\s*SELECT\b", sql, re.I):
        return "scalar"
    if re.search(r"\bGROUP\s+BY\b", sql, re.I):
        return "group"
    if re.match(r"\s*SELECT\s*\{\s*(COUNT|AVG|SUM|MIN|MAX)\s*\(", sql, re.I):
        return "aggregate"
    return "rows"


@lru_cache(maxsize=1024)
def _sql_key(sql: str) -> str:
    """Ignore formatting and keyword case, while preserving quoted tag patterns."""
    return " ".join(t if t.startswith("'") else t.upper() for t in SQL_TOKENS.findall(sql))


def split_pairs(records: list, *, seed: int, validation_ratio: float = 0.05):
    """Split distinct source SQLs by family; keep all paraphrases in source order."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be strictly between 0 and 1")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(records, list):
        raise ValueError("expected a JSON array of generalized pairs")

    ids = set()
    root_sql = {}
    question_sql = {}
    families = defaultdict(set)
    for row_num, row in enumerate(records, 1):
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), str) and row[key].strip() for key in ("id", "input", "output")
        ):
            raise ValueError(f"row {row_num}: expected nonempty string id, input and output")
        pair_id = row["id"]
        if not PAIR_ID.fullmatch(pair_id) or pair_id in ids:
            raise ValueError(f"row {row_num}: invalid or duplicate pair id {pair_id!r}")
        ids.add(pair_id)
        root = pair_id.split(".", 1)[0]
        key = _sql_key(row["output"])
        if question_sql.setdefault(row["input"], key) != key:
            raise ValueError(f"row {row_num}: same input has conflicting SQL targets")
        if root not in root_sql:
            root_sql[root] = key
            families[sql_family(row["output"])].add(key)
        elif root_sql[root] != key:
            raise ValueError(f"source {root} has conflicting SQL targets")

    # Group on SQL as well as parent ID: imports or regenerated files may
    # assign two IDs to the same target. Sorting also makes reordering harmless.
    all_keys = sorted(set(root_sql.values()))
    if len(all_keys) < 2:
        raise ValueError("at least two distinct source SQLs are needed for train and validation")
    rng = random.Random(seed)
    validation = set()
    for family in sorted(families):
        keys = sorted(families[family])
        if len(keys) > 1:
            count = min(len(keys) - 1, max(1, round(len(keys) * validation_ratio)))
            validation.update(rng.sample(keys, count))
    if not validation:  # tiny input: every represented family is a singleton
        count = min(len(all_keys) - 1, max(1, round(len(all_keys) * validation_ratio)))
        validation.update(rng.sample(all_keys, count))

    train_rows = []
    validation_rows = []
    for row in records:
        split = validation_rows if root_sql[row["id"].split(".", 1)[0]] in validation else train_rows
        split.append(row)
    return train_rows, validation_rows


def split_paths(input_path: Path) -> tuple[Path, Path]:
    source = Path(input_path)
    return (source.with_name(f"{source.stem}_{TRAIN_FILE}"),
            source.with_name(f"{source.stem}_{VALIDATION_FILE}"))


def read_back_splits(train_path: Path, validation_path: Path) -> tuple[list, list]:
    """Read one published generation; reject a build that overlaps the snapshot."""
    paths = (Path(train_path), Path(validation_path))
    directories = {path.parent for path in paths}

    def snapshot():
        for directory in directories:
            if (directory / STAGING_DIR).exists():
                raise ValueError(f"split publication is in progress or interrupted in {directory}")
        stats = [path.stat() for path in paths]
        return [(s.st_ino, s.st_size, s.st_mtime_ns) for s in stats]

    before = snapshot()
    with paths[0].open(encoding="utf-8-sig") as source:
        train = json.load(source)
    with paths[1].open(encoding="utf-8-sig") as source:
        validation = json.load(source)
    if snapshot() != before:
        raise ValueError("splits changed while being read; retry after publication")
    return train, validation


def build(input_path: Path, *, seed: int, validation_ratio: float = 0.05) -> tuple[int, int]:
    """Publish both splits next to the input, using its stem as the common prefix."""
    input_path = Path(input_path)
    train_path, validation_path = split_paths(input_path)
    if input_path.resolve() in {train_path.resolve(), validation_path.resolve()}:
        raise ValueError("the source JSON must be separate from the generated splits")
    with input_path.open(encoding="utf-8-sig") as source:
        train, validation = split_pairs(json.load(source), seed=seed, validation_ratio=validation_ratio)

    # Stage both files on the destination filesystem. The exclusive directory
    # prevents two builders interleaving their replacements.
    stage = input_path.parent / STAGING_DIR
    try:
        stage.mkdir()
    except FileExistsError as exc:
        raise ValueError(f"build already in progress or interrupted: inspect {stage}") from exc
    try:
        staged_train = stage / train_path.name
        staged_validation = stage / validation_path.name
        write_pairs(staged_train, train)
        write_pairs(staged_validation, validation)
        previous = stage / "previous-train.json"
        if train_path.exists():
            try:
                os.link(train_path, previous)
            except OSError:
                shutil.copyfile(train_path, previous)
        try:
            staged_train.replace(train_path)
            staged_validation.replace(validation_path)
        finally:
            if not staged_train.exists() and staged_validation.exists():
                if previous.exists():
                    previous.replace(train_path)
                else:
                    train_path.unlink()
    finally:
        shutil.rmtree(stage)
    return len(train), len(validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="source JSON, also determines both output paths")
    parser.add_argument("--seed", type=int, required=True, help="seed for the train/validation split")
    parser.add_argument("--validation-ratio", type=float, default=0.05,
                        help="fraction of source SQLs per family (default: 0.05)")
    args = parser.parse_args()
    try:
        train, validation = build(args.input, seed=args.seed, validation_ratio=args.validation_ratio)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    train_path, validation_path = split_paths(args.input)
    print(f"{train_path}: {train:,} train pairs\n{validation_path}: {validation:,} validation pairs")


if __name__ == "__main__":
    main()
