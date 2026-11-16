"""Write the front model's training, validation and optional test splits beside the source JSON."""

import argparse
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys

if __package__:
    from .build_back import read_back_splits, split_paths as back_split_paths
    from .generalized_pairs.generate import write_pairs
else:
    from build_back import read_back_splits, split_paths as back_split_paths
    from generalized_pairs.generate import write_pairs


# 팀원 2 원본 SQL . 태그 질문 . DB 매핑 . 사용자 질문
QUESTION_ID = re.compile(r"[1-9][0-9]*(?:\.[1-9][0-9]*){3}")
SPLITS = ("train", "validation", "test")
STAGING_DIR = ".front-build-in-progress"


def drop_conflicts(records: list):
    """Remove every question that was generated with two different targets; return (kept, conflicts).

    conflicts maps each removed question to the number of rows dropped for it.
    """
    outputs = {}
    for row in records:
        if isinstance(row, dict) and isinstance(row.get("input"), str):
            outputs.setdefault(row["input"], set()).add(row.get("output"))
    conflicting = {question for question, targets in outputs.items() if len(targets) > 1}
    kept, conflicts = [], {}
    for row in records:
        if isinstance(row, dict) and row.get("input") in conflicting:
            conflicts[row["input"]] = conflicts.get(row["input"], 0) + 1
        else:
            kept.append(row)
    return kept, conflicts


def back_validation_roots(back_input: Path) -> tuple[set, set]:
    """Source SQL ids of the published back-model splits: (all ids, validation ids)."""
    train, validation = read_back_splits(*back_split_paths(back_input))
    validation_roots = {row["id"].split(".", 1)[0] for row in validation}
    return {row["id"].split(".", 1)[0] for row in train} | validation_roots, validation_roots


def split_questions(records: list, *, seed: int, validation_ratio: float = 0.1, test_ratio: float = 0.0,
                    shared=None):
    """Split by source SQL so every question built from one generalized SQL lands in the same split.

    shared is (all source ids, validation source ids) of the back-model split. When given, the
    front model validates on exactly the back model's validation SQLs instead of drawing its own.
    """
    if shared is not None:
        if test_ratio:
            raise ValueError("a test split cannot be drawn when the back-model split is shared")
    elif not 0 < validation_ratio < 1 or not 0 <= test_ratio < 1 or validation_ratio + test_ratio >= 1:
        raise ValueError("ratios must satisfy 0 < validation, 0 <= test and validation + test < 1")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(records, list):
        raise ValueError("expected a JSON array of natural questions")

    ids = set()
    question_output = {}
    for row_num, row in enumerate(records, 1):
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), str) and row[key].strip() for key in ("id", "input", "output")
        ):
            raise ValueError(f"row {row_num}: expected nonempty string id, input and output")
        if not QUESTION_ID.fullmatch(row["id"]) or row["id"] in ids:
            raise ValueError(f"row {row_num}: invalid or duplicate question id {row['id']!r}")
        ids.add(row["id"])
        # 같은 질문에 정답이 둘이면 앞모델이 배울 수 없다.
        if question_output.setdefault(row["input"], row["output"]) != row["output"]:
            raise ValueError(f"row {row_num}: same input has conflicting outputs: {row['input']!r}")

    # 숫자 순으로 정렬해 입력 순서와 무관하게 같은 시드면 같은 분할이 나온다.
    roots = sorted({row["id"].split(".", 1)[0] for row in records}, key=int)
    if shared is not None:
        known, validation_roots = shared
        unknown = [root for root in roots if root not in known]
        if unknown:
            raise ValueError(f"source SQL {unknown[0]} is missing from the back-model splits")
        assigned = {root: "validation" for root in roots if root in validation_roots}
        if not assigned or len(assigned) == len(roots):
            raise ValueError("the shared split leaves the front training or validation split empty")
        return _assign(records, assigned)
    needed = 3 if test_ratio else 2
    if len(roots) < needed:
        raise ValueError(f"at least {needed} distinct source SQLs are needed")
    test_count = min(len(roots) - 2, max(1, round(len(roots) * test_ratio))) if test_ratio else 0
    validation_count = min(len(roots) - test_count - 1, max(1, round(len(roots) * validation_ratio)))
    shuffled = random.Random(seed).sample(roots, len(roots))
    assigned = {root: "test" for root in shuffled[:test_count]}
    assigned.update((root, "validation") for root in shuffled[test_count:test_count + validation_count])
    return _assign(records, assigned)


def _assign(records, assigned):
    splits = {name: [] for name in SPLITS}
    for row in records:
        splits[assigned.get(row["id"].split(".", 1)[0], "train")].append(row)
    return splits


def split_paths(input_path: Path) -> dict:
    source = Path(input_path)
    return {name: source.with_name(f"{source.stem}_front_{name}.json") for name in SPLITS}


def build(input_path: Path, *, seed: int, validation_ratio: float = 0.1, test_ratio: float = 0.0,
          back_input: Path | None = None) -> tuple[dict, dict]:
    """Write each nonempty split next to the input; return rows per split and the dropped conflicts."""
    input_path = Path(input_path)
    paths = split_paths(input_path)
    if input_path.resolve() in {path.resolve() for path in paths.values()}:
        raise ValueError("the source JSON must be separate from the generated splits")
    with input_path.open(encoding="utf-8-sig") as source:
        records = json.load(source)
    if not isinstance(records, list):
        raise ValueError("expected a JSON array of natural questions")
    # 같은 질문에 정답이 둘이면 앞모델이 배울 수 없으니 그 질문의 예시만 빼고 계속한다.
    records, conflicts = drop_conflicts(records)
    shared = None if back_input is None else back_validation_roots(back_input)
    splits = split_questions(records, seed=seed, validation_ratio=validation_ratio,
                             test_ratio=test_ratio, shared=shared)
    publish(paths, splits)
    return {name: len(rows) for name, rows in splits.items()}, conflicts


def publish(paths: dict, splits: dict):
    """Replace every split together; on any failure restore all files of the previous build.

    An empty split removes its file so a stale evaluation set never mixes with this split.
    """
    directory = next(iter(paths.values())).parent
    stage = directory / STAGING_DIR
    try:
        stage.mkdir()
    except FileExistsError as exc:
        raise ValueError(f"build already in progress or interrupted: inspect {stage}") from exc
    try:
        for name, rows in splits.items():
            if rows:
                write_pairs(stage / paths[name].name, rows)
        previous = {}
        for name, path in paths.items():
            if path.exists():
                previous[name] = stage / f"previous-{path.name}"
                try:
                    os.link(path, previous[name])
                except OSError:
                    shutil.copyfile(path, previous[name])
        replaced = []
        try:
            for name, path in paths.items():
                replaced.append(name)
                if splits[name]:
                    (stage / path.name).replace(path)
                else:
                    path.unlink(missing_ok=True)
        except BaseException:
            for name in replaced:
                if name in previous:
                    previous[name].replace(paths[name])
                else:
                    paths[name].unlink(missing_ok=True)
            raise
    finally:
        shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="natural_questions output JSON, also determines the output paths")
    parser.add_argument("--seed", type=int, required=True, help="seed for the split")
    parser.add_argument("--validation-ratio", type=float, default=0.1,
                        help="fraction of source SQLs for validation (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.0,
                        help="fraction of source SQLs for test; 0 writes no test file (default: 0)")
    parser.add_argument("--back-input", type=Path,
                        help="back-model source JSON whose published splits are reused, so both models "
                             "validate on the same source SQLs (ignores --validation-ratio)")
    args = parser.parse_args()
    try:
        counts, conflicts = build(args.input, seed=args.seed, validation_ratio=args.validation_ratio,
                                  test_ratio=args.test_ratio, back_input=args.back_input)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if conflicts:
        print(f"dropped {sum(conflicts.values()):,} rows of {len(conflicts):,} questions with conflicting outputs:",
              file=sys.stderr)
        for question, count in conflicts.items():
            print(f"  {count} x {question!r}", file=sys.stderr)
    for name, path in split_paths(args.input).items():
        if counts[name]:
            print(f"{path}: {counts[name]:,} {name} questions")


if __name__ == "__main__":
    main()
