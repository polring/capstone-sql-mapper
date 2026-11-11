"""Write the front model's training, validation and optional test splits beside the source JSON."""

import argparse
import json
from pathlib import Path
import random
import re

if __package__:
    from .generalized_pairs.generate import write_pairs
else:
    from generalized_pairs.generate import write_pairs


# 팀원 2 원본 SQL . 태그 질문 . DB 매핑 . 사용자 질문
QUESTION_ID = re.compile(r"[1-9][0-9]*(?:\.[1-9][0-9]*){3}")
SPLITS = ("train", "validation", "test")


def split_questions(records: list, *, seed: int, validation_ratio: float = 0.1, test_ratio: float = 0.0):
    """Split by source SQL so every question built from one generalized SQL lands in the same split."""
    if not 0 < validation_ratio < 1 or not 0 <= test_ratio < 1 or validation_ratio + test_ratio >= 1:
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
    needed = 3 if test_ratio else 2
    if len(roots) < needed:
        raise ValueError(f"at least {needed} distinct source SQLs are needed")
    test_count = min(len(roots) - 2, max(1, round(len(roots) * test_ratio))) if test_ratio else 0
    validation_count = min(len(roots) - test_count - 1, max(1, round(len(roots) * validation_ratio)))
    shuffled = random.Random(seed).sample(roots, len(roots))
    assigned = {root: "test" for root in shuffled[:test_count]}
    assigned.update((root, "validation") for root in shuffled[test_count:test_count + validation_count])

    splits = {name: [] for name in SPLITS}
    for row in records:
        splits[assigned.get(row["id"].split(".", 1)[0], "train")].append(row)
    return splits


def split_paths(input_path: Path) -> dict:
    source = Path(input_path)
    return {name: source.with_name(f"{source.stem}_front_{name}.json") for name in SPLITS}


def build(input_path: Path, *, seed: int, validation_ratio: float = 0.1, test_ratio: float = 0.0) -> dict:
    """Write each nonempty split next to the input; return the number of rows per split."""
    input_path = Path(input_path)
    paths = split_paths(input_path)
    if input_path.resolve() in {path.resolve() for path in paths.values()}:
        raise ValueError("the source JSON must be separate from the generated splits")
    with input_path.open(encoding="utf-8-sig") as source:
        splits = split_questions(json.load(source), seed=seed,
                                 validation_ratio=validation_ratio, test_ratio=test_ratio)
    for name, rows in splits.items():
        if rows:
            write_pairs(paths[name], rows)
        else:
            # 이전 실행의 평가셋이 남아 이번 분할과 섞이지 않게 한다.
            paths[name].unlink(missing_ok=True)
    return {name: len(rows) for name, rows in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="natural_questions output JSON, also determines the output paths")
    parser.add_argument("--seed", type=int, required=True, help="seed for the split")
    parser.add_argument("--validation-ratio", type=float, default=0.1,
                        help="fraction of source SQLs for validation (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.0,
                        help="fraction of source SQLs for test; 0 writes no test file (default: 0)")
    args = parser.parse_args()
    try:
        counts = build(args.input, seed=args.seed,
                       validation_ratio=args.validation_ratio, test_ratio=args.test_ratio)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for name, path in split_paths(args.input).items():
        if counts[name]:
            print(f"{path}: {counts[name]:,} {name} questions")


if __name__ == "__main__":
    main()
