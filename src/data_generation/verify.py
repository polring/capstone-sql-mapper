"""Check that the three generation stages and the front splits of one dataset folder agree.

Run with python -m src.data_generation.verify --data-dir data/<version>/generated.
Exits with status 1 when any check fails; INFO lines never fail the run.
"""

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import sys

from ..back.tokenizer import TAG_RE as BACK_TAG_RE
from ..front.tokenizer import FrontTokenizer, schema_names
from .build_front import QUESTION_ID, SPLITS, drop_conflicts, split_paths
from .natural_questions.generate import LIKE_FORMS, TAG_RE, like_mode, parse_tags

PAIRS, MAPPED, QUESTIONS = "pairs.json", "schema_mapping.json", "natural_questions.json"
COLUMN = re.compile(r"TABLE1\.COL(\d)")
VALUE = re.compile(r"\[(TABLE1\.COL\d|AGG_VALUE), \w+, #(\d)\]")
BRACKETS = re.compile(r"[<>{}\[\]]")
LIKE_SQL = re.compile(r"LIKE '(%?)\[(TABLE1\.COL\d), string, #(\d)\](%?)'")
MAX_EXAMPLES = 5


def parent(record_id):
    return record_id.rsplit(".", 1)[0]


def load(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        rows = json.load(source)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array")
    return rows


def in_first_appearance_order(numbers):
    """True when the distinct numbers appear as 1, 2, 3, ... in reading order."""
    seen = list(dict.fromkeys(numbers))
    return seen == list(range(1, len(seen) + 1))


def check_pairs(pairs):
    """Stage 1: tag grammar and the numbering rules the inference mapper will reproduce."""
    failures = {name: [] for name in ("pair id", "tag grammar", "column numbering", "value numbering",
                                      "AGG_VALUE uses one slot", "SQL tags appear in the question")}
    for row in pairs:
        question, sql = row["input"], row["output"]
        if not re.fullmatch(r"[1-9][0-9]*\.[1-9][0-9]*", row["id"]):
            failures["pair id"].append(row["id"])
        if BRACKETS.search(BACK_TAG_RE.sub("", question)):
            failures["tag grammar"].append(row["id"])
        if not in_first_appearance_order(int(n) for n in COLUMN.findall(question)):
            failures["column numbering"].append(row["id"])
        slots = {}
        for key, number in VALUE.findall(question):
            slots.setdefault(key, []).append(int(number))
        if not all(in_first_appearance_order(numbers) for numbers in slots.values()):
            failures["value numbering"].append(row["id"])
        if len(set(slots.get("AGG_VALUE", []))) > 1:
            failures["AGG_VALUE uses one slot"].append(row["id"])
        question_fields = set(re.findall(r"TABLE1\.COL\d, \w+(?:, #\d)?|AGG_VALUE, numeric, #\d", question))
        sql_fields = set(re.findall(r"TABLE1\.COL\d, \w+(?:, #\d)?|AGG_VALUE, numeric, #\d", sql))
        if sql_fields - question_fields:
            failures["SQL tags appear in the question"].append(row["id"])
    return failures


def check_lineage(pairs, mapped, questions):
    """Every stage keeps its parent's sentence, and the mapping restores the back-model input."""
    by_pair = {row["id"]: row for row in pairs}
    by_mapped = {row["id"]: row for row in mapped}
    failures = {name: [] for name in ("stage 2 keeps the stage 1 question", "stage 3 keeps the stage 2 sentence",
                                      "reverse mapping restores the back-model input", "question id")}
    for row in mapped:
        source = by_pair.get(parent(row["id"]))
        if source is None or row["output"] != source["input"]:
            failures["stage 2 keeps the stage 1 question"].append(row["id"])
    for row in questions:
        if not QUESTION_ID.fullmatch(row["id"]):
            failures["question id"].append(row["id"])
        source = by_mapped.get(parent(row["id"]))
        if source is None or row["output"] != source["input"]:
            failures["stage 3 keeps the stage 2 sentence"].append(row["id"])
            continue
        reverse = {real: general for general, real in source["mapping"].items()}
        names = re.compile("|".join(re.escape(name) for name in sorted(reverse, key=len, reverse=True))
                           + r"(?![A-Za-z_0-9.])")
        restored = names.sub(lambda match: reverse[match[0]], row["output"])
        if restored != by_pair[parent(source["id"])]["input"]:
            failures["reverse mapping restores the back-model input"].append(row["id"])
    return failures, by_pair, by_mapped


def check_questions(questions, by_pair, by_mapped, db_path):
    """Stage 3: values, LIKE direction, DB contents and what the front tokenizer can emit."""
    failures = {name: [] for name in ("value tags match the values field", "value text appears in the question",
                                      "LIKE direction matches the SQL", "string value exists in the DB",
                                      "front tokenizer round trip")}
    tokenizer = FrontTokenizer(4096, schema_names(db_path), [])
    matches = {}
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for row in questions:
            source = by_mapped.get(parent(row["id"]))
            if source is None:
                continue  # already reported by the lineage check
            sentence = row["output"]
            try:
                if tokenizer.decode_output(tokenizer.encode_output(sentence)) != sentence:
                    failures["front tokenizer round trip"].append(row["id"])
            except ValueError:
                failures["front tokenizer round trip"].append(row["id"])
            tags = [tag for tag in parse_tags(sentence) if tag.kind == "value"]
            if {tag.mark for tag in tags} != set(row["values"]):
                failures["value tags match the values field"].append(row["id"])
                continue
            if any(str(value["text"]) not in row["input"] for value in row["values"].values()):
                failures["value text appears in the question"].append(row["id"])
            sql = by_pair[parent(source["id"])]["output"]
            expected = {}
            for before, key, number, after in LIKE_SQL.findall(sql):
                mark = f"[{source['mapping'][key]}, string, #{number}]"
                expected[mark] = "contains" if before and after else "ends" if before else "starts"
            for tag in dict.fromkeys(tags):
                mode = None if tag.key == "AGG_VALUE" else like_mode(sentence, tag, LIKE_FORMS)
                if mode != expected.get(tag.mark):
                    failures["LIKE direction matches the SQL"].append(row["id"])
                if tag.key == "AGG_VALUE" or TAG_RE.fullmatch(tag.mark)["value_type"] != "string":
                    continue
                stored = row["values"][tag.mark]["db_value"]
                query = (tag.key, mode, stored)
                if query not in matches:
                    table, column = tag.key.split(".")
                    pattern = {"starts": f"{stored}%", "ends": f"%{stored}", "contains": f"%{stored}%"}.get(mode)
                    matches[query] = db.execute(
                        f'SELECT EXISTS(SELECT 1 FROM "{table}" WHERE "{column}" {"LIKE" if pattern else "="} ?)',
                        (pattern or stored,)).fetchone()[0]
                if not matches[query]:
                    failures["string value exists in the DB"].append(row["id"])
    return failures


def check_splits(questions, paths):
    """The published splits hold every unambiguous question once, grouped by source SQL."""
    failures = {name: [] for name in ("splits cover the unambiguous questions", "source SQL stays in one split")}
    kept, conflicts = drop_conflicts(questions)
    splits = {name: load(path) if path.is_file() else [] for name, path in paths.items()}
    published = Counter(row["id"] for rows in splits.values() for row in rows)
    expected = Counter(row["id"] for row in kept)
    failures["splits cover the unambiguous questions"] = sorted((published - expected) + (expected - published))
    owner = {}
    for name, rows in splits.items():
        for row in rows:
            root = row["id"].split(".", 1)[0]
            if owner.setdefault(root, name) != name:
                failures["source SQL stays in one split"].append(row["id"])
    return failures, {name: len(rows) for name, rows in splits.items()}, conflicts


def verify(data_dir, db_path):
    """Return (failures, info); failures maps each failed check to the offending record ids."""
    data_dir = Path(data_dir)
    pairs, mapped, questions = (load(data_dir / name) for name in (PAIRS, MAPPED, QUESTIONS))
    results = check_pairs(pairs)
    lineage, by_pair, by_mapped = check_lineage(pairs, mapped, questions)
    results.update(lineage)
    results.update(check_questions(questions, by_pair, by_mapped, db_path))
    split_failures, split_counts, conflicts = check_splits(questions, split_paths(data_dir / QUESTIONS))
    results.update(split_failures)

    mapped_parents = {parent(row["id"]) for row in mapped}
    unmapped = [row["id"] for row in pairs if row["id"] not in mapped_parents]
    info = {
        "records": {PAIRS: len(pairs), MAPPED: len(mapped), QUESTIONS: len(questions)},
        "splits": split_counts,
        "source SQLs": len({row["id"].split(".", 1)[0] for row in pairs}),
        "pairs without a DB mapping": len(unmapped),
        "source SQLs without a DB mapping": len({i.split(".", 1)[0] for i in unmapped}
                                                - {i.split(".", 1)[0] for i in mapped_parents}),
        "questions dropped for conflicting targets": sum(conflicts.values()),
    }
    return results, info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help=f"folder holding {PAIRS}, {MAPPED}, {QUESTIONS} and the front splits")
    parser.add_argument("--db", type=Path, default=Path("data/v1/shop.db"))
    args = parser.parse_args(argv)
    try:
        results, info = verify(args.data_dir, args.db)
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        parser.error(f"cannot verify {args.data_dir}: {type(error).__name__}: {error}")
    for name, value in info.items():
        print(f"INFO  {name}: {value}")
    for name, ids in results.items():
        if ids:
            print(f"FAIL  {name}: {len(ids)} records, e.g. {', '.join(ids[:MAX_EXAMPLES])}")
        else:
            print(f"PASS  {name}")
    failed = sum(bool(ids) for ids in results.values())
    print(f"{len(results) - failed} of {len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
