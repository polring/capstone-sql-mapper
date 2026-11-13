"""Convert front-model JSON pairs to back-model inputs and restoration records."""

import argparse
import json
from pathlib import Path

from src.data_generation.generalized_pairs.generate import write_pairs
from src.mapper.mapper import map_question


def map_records(records):
    if not isinstance(records, list):
        raise ValueError("expected a JSON array of front-model input/output pairs")
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"row {index}: expected a JSON object")
        try:
            question, mapping = map_question(record.get("input"), record.get("output"))
        except ValueError as error:
            raise ValueError(f"row {index}: {error}") from error
        result = {"input": question, "mapping": mapping}
        if "id" in record:
            result["id"] = record["id"]
        yield result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="front-model inference JSON")
    parser.add_argument("--output", type=Path, required=True, help="back-model inputs with mapping records")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must be separate from --input")
    try:
        with args.input.open(encoding="utf-8-sig") as source:
            records = json.load(source)
        write_pairs(args.output, map_records(records))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
