"""Generate tagged SQL from a generalized question or a JSON file using a back-model GGUF."""

import argparse
import json
from pathlib import Path

import torch

from src.back.generate import generate_results
from src.back.gguf import load_gguf
from src.data_generation.generalized_pairs.generate import write_pairs


def read_questions(path):
    with path.open(encoding="utf-8-sig") as source:
        rows = json.load(source)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array of objects with an input field")
    questions = []
    for index, row in enumerate(rows, 1):
        question = row.get("input") if isinstance(row, dict) else None
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}: row {index} needs a nonempty input string")
        questions.append(question)
    return questions


def predict_pairs(model, tokenizer, questions, batch_size, device):
    for start in range(0, len(questions), batch_size):
        batch = questions[start:start + batch_size]
        results = generate_results(model, tokenizer, batch, device=device)
        for question, result in zip(batch, results, strict=True):
            yield {"input": question, "output": result.sql}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="F32 capstone_sql contract 3 GGUF")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="generalized question with its original tags")
    source.add_argument("--input", type=Path, help="read questions from the input fields of a JSON array")
    parser.add_argument("--output", type=Path, help="write input/output pairs to this JSON file (required with --input)")
    parser.add_argument("--batch-size", type=int, help="questions per batch in file mode (default: 32)")
    parser.add_argument("--device", choices=("cpu", "cuda", "gpu"), required=True)
    args = parser.parse_args(argv)
    if args.input is not None:
        if args.output is None:
            parser.error("--input requires --output")
        if any(path.suffix.lower() != ".json" for path in (args.input, args.output)):
            parser.error("--input and --output must be .json paths")
        if args.output.resolve() in (args.input.resolve(), args.model.resolve()):
            parser.error("--output must be separate from --input and --model")
        if args.batch_size is not None and args.batch_size < 1:
            parser.error("--batch-size must be positive")
    else:
        if args.output is not None or args.batch_size is not None:
            parser.error("--output and --batch-size require --input")
        if not args.question.strip():
            parser.error("--question must contain a question")
    device = "cuda" if args.device == "gpu" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment")
    try:
        model, tokenizer = load_gguf(args.model, device=device)
        if args.input is not None:
            questions = read_questions(args.input)
            pairs = predict_pairs(model, tokenizer, questions, args.batch_size or 32, device)
            write_pairs(args.output, pairs)
            print(f"Saved {len(questions)} input/output pairs to {args.output}")
            return
        result, = generate_results(model, tokenizer, [args.question], device=device)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    if result.stop_reason != "eos":
        parser.exit(1, "Model reached its context limit before completing the SQL.\n")
    print(result.sql)


if __name__ == "__main__":
    main()
