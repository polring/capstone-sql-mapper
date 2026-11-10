"""Score greedy SQL predictions against the specified evaluation dataset."""

import argparse
import json
from pathlib import Path
import re

import torch

from src.back.config import ModelConfig
from src.back.generate import generate_results
from src.back.model import BackModel
from src.back.tokenizer import BackTokenizer, TAG
from src.data_generation.build_back import sql_family
from training.back.dataset import FAMILIES, read_pairs


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported back-model checkpoint version")
    tokenizer = BackTokenizer.from_dict(checkpoint["tokenizer"])
    config = ModelConfig(**checkpoint["model_config"])
    model = BackModel(config, tokenizer.sql_ids)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), tokenizer, checkpoint


TABLE = re.compile(r"\bTABLE[0-9]+\b")
COLUMN = re.compile(r"\bTABLE[0-9]+\.COL[0-9]+,\s*(?:string|numeric|date)\b")
VALUE = re.compile(r"\[(?:TABLE[0-9]+\.COL[0-9]+|AGG_VALUE),\s*(?:string|numeric|date),\s*#[0-9]+\]")
# SELECT wraps its complete aggregate expression in braces; its column, if any, is a tag.
SQL_TAG = re.compile(rf"\{{(?:COUNT\(\*\)|(?:AVG|SUM|MIN|MAX)\({TAG}\))\}}|{TAG}")
TAG_FRAGMENT = re.compile(r"TABLE[0-9]+|COL[0-9]+|AGG_VALUE|#[0-9]+|[][{}]")


def unmapped_tags(question, prediction):
    """A column mention implies its table; value slots include both the column and #n."""
    refs = set()
    for pattern in (TABLE, COLUMN, VALUE):
        source = {ref.replace(" ", "") for ref in pattern.findall(question)}
        refs.update(ref for ref in pattern.findall(prediction) if ref.replace(" ", "") not in source)
    return sorted(refs)


@torch.inference_mode()
def score_pairs(model, tokenizer, rows, *, device, batch_size, with_predictions=False):
    if batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    model.eval()
    families = {family: {"correct": 0, "total": 0} for family in FAMILIES}
    correct = unmapped = malformed = tag_errors = non_eos = 0
    predictions = []
    for begin in range(0, len(rows), batch_size):
        batch = rows[begin:begin + batch_size]
        questions = [r["input"] for r in batch]
        outputs = generate_results(model, tokenizer, questions, device=device)
        if len(outputs) != len(batch):
            raise RuntimeError("generator returned the wrong batch size")
        for row, generation in zip(batch, outputs):
            exact = generation.stop_reason == "eos" and generation.sql == row["output"]
            family = sql_family(row["output"])
            missing = unmapped_tags(row["input"], generation.sql)
            broken = bool(TAG_FRAGMENT.search(SQL_TAG.sub("", generation.sql)))
            correct += exact
            unmapped += bool(missing)
            malformed += broken
            tag_errors += bool(missing) or broken
            non_eos += generation.stop_reason != "eos"
            families[family]["total"] += 1
            families[family]["correct"] += exact
            if with_predictions:
                predictions.append({
                    **row, "family": family, "prediction": generation.sql,
                    "stop_reason": generation.stop_reason, "exact": exact, "unmapped_tags": missing,
                    "malformed_tag": broken,
                })
    for counts in families.values():
        counts["em"] = counts["correct"] / counts["total"] if counts["total"] else None
    scores = {"correct": correct, "total": len(rows), "em": correct / len(rows), "families": families,
              "rows_with_tag_errors": tag_errors, "rows_with_unmapped_tags": unmapped,
              "rows_with_malformed_tags": malformed, "without_eos": non_eos}
    return scores, predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="evaluation dataset JSON path")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", required=True, help="PyTorch device for evaluation (e.g. cpu or cuda)")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args(argv)
    if args.predictions is not None and args.predictions.resolve() in {
        args.data.resolve(), args.checkpoint.resolve()
    }:
        parser.error("--predictions must be separate from --data and --checkpoint")
    if args.threads:
        torch.set_num_threads(args.threads)
    model, tokenizer, checkpoint = load_model(args.checkpoint, args.device)
    scores, predictions = score_pairs(model, tokenizer, read_pairs(args.data), device=args.device,
                                      batch_size=args.batch_size, with_predictions=bool(args.predictions))
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        args.predictions.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"data": str(args.data), "checkpoint_epoch": checkpoint["epoch"], **scores},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
