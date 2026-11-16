"""Score greedy tagged-sentence predictions against the specified evaluation dataset."""

import argparse
import json
from pathlib import Path
import re

import torch

from src.back.config import ModelConfig
from src.back.generate import generate_results
from src.back.model import BackModel
from src.front.tokenizer import FrontTokenizer, TAG_RE
from training.front.dataset import read_pairs

TAG_FRAGMENT = re.compile(r"[][<>{}]")


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("model") != "front":
        raise ValueError("unsupported front-model checkpoint")
    tokenizer = FrontTokenizer.from_dict(checkpoint["tokenizer"])
    config = ModelConfig(**checkpoint["model_config"])
    model = BackModel(config, tokenizer.output_ids)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), tokenizer, checkpoint


def frame(sentence):
    """The sentence with every tag replaced by one placeholder."""
    return TAG_RE.sub("\0", sentence)


@torch.inference_mode()
def score_pairs(model, tokenizer, rows, *, device, batch_size, with_predictions=False):
    if batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    model.eval()
    correct = tags_correct = frames_correct = malformed = non_eos = 0
    predictions = []
    for begin in range(0, len(rows), batch_size):
        batch = rows[begin:begin + batch_size]
        questions = [r["input"] for r in batch]
        outputs = generate_results(model, tokenizer, questions, device=device)
        if len(outputs) != len(batch):
            raise RuntimeError("generator returned the wrong batch size")
        for row, generation in zip(batch, outputs):
            prediction = generation.sql
            finished = generation.stop_reason == "eos"
            exact = finished and prediction == row["output"]
            same_tags = finished and TAG_RE.findall(prediction) == TAG_RE.findall(row["output"])
            same_frame = finished and frame(prediction) == frame(row["output"])
            broken = bool(TAG_FRAGMENT.search(TAG_RE.sub("", prediction)))
            correct += exact
            tags_correct += same_tags
            frames_correct += same_frame
            malformed += broken
            non_eos += not finished
            if with_predictions:
                predictions.append({
                    **row, "prediction": prediction, "stop_reason": generation.stop_reason, "exact": exact,
                    "tags_exact": same_tags, "frame_exact": same_frame, "malformed_tag": broken,
                })
    total = len(rows)
    scores = {"correct": correct, "total": total, "em": correct / total,
              "tag_em": tags_correct / total, "frame_em": frames_correct / total,
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
