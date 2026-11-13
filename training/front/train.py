"""Train the front model or run its memorization check; the decoder and loop are the back model's."""

from dataclasses import asdict, replace
from functools import partial
import json
import math
import random

import torch
from torch.utils.data import DataLoader

from src.back.model import BackModel
from src.front.tokenizer import FrontTokenizer
from training.back.train import (parse_args, read_configs, save_checkpoint, split_fingerprint, train_epoch,
                                 validation_loss)
from training.front.dataset import (TRAIN_FILE, VALIDATION_FILE, PairDataset, collate_pairs, overfit_pairs,
                                    read_pairs)
from training.front.evaluate import score_pairs


def training_split_paths(args):
    def pair_for(path):
        if path.match(TRAIN_FILE) or path.match(f"*_{TRAIN_FILE}"):
            prefix = path.name[:-len(TRAIN_FILE)]
            return path, path.with_name(prefix + VALIDATION_FILE)
        return None

    if args.train_file is not None:
        pair = pair_for(args.train_file)
        if pair is None:
            raise ValueError(f"--train-file must be named {TRAIN_FILE} or <dataset>_{TRAIN_FILE}")
        return pair

    matches = []
    for path in args.data_dir.iterdir():
        pair = pair_for(path)
        if pair and path.is_file() and (args.overfit or pair[1].is_file()):
            matches.append(pair)
    if not matches:
        raise ValueError(f"no matching dataset splits in {args.data_dir}; run the dataset build first")
    if len(matches) > 1:
        raise ValueError(f"multiple datasets in {args.data_dir}; select one with --train-file")
    return matches[0]


def main(argv=None):
    args = parse_args(argv)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("GPU training requires an available CUDA device and a CUDA-enabled PyTorch installation")
    saved = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 1 or saved.get("model") != "front" or "optimizer_state" not in saved:
            raise ValueError("resume requires the epoch-boundary last.pt checkpoint of a front-model run")
        if args.overfit is not None and args.overfit != saved["overfit"]:
            raise ValueError("overfit must match the checkpoint")
        args.overfit = saved["overfit"]
    if not args.overfit and args.eval_every not in (None, 1):
        raise ValueError("normal training validates every epoch")
    model_config, config = read_configs(args, saved)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((run_dir / f).exists() for f in ("best.pt", "last.pt", "history.jsonl")):
        raise ValueError(f"{run_dir} already contains a run; specify --resume or choose a new --run-dir")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    bf16 = (config.precision != "fp32" and device.type == "cuda" and torch.cuda.is_bf16_supported())
    if config.precision == "bf16" and not bf16:
        raise ValueError("bf16 requested but not supported on the selected CUDA device")

    train_path, validation_path = training_split_paths(args)
    if args.overfit:
        train_rows = overfit_pairs(read_pairs(train_path), args.overfit, config.seed)
        val_rows = train_rows
    else:
        train_rows, val_rows = read_pairs(train_path), read_pairs(validation_path)
    fingerprints = (split_fingerprint(train_rows), split_fingerprint(val_rows))
    if saved and tuple(saved["split_hashes"]) != fingerprints:
        raise ValueError("training or validation data changed since checkpoint")
    if args.overfit:
        (run_dir / "overfit_pairs.json").write_text(json.dumps(train_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Target sentences share their English words with the questions, so both feed the word BPE.
    tokenizer = (FrontTokenizer.from_dict(saved["tokenizer"]) if saved else
                 FrontTokenizer.fit((t for r in train_rows for t in (r["input"], r["output"])),
                                    vocab_size=model_config.vocab_size))
    train_set = PairDataset(train_rows, tokenizer)
    val_set = train_set if args.overfit else PairDataset(val_rows, tokenizer)
    requested_context = saved.get("requested_context_length", model_config.context_length) if saved else model_config.context_length
    required_context = max(train_set.max_length, val_set.max_length)
    if required_context > model_config.context_length:
        if saved:
            raise ValueError("encoded data no longer fits the saved model context")
        model_config = replace(model_config, context_length=required_context)
    collate = partial(collate_pairs, pad_id=tokenizer.pad_id)
    rng = torch.Generator()
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=rng, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, collate_fn=collate)
    model = BackModel(model_config, tokenizer.output_ids).to(device)
    if saved:
        model.load_state_dict(saved["model_state"])
    decay_shapes = [p for p in model.parameters() if p.ndim >= 2]
    non_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay_shapes, "weight_decay": config.weight_decay},
                                   {"params": non_decay, "weight_decay": 0.0}],
                                  lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps)
    update_count = math.ceil(len(train_loader) / config.accumulation_steps)
    total_steps = update_count * config.epochs
    first_epoch, step, bad_epochs, best_score, best_loss = 1, 0, 0, -1.0, math.inf
    if saved:
        optimizer.load_state_dict(saved["optimizer_state"])
        torch.set_rng_state(saved["rng_state"])
        if device.type == "cuda" and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
        first_epoch = saved["epoch"] + 1
        step, bad_epochs = saved["step"], saved["bad_epochs"]
        best_score, best_loss = saved["best_em"], saved["best_loss"]
        del saved
    interval = args.eval_every or (10 if args.overfit else 1)
    metadata = {"format_version": 1, "model": "front", "model_config": asdict(model_config),
                "train_config": asdict(config), "requested_context_length": requested_context,
                "tokenizer": tokenizer.to_dict(), "overfit": args.overfit, "split_hashes": fingerprints}
    print(json.dumps({"run_dir": str(run_dir), "device": str(device), "bf16": bf16,
                      "train_pairs": len(train_rows), "scored_pairs": len(val_rows), "overfit": args.overfit,
                      "encoded_max_length": required_context, "model": asdict(model_config),
                      "train": asdict(config)}), flush=True)
    for epoch in range(first_epoch, config.epochs + 1):
        rng.manual_seed(config.seed + epoch)
        step, train_loss, accuracy = train_epoch(model, train_loader, optimizer, config, device, bf16, step, total_steps)
        if epoch % interval and epoch != config.epochs:
            continue
        metrics = {"epoch": epoch, "step": step, "train_loss": train_loss, "teacher_token_accuracy": accuracy}
        # Bad teacher-forced accuracy makes a fully exact free-running result impossible
        # for those weights; avoid unproductive long generations early in smoke training.
        check_greedy = not args.overfit or accuracy >= 0.98 or epoch == config.epochs
        improved = False
        if check_greedy:
            eval_loss = validation_loss(model, val_loader, device, bf16)
            scores, _ = score_pairs(model, tokenizer, val_rows, device=device, batch_size=args.eval_batch_size)
            metrics.update({"eval_split": "overfit_train" if args.overfit else "validation",
                            "eval_loss": eval_loss, **scores})
            improved = (scores["em"], -eval_loss) > (best_score, -best_loss)
            if improved:
                best_score, best_loss, bad_epochs = scores["em"], eval_loss, 0
            else:
                bad_epochs += 1
        state = {**metadata, "model_state": model.state_dict(), "epoch": epoch, "step": step,
                 "best_em": best_score, "best_loss": best_loss, "bad_epochs": bad_epochs, "metrics": metrics}
        if improved:
            save_checkpoint(run_dir / "best.pt", state)
        state.update({"optimizer_state": optimizer.state_dict(), "rng_state": torch.get_rng_state(),
                      "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None})
        save_checkpoint(run_dir / "last.pt", state)
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        if (args.overfit and best_score == 1.0) or (not args.overfit and config.patience and bad_epochs >= config.patience):
            break
    if args.overfit and best_score != 1.0:
        raise SystemExit(f"memorization check incomplete; best greedy exact match {max(0., best_score):.1%}")


if __name__ == "__main__":
    main()
