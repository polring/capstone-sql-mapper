"""Train or run a memorization check with CLI run settings and JSON hyperparameters."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields, replace
from functools import partial
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.back.config import ModelConfig
from src.back.model import BackModel
from src.back.tokenizer import BackTokenizer
from src.data_generation.build_back import TRAIN_FILE, VALIDATION_FILE, read_back_splits
from training.back.config import TrainConfig
from training.back.dataset import IGNORE, PairDataset, collate_pairs, overfit_pairs, read_pairs, validate_pairs
from training.back.evaluate import score_pairs


def save_checkpoint(path, state):
    # A crash during serialization must leave the previous resumable file intact.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, delete=False) as temp:
        tmp = Path(temp.name)
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def split_fingerprint(rows):
    digest = sha256()
    for row in rows:
        for key in ("input", "output"):
            field = row[key].encode("utf-8")
            digest.update(len(field).to_bytes(8, "little"))
            digest.update(field)
    return digest.hexdigest()


def token_loss(model, inputs, targets):
    logits = model(inputs, compact=True)
    labels = model.sql_targets(targets)
    loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), ignore_index=IGNORE,
                           reduction="sum")
    return loss, logits.detach(), labels


def autocast_context(bf16):
    return torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()


@torch.inference_mode()
def validation_loss(model, loader, device, bf16):
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_tokens = 0
    for inputs, targets in loader:
        total_tokens += int(targets.ne(IGNORE).sum())
        with autocast_context(bf16):
            loss, _, _ = token_loss(model, inputs.to(device), targets.to(device))
        total_loss += loss
    return float(total_loss) / total_tokens


def train_epoch(model, loader, optimizer, config, device, bf16, step, total_steps):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    window_tokens = epoch_tokens = 0
    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        tokens = int(targets.ne(IGNORE).sum())
        window_tokens += tokens
        epoch_tokens += tokens
        with autocast_context(bf16):
            loss, logits, labels = token_loss(model, inputs.to(device), targets.to(device))
        loss.backward()
        total_loss += loss.detach()
        correct += logits.argmax(-1).eq(labels).sum()
        if batch_index % config.accumulation_steps == 0 or batch_index == len(loader):
            step += 1
            # Normalize by all non-padding SQL tokens, including in a partial last window.
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_tokens)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate(step, total_steps)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window_tokens = 0
    return step, float(total_loss) / epoch_tokens, float(correct) / epoch_tokens


def check_object(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {label} keys: {', '.join(sorted(unknown))}")


def config_overrides(values, config_class, section):
    definitions = {field.name: field for field in fields(config_class)}
    check_object(values, definitions.keys(), section)
    overrides = {name: value for name, value in values.items() if value is not None}
    for name, value in overrides.items():
        expected = definitions[name].type
        if expected is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        else:
            valid = type(value) is expected
        if not valid:
            raise ValueError(f"{section}.{name} must be a finite number" if expected is float
                             else f"{section}.{name} must be of type {expected.__name__}")
    return overrides


def unique_keys(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def load_options(path):
    if path.suffix.lower() != ".json":
        raise ValueError("expected a .json settings file")
    with path.open(encoding="utf-8-sig") as source:
        options = json.load(source, object_pairs_hook=unique_keys)
    defaults = {
        "overfit": None, "threads": None, "eval_every": None, "eval_batch_size": 32,
        "model_config": {}, "train_config": {},
    }
    if isinstance(options, dict):
        cli_keys = options.keys() & {"data_dir", "train_file", "run_dir", "device", "resume", "seed"}
        if cli_keys:
            flags = ", ".join("--" + key.replace("_", "-") for key in sorted(cli_keys))
            raise ValueError(f"specify these settings on the command line: {flags}")
    check_object(options, defaults.keys() | {"description"}, "top-level")
    if not isinstance(options.pop("description", ""), str):
        raise ValueError("description must be a string")
    overrides = {name: value for name, value in options.items() if value is not None}
    run = argparse.Namespace(**(defaults | overrides))
    run.model_config = config_overrides(run.model_config, ModelConfig, "model_config")
    if isinstance(run.train_config, dict) and "seed" in run.train_config:
        raise ValueError("specify the training seed on the command line with --seed")
    run.train_config = config_overrides(run.train_config, TrainConfig, "train_config")
    for name in ("threads", "overfit", "eval_every", "eval_batch_size"):
        value = getattr(run, name)
        if value is None:
            continue
        minimum = 4 if name == "overfit" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    return run


def cli_path(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("expected a nonempty path")
    return Path(value)


def cli_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "gpu":
        return "cuda"
    try:
        torch.device(value)
    except RuntimeError as exc:
        raise argparse.ArgumentTypeError(f"invalid device: {value}") from exc
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True, type=cli_path, help="training settings JSON")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--data-dir", type=cli_path, help="directory containing a single dataset's splits")
    data.add_argument("--train-file", type=cli_path, help="training split JSON; selects its matching validation file")
    parser.add_argument("--run-dir", required=True, type=cli_path, help="directory for checkpoints and training history")
    parser.add_argument("--seed", required=True, type=int, help="training random seed")
    parser.add_argument("--device", required=True, type=cli_device, help="training device, e.g. cpu, cuda, gpu or auto")
    parser.add_argument("--resume", type=cli_path, help="last.pt checkpoint to resume")
    args = parser.parse_args(argv)
    try:
        options = load_options(args.config)
    except (OSError, ValueError, OverflowError) as exc:
        parser.error(f"{args.config}: {exc}")
    return argparse.Namespace(**vars(args), **vars(options))


def read_configs(args, saved):
    if saved:
        model = ModelConfig(**saved["model_config"])
        train = TrainConfig(**saved["train_config"])
        if args.seed != train.seed:
            raise ValueError("--seed must match the checkpoint seed while resuming")
        for section, base, overrides in (
            ("model_config", model, args.model_config), ("train_config", train, args.train_config)
        ):
            for name, value in overrides.items():
                if (section, name) == ("model_config", "context_length") and value == saved.get("requested_context_length"):
                    continue
                if (section, name) != ("train_config", "epochs") and value != getattr(base, name):
                    raise ValueError(f"cannot change {section}.{name} while resuming")
        return model, replace(train, **args.train_config)
    defaults = {"epochs": 400, "patience": 0} if args.overfit else {}
    return ModelConfig(**args.model_config), TrainConfig(seed=args.seed, **(defaults | args.train_config))


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
        if saved.get("format_version") != 1 or "optimizer_state" not in saved:
            raise ValueError("resume requires the epoch-boundary last.pt checkpoint")
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
        train_rows, val_rows = read_back_splits(train_path, validation_path)
        validate_pairs(train_rows, train_path)
        validate_pairs(val_rows, validation_path)
    fingerprints = (split_fingerprint(train_rows), split_fingerprint(val_rows))
    if saved and tuple(saved["split_hashes"]) != fingerprints:
        raise ValueError("training or validation data changed since checkpoint")
    if args.overfit:
        (run_dir / "overfit_pairs.json").write_text(json.dumps(train_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tokenizer = (BackTokenizer.from_dict(saved["tokenizer"]) if saved else
                 BackTokenizer.fit((r["input"] for r in train_rows), vocab_size=model_config.vocab_size))
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
    model = BackModel(model_config, tokenizer.sql_ids).to(device)
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
    metadata = {"format_version": 1, "model_config": asdict(model_config), "train_config": asdict(config),
                "requested_context_length": requested_context,
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
