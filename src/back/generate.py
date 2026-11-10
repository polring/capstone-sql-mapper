"""Greedy SQL decoding from generalized questions, with no target SQL input."""

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class Generation:
    sql: str
    stop_reason: str


@torch.inference_mode()
def generate_results(model, tokenizer, questions, *, device=None, max_new_tokens=None):
    if isinstance(questions, str):
        raise TypeError("pass a sequence of questions, not a single string")
    if max_new_tokens is not None and max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    device = device or next(model.parameters()).device
    prefixes = [torch.tensor([tokenizer.bos_id, *tokenizer.encode_question(q), tokenizer.sep_id],
                             dtype=torch.long) for q in questions]
    if not prefixes:
        return []
    context = model.config.context_length
    sizes = [len(x) for x in prefixes]
    if max(sizes) > context:
        raise ValueError("question with BOS/SEP exceeds model context")
    # The context has room for both the prefix and every emitted token (including EOS).
    budgets = [context - n for n in sizes]
    if max_new_tokens is not None:
        budgets = [min(n, max_new_tokens) for n in budgets]
    steps = max(budgets)
    budgets_gpu = torch.tensor(budgets, device=device)
    lengths = torch.tensor(sizes, device=device)
    ids = pad_sequence(prefixes, batch_first=True, padding_value=tokenizer.pad_id).to(device)
    valid = torch.arange(ids.shape[1], device=device)[None, :] < lengths[:, None]
    output = torch.full((len(prefixes), steps), tokenizer.pad_id, dtype=torch.long, device=device)
    active = budgets_gpu > 0
    original_mode = model.training
    model.eval()
    try:
        if steps:
            logits, cache = model.prefill(ids, lengths)
        for step in range(steps):
            tokens = model.allowed_ids[logits.argmax(dim=-1)]
            output[:, step] = torch.where(active, tokens, tokenizer.pad_id)
            active = active & (tokens != tokenizer.eos_id) & (step + 1 < budgets_gpu)
            if not active.any():
                break
            valid = torch.cat((valid, active[:, None]), dim=1)
            positions = (lengths + step).clamp(max=context - 1)
            logits, cache = model.decode_step(tokens, positions, cache, valid)
    finally:
        model.train(original_mode)

    results = []
    for row, size, budget in zip(output.tolist(), sizes, budgets):
        if tokenizer.eos_id in row:
            end = row.index(tokenizer.eos_id)
            reason = "eos"
        else:
            end = budget
            reason = "context" if size + budget == context else "max_new_tokens"
        results.append(Generation(tokenizer.decode_sql(row[:end]), reason))
    return results


def generate_sql(model, tokenizer, questions, *, device=None, max_new_tokens=None):
    return [r.sql for r in generate_results(model, tokenizer, questions, device=device,
                                             max_new_tokens=max_new_tokens)]
