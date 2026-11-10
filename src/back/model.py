"""Causal Transformer shared by back-model training and generation."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class TokenEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

    def forward(self, ids):
        return self.embedding(ids)


def _rotate(x, cos, sin):
    even, odd = x[..., ::2], x[..., 1::2]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, rope, past, valid_keys, cache):
        batch, length, dim = x.shape
        q, k, v = self.qkv_proj(x).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = _rotate(q, *rope), _rotate(k, *rope)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=valid_keys, is_causal=past is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        h = self.out_proj(attended.transpose(1, 2).reshape(batch, length, dim))
        return h, (k, v) if cache else None


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.eps)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.eps)
        self.attn = Attention(cfg)
        self.ffn = FeedForward(cfg)

    def forward(self, x, rope, past, valid_keys, cache):
        h, kv = self.attn(self.attn_norm(x), rope, past, valid_keys, cache)
        x = x + h
        return x + self.ffn(self.ffn_norm(x)), kv


class BackModel(nn.Module):
    def __init__(self, config: ModelConfig, sql_ids):
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.RMSNorm(config.d_model, eps=config.eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.embedding.embedding.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

        allowed = torch.tensor(list(sql_ids), dtype=torch.long)
        if (not allowed.numel() or allowed.min() < 0 or allowed.max() >= config.vocab_size
                or len(allowed.unique()) != len(allowed)):
            raise ValueError("invalid SQL token whitelist")
        self.register_buffer("allowed_ids", allowed, persistent=False)
        reverse = torch.full((config.vocab_size,), -1, dtype=torch.long)
        reverse[allowed] = torch.arange(len(allowed))
        self.register_buffer("_reverse_sql_ids", reverse, persistent=False)

        head_dim = config.d_model // config.num_heads
        inv_freq = 1.0 / (config.theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(config.context_length).float(), inv_freq)
        self.register_buffer("_cos", angles.cos(), persistent=False)
        self.register_buffer("_sin", angles.sin(), persistent=False)

    def sql_targets(self, targets):
        """Convert full token IDs to output classes, leaving ignored targets at -100."""
        ignored = targets == -100
        return self._reverse_sql_ids[targets.masked_fill(ignored, 0)].masked_fill(ignored, -100)

    def _hidden(self, ids, positions, *, past=None, valid_keys=None, cache=False):
        h = self.embedding(ids)
        # positions [1 or B, T]; share each angle across all heads and decoder layers.
        rope = (self._cos[positions].unsqueeze(1), self._sin[positions].unsqueeze(1))
        next_cache = []
        for layer, block in enumerate(self.blocks):
            kv = past[layer] if past is not None else None
            h, stored = block(h, rope, kv, valid_keys, cache)
            if cache:
                next_cache.append(stored)
        return self.final_norm(h), next_cache

    def _sql_logits(self, hidden):
        # Project only onto legal output rows; every generated token still uses its shared ID.
        return F.linear(hidden, self.lm_head.weight.index_select(0, self.allowed_ids))

    def forward(self, input_ids, *, compact=False):
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.context_length:
            raise ValueError("expected [batch, sequence] within model context")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        h, _ = self._hidden(input_ids, positions)
        logits = self._sql_logits(h)
        if compact:
            return logits
        full = logits.new_full((*logits.shape[:-1], self.config.vocab_size), float("-inf"))
        return full.index_copy(-1, self.allowed_ids, logits)

    def prefill(self, ids, lengths):
        """Right-padded question prefixes; return last real logits and per-layer KV."""
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        h, cache = self._hidden(ids, positions, cache=True)
        rows = torch.arange(ids.shape[0], device=ids.device)
        return self._sql_logits(h[rows, lengths - 1]), cache

    def decode_step(self, ids, positions, cache, valid_keys):
        """Append one output token. Ignore right-padding gaps in the cached prefixes."""
        h, cache = self._hidden(ids[:, None], positions[:, None], past=cache,
                                valid_keys=valid_keys[:, None, None, :], cache=True)
        return self._sql_logits(h[:, 0]), cache
