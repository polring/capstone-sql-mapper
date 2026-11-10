"""Architecture recorded with each back-model checkpoint."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 256
    num_layers: int = 8
    num_heads: int = 4
    ffn_dim: int = 1024
    context_length: int = 512
    dropout: float = 0.0
    eps: float = 1e-6
    theta: float = 10000.0

    def __post_init__(self):
        sizes = (self.vocab_size, self.d_model, self.num_layers, self.num_heads,
                 self.ffn_dim, self.context_length)
        if any(type(n) is not int or n < 1 for n in sizes):
            raise ValueError("model sizes must be positive integers")
        if self.d_model % (2 * self.num_heads):
            raise ValueError("each attention head requires a whole, even number of channels")
        if not (0 <= self.dropout < 1 and isfinite(self.eps) and self.eps > 0
                and isfinite(self.theta) and self.theta > 0):
            raise ValueError("invalid dropout, normalization epsilon, or RoPE theta")
