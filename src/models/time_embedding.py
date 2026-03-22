import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def get_time_embedder(embed_dim: int, time_dim_mult: int = 2) -> nn.Sequential:
    """Sinusoidal -> Linear -> GELU -> Linear."""
    time_dim = embed_dim * time_dim_mult
    return nn.Sequential(
        SinusoidalPosEmb(embed_dim),
        nn.Linear(embed_dim, time_dim),
        nn.GELU(),
        nn.Linear(time_dim, time_dim),
    )
