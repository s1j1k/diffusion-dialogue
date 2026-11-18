"""Transformer denoiser that predicts the clean embedding x0 from a noisy x_t."""
import math

import torch
import torch.nn as nn


def sinusoidal(t, dim, max_period=10_000):
    """Standard sinusoidal features of integer timesteps t, shape [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Denoiser(nn.Module):
    """
    x_t [B, L, emb_dim] and t [B]  ->  predicted x0 [B, L, emb_dim].

    The word embedding lives here too: it maps tokens into the space the
    diffusion runs in, and its transpose is the output head for rounding back
    to tokens, so both are trained together.
    """

    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(vocab_size, cfg.emb_dim)
        # Unit-variance embeddings put the signal on the same scale as the
        # unit-variance diffusion noise
        nn.init.normal_(self.embedding.weight, std=1.0)

        d = cfg.d_model
        self.in_proj = nn.Sequential(nn.Linear(cfg.emb_dim, d), nn.Tanh(), nn.Linear(d, d))
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, 4 * cfg.time_dim), nn.SiLU(), nn.Linear(4 * cfg.time_dim, d))
        self.position = nn.Embedding(cfg.seq_len, d)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, 4 * d, cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out_proj = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, cfg.emb_dim))

    def embed(self, ids):
        return self.embedding(ids)

    def logits(self, x):
        """Token scores for embeddings x: similarity to every vocabulary embedding."""
        return x @ self.embedding.weight.t()

    def nearest_tokens(self, x):
        """Index of the closest vocabulary embedding (L2) for every position."""
        w = self.embedding.weight
        dist = (x * x).sum(-1, keepdim=True) - 2 * x @ w.t() + (w * w).sum(-1)
        return dist.argmin(-1)

    def forward(self, x_t, t):
        L = x_t.shape[1]
        pos = torch.arange(L, device=x_t.device)
        h = self.in_proj(x_t) + self.position(pos)[None] + self.time_mlp(sinusoidal(t, self.cfg.time_dim))[:, None]
        h = self.encoder(self.drop(h))
        return self.out_proj(self.norm(h))
