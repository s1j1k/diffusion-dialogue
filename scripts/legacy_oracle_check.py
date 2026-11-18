"""Measure the original (2024) sampler against a perfect denoiser.

Loads src/diffusion_model.py from the last commit of the original code
(git history) and runs its sampling loop with a "network" that always returns
the true clean embedding. A correct sampler must then reproduce the target.
The same check is run on the rewritten sampler (dialdiff) for comparison.

    python scripts/legacy_oracle_check.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from dialdiff import Config, Denoiser, TextDiffusion  # noqa: E402

LEGACY_COMMIT = "ce8a1ff"


def load_legacy():
    src = subprocess.check_output(
        ["git", "show", f"{LEGACY_COMMIT}:src/diffusion_model.py"], cwd=ROOT).decode()
    path = os.path.join(tempfile.mkdtemp(), "legacy_diffusion.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("legacy_diffusion", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def token_accuracy(x, emb_table, ids, mask):
    """Decode by highest dot product with the embedding table, as the legacy lm_head did."""
    pred = (x @ emb_table.t()).argmax(-1)
    return (pred[mask] == ids[mask]).float().mean().item()


def main():
    torch.manual_seed(0)
    legacy = load_legacy()
    V, L, B, n_cond = 30522, 64, 8, 24
    T = 2000

    # Legacy setup: 768-d BERT-like embeddings (per-dimension std ~0.05)
    for name, std, dim in [("BERT-scale embeddings (as in 2024)", 0.05, 768),
                           ("unit-variance embeddings", 1.0, 768)]:
        table = torch.randn(V, dim) * std
        ids = torch.randint(1000, V, (B, L))
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, n_cond:] = True
        clean = table[ids]

        scale = 1000 / T
        betas = np.linspace(scale * 0.0001, scale * 0.02, T, dtype=np.float64)
        diff = legacy.GaussianDiffusion(betas=betas)
        x_noised = torch.where(mask[..., None], torch.randn_like(clean), clean)
        m3 = mask[..., None].expand_as(clean)

        def oracle(x, t, **kw):
            return clean

        # The legacy p_sample_loop keeps every intermediate sample; keep only the last
        final = None
        for out in diff.p_sample_loop_progressive(model=oracle, shape=tuple(clean.shape), noise=x_noised,
                                                  device="cpu", progress=False, clamp_step=None,
                                                  mask=m3.long(), x_start=clean):
            final = out["sample"]
        acc = token_accuracy(final, table, ids, mask)
        print(f"legacy sampler, {name}: token accuracy {acc:.3f}")

    # Rewritten sampler, same check
    cfg = Config(seq_len=L, emb_dim=128, d_model=64, n_layers=1, n_heads=2, num_steps=T)
    model = Denoiser(V, cfg)
    ids = torch.randint(1000, V, (B, L))
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, n_cond:] = True
    clean = model.embed(ids).detach()
    out = TextDiffusion(T, "sqrt").sample(model, ids, mask, steps=200, x0_fn=lambda x, t: clean)
    print(f"rewritten sampler (200 steps): token accuracy {(out[mask] == ids[mask]).float().mean().item():.3f}")


if __name__ == "__main__":
    main()
