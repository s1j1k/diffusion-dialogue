"""Noise schedules.

A schedule is described by alpha_bar[t], the fraction of the clean signal's
variance left after t+1 forward steps (t = 0 .. T-1). Everything else the
diffusion needs is derived from it.
"""
import math

import torch


def _from_alpha_bar_fn(fn, num_steps, max_beta=0.999):
    """Discretise a continuous alpha_bar(s), s in [0, 1], into num_steps steps."""
    betas = []
    for i in range(num_steps):
        s0, s1 = i / num_steps, (i + 1) / num_steps
        betas.append(min(1.0 - fn(s1) / fn(s0), max_beta))
    return torch.cumprod(1.0 - torch.tensor(betas, dtype=torch.float64), dim=0)


def alpha_bar_schedule(name, num_steps):
    """Return alpha_bar as a float64 tensor of shape [num_steps]."""
    if name == "linear":
        # Ho et al. (2020), rescaled so the total noise is independent of num_steps
        scale = 1000.0 / num_steps
        betas = torch.linspace(scale * 1e-4, scale * 0.02, num_steps, dtype=torch.float64)
        return torch.cumprod(1.0 - betas, dim=0)
    if name == "cosine":
        # Nichol & Dhariwal (2021)
        return _from_alpha_bar_fn(
            lambda s: math.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2, num_steps)
    if name == "sqrt":
        # Li et al. (2022), Diffusion-LM: adds noise quickly at the start, which suits
        # word embeddings because small noise barely changes the nearest token
        return _from_alpha_bar_fn(lambda s: 1.0 - math.sqrt(s + 1e-4), num_steps)
    raise ValueError(f"unknown schedule {name!r}")
