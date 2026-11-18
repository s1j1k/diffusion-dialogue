import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dialdiff import Config, Denoiser, TextDiffusion  # noqa: E402
from dialdiff.data import encode_pair  # noqa: E402
from dialdiff.schedule import alpha_bar_schedule  # noqa: E402

VOCAB = 50


def tiny_config(**kw):
    base = dict(seq_len=16, emb_dim=16, d_model=32, n_layers=2, n_heads=2, dropout=0.0,
                time_dim=16, num_steps=100)
    base.update(kw)
    return Config(**base)


def batch(B=4, L=16, n_cond=6):
    torch.manual_seed(0)
    ids = torch.randint(5, VOCAB, (B, L))
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, n_cond:] = True
    return ids, mask


@pytest.mark.parametrize("name", ["linear", "cosine", "sqrt"])
def test_schedules_decrease_from_one_to_zero(name):
    abar = alpha_bar_schedule(name, 1000)
    assert abar.shape == (1000,)
    assert torch.all(abar[1:] <= abar[:-1])
    assert 0.9 < abar[0] <= 1.0
    assert abar[-1] < 0.05


def test_partial_noising_leaves_source_untouched():
    cfg = tiny_config()
    ids, mask = batch()
    model = Denoiser(VOCAB, cfg)
    diff = TextDiffusion(cfg.num_steps)
    x0 = model.embed(ids)
    t = torch.full((ids.shape[0],), cfg.num_steps - 1)
    x_t = diff.q_sample(x0, t, mask)
    assert torch.equal(x_t[~mask], x0[~mask])
    assert not torch.allclose(x_t[mask], x0[mask])


@pytest.mark.parametrize("steps", [1, 10, 100])
def test_sampler_recovers_target_from_perfect_denoiser(steps):
    """
    With a network that always returns the true clean embedding, sampling must
    reproduce the target exactly, for any number of steps. The original code
    read the network output as a noise prediction and failed this.
    """
    cfg = tiny_config()
    ids, mask = batch()
    model = Denoiser(VOCAB, cfg)
    diff = TextDiffusion(cfg.num_steps)
    clean = model.embed(ids)

    out = diff.sample(model, ids, mask, steps=steps, x0_fn=lambda x, t: clean)
    assert torch.equal(out, ids)


def test_encode_pair_layout():
    class Tok:
        cls_token_id, sep_token_id, pad_token_id = 101, 102, 0

    ids, mask = encode_pair([1, 2, 3], [7, 8], Tok, seq_len=10)
    assert ids == [101, 1, 2, 3, 102, 7, 8, 102, 0, 0]
    assert mask == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    ids, mask = encode_pair(list(range(1, 20)), [7, 8], Tok, seq_len=10)
    assert len(ids) == 10 and ids[-1] == 102 and sum(mask) == 3


def test_training_reduces_loss_and_learns_a_tiny_mapping():
    """Overfit one batch: the response is a fixed function of the source."""
    torch.manual_seed(0)
    cfg = tiny_config(num_steps=50)
    B, L, n_cond = 8, 16, 8
    src = torch.randint(5, VOCAB, (B, n_cond))
    ids = torch.cat([src, (src + 1) % VOCAB], dim=1)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, n_cond:] = True

    model = Denoiser(VOCAB, cfg)
    diff = TextDiffusion(cfg.num_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(400):
        loss = diff.loss(model, ids, mask)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.5 * first

    model.eval()
    out = diff.sample(model, ids, mask, steps=20)
    accuracy = (out[mask] == ids[mask]).float().mean().item()
    assert accuracy > 0.9, accuracy
