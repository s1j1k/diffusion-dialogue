"""Which legacy defect breaks sampling? Patch one at a time, perfect denoiser."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legacy_oracle_check import load_legacy, token_accuracy
torch.manual_seed(0)
legacy = load_legacy()
V, L, B, n_cond, T = 30522, 64, 8, 24, 2000
table = torch.randn(V, 768) * 0.05
ids = torch.randint(1000, V, (B, L)); mask = torch.zeros(B, L, dtype=torch.bool); mask[:, n_cond:] = True
clean = table[ids]; m3 = mask[..., None].expand_as(clean)
betas = np.linspace(0.5 * 1e-4, 0.5 * 0.02, T)
def run(read_as_x0, clip):
    d = legacy.GaussianDiffusion(betas=betas)
    if read_as_x0:
        d._predict_xstart_from_eps = lambda x_t, t, eps: eps   # treat output as x0
    x = torch.where(mask[..., None], torch.randn_like(clean), clean); final = None
    for o in d.p_sample_loop_progressive(model=lambda x, t, **k: clean, shape=tuple(clean.shape), noise=x,
                                         device="cpu", clip_denoised=clip, mask=m3.long(), x_start=clean):
        final = o["sample"]
    return token_accuracy(final, table, ids, mask)
for rx in (False, True):
    for clip in (True, False):
        print(f"output read as x0={rx!s:5} clip_denoised={clip!s:5} -> token accuracy {run(rx, clip):.3f}")
