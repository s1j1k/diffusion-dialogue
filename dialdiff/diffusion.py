"""Continuous diffusion over word embeddings, conditioned by partial noising.

Notation: x0 is the clean embedding sequence, x_t the noised one after step t,
abar[t] the remaining signal variance (see schedule.py). Only positions where
mask == 1 (the response) are ever noised; the source stays clean and acts as
the condition, so no separate classifier or encoder is needed.

The network is always read as a prediction of x0. Training compares it with
x0, and sampling uses it as x0. Mixing this up with a noise (epsilon)
prediction is the bug documented in docs/TECHNICAL_NOTES.md.
"""
import torch
import torch.nn.functional as F

from .schedule import alpha_bar_schedule


class TextDiffusion:
    def __init__(self, num_steps, schedule="sqrt", device="cpu"):
        abar = alpha_bar_schedule(schedule, num_steps)
        self.num_steps = num_steps
        self.sqrt_abar = abar.sqrt().float().to(device)
        self.sqrt_1m_abar = (1.0 - abar).sqrt().float().to(device)

    def _at(self, table, t):
        return table[t][:, None, None]

    def q_sample(self, x0, t, mask, noise=None):
        """Noise the masked positions of x0 to step t; leave the rest untouched."""
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self._at(self.sqrt_abar, t) * x0 + self._at(self.sqrt_1m_abar, t) * noise
        return torch.where(mask[..., None], x_t, x0)

    def loss(self, model, ids, mask, nll_fraction=1.0):
        """
        Returns a dict of per-batch scalars; "loss" is what gets optimised.

        mse:   ||x0_hat - x0||^2 on the response positions (x0 at t = 0 is the
               clean embedding itself, which anchors the embedding scale)
        nll:   -log p(token | x0), keeps embeddings decodable ("rounding" loss)
        prior: ||sqrt(abar_T) x0||^2, keeps the last step close to pure noise
        """
        B = ids.shape[0]
        emb = model.embed(ids)
        t = torch.randint(0, self.num_steps, (B,), device=ids.device)

        # x0 is a slightly noisy copy of the embedding (step "-1" of the chain)
        x0 = emb + self.sqrt_1m_abar[0] * torch.randn_like(emb)
        x_t = self.q_sample(x0, t, mask)
        x0_hat = model(x_t, t)

        m = mask[..., None].float()
        n = m.sum((1, 2)).clamp(min=1) * emb.shape[-1]
        target = torch.where((t == 0)[:, None, None], emb, x0)
        mse = (((x0_hat - target) ** 2) * m).sum((1, 2)) / n

        # The vocabulary projection is the most expensive op in the step; an
        # unbiased estimate from a random subset of positions is enough
        if nll_fraction < 1.0:
            k = max(1, int(round(ids.shape[1] * nll_fraction)))
            pos = torch.rand(ids.shape, device=ids.device).argsort(dim=1)[:, :k]
            x0_sub = x0.gather(1, pos[..., None].expand(-1, -1, x0.shape[-1]))
            nll = F.cross_entropy(model.logits(x0_sub).transpose(1, 2), ids.gather(1, pos), reduction="none").mean(1)
        else:
            nll = F.cross_entropy(model.logits(x0).transpose(1, 2), ids, reduction="none").mean(1)
        prior = (((self.sqrt_abar[-1] * x0) ** 2) * m).sum((1, 2)) / n

        loss = mse + nll + prior
        return {"loss": loss.mean(), "mse": mse.mean(), "nll": nll.mean(), "prior": prior.mean()}

    @torch.no_grad()
    def sample(self, model, ids, mask, steps=None, clamp=True, clamp_max_t=None, x0_fn=None):
        """
        Generate the masked positions of ids. Returns token ids [B, L].

        Uses the deterministic DDIM update written for an x0 prediction:
            eps_hat = (x_t - sqrt(abar_t) x0_hat) / sqrt(1 - abar_t)
            x_s     = sqrt(abar_s) x0_hat + sqrt(1 - abar_s) eps_hat
        so any number of steps <= num_steps can be used.

        clamp: snap x0_hat to the nearest word embedding (the "clamping
               trick" of Diffusion-LM / DiffuSeq).
        clamp_max_t: only clamp at steps t <= clamp_max_t. At high noise the
               prediction is close to the mean embedding, and snapping it
               locks in whichever tokens lie nearest the origin.
        x0_fn: optional replacement for the network, used by the tests.
        """
        steps = steps or self.num_steps
        emb = model.embed(ids)
        m = mask[..., None]
        x = torch.where(m, torch.randn_like(emb), emb)

        ts = torch.linspace(self.num_steps - 1, 0, steps).round().long().unique(sorted=True).flip(0)
        ts = ts.tolist()
        for i, t in enumerate(ts):
            t_vec = torch.full((ids.shape[0],), t, device=ids.device, dtype=torch.long)
            x0_hat = x0_fn(x, t_vec) if x0_fn is not None else model(x, t_vec)
            if clamp and (clamp_max_t is None or t <= clamp_max_t):
                x0_hat = model.embed(model.nearest_tokens(x0_hat))
            x0_hat = torch.where(m, x0_hat, emb)
            if i == len(ts) - 1:
                x = x0_hat
                break
            s = ts[i + 1]
            eps_hat = (x - self.sqrt_abar[t] * x0_hat) / self.sqrt_1m_abar[t]
            x = self.sqrt_abar[s] * x0_hat + self.sqrt_1m_abar[s] * eps_hat
            x = torch.where(m, x, emb)

        return torch.where(mask, model.nearest_tokens(x), ids)
