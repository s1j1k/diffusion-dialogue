"""Train the dialogue diffusion model.

    python scripts/train.py --data datasets/CommonsenseConversation --out runs/base
    python scripts/train.py --preset cpu-small --train-limit 20000 --out runs/cpu   # quick CPU run
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dialdiff import Config, Denoiser, TextDiffusion  # noqa: E402
from dialdiff.data import DialogueDataset, load_pairs  # noqa: E402

# Softmax over a large vocabulary underflows to denormal floats, which make
# CPU matrix multiplies up to ~100x slower; flushing them to zero is harmless
torch.set_flush_denormal(True)

PRESETS = {
    # Fits a laptop CPU: short sequences, small transformer
    "cpu-small": dict(seq_len=64, d_model=256, n_layers=4, n_heads=4, batch_size=32,
                      train_steps=6000, warmup_steps=300, ema_decay=0.999, lr=3e-4, nll_fraction=0.25),
    # Single T4/A100 GPU (e.g. Colab)
    "gpu-base": dict(),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/CommonsenseConversation")
    p.add_argument("--out", required=True)
    p.add_argument("--preset", choices=PRESETS, default="gpu-base")
    p.add_argument("--train-limit", type=int)
    p.add_argument("--valid-limit", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for f in Config.__dataclass_fields__.values():
        p.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=None)
    return p.parse_args()


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    frac = (step - cfg.warmup_steps) / max(1, cfg.train_steps - cfg.warmup_steps)
    return cfg.lr * max(0.0, 1.0 - frac)


def batches(loader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model, diffusion, loader, device, max_batches=20):
    model.eval()
    totals, n = {}, 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        out = diffusion.loss(model, b["ids"].to(device), b["mask"].to(device))
        for k, v in out.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
    model.train()
    return {k: v / n for k, v in totals.items()}


def main():
    args = parse_args()
    cfg = Config(**PRESETS[args.preset])
    for f in Config.__dataclass_fields__:
        v = getattr(args, f)
        if v is not None:
            setattr(cfg, f, v)
    torch.manual_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    train = DialogueDataset(load_pairs(os.path.join(args.data, "train.jsonl"), args.train_limit), tok, cfg.seq_len)
    valid = DialogueDataset(load_pairs(os.path.join(args.data, "valid.jsonl"), args.valid_limit), tok, cfg.seq_len)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid, batch_size=cfg.batch_size)

    model = Denoiser(len(tok), cfg).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = TextDiffusion(cfg.num_steps, cfg.schedule, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.1f}M parameters, {len(train)} training pairs, device {device}")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**cfg.to_dict(), "train_pairs": len(train), "params": n_params}, f, indent=2)

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    data = batches(train_loader)
    start = time.time()
    for step in range(cfg.train_steps):
        b = next(data)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        out = diffusion.loss(model, b["ids"].to(device), b["mask"].to(device), cfg.nll_fraction)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1.0 - cfg.ema_decay)

        if step % 50 == 0 or step == cfg.train_steps - 1:
            rec = {"step": step, "lr": lr_at(step, cfg), "grad_norm": grad_norm.item(),
                   **{k: v.item() for k, v in out.items()}, "elapsed_s": round(time.time() - start, 1)}
            if step % args.eval_every == 0 or step == cfg.train_steps - 1:
                rec.update({"valid_" + k: v for k, v in evaluate(ema, diffusion, valid_loader, device).items()})
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if not math.isfinite(rec["loss"]):
                raise SystemExit("loss is not finite")

    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "ema": ema.state_dict(),
                "tokenizer": cfg.tokenizer}, os.path.join(args.out, "checkpoint.pt"))
    print("saved", os.path.join(args.out, "checkpoint.pt"))


if __name__ == "__main__":
    main()
