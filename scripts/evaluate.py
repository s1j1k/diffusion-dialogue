"""Generate responses for a test set and score them.

    python scripts/evaluate.py --ckpt runs/base/checkpoint.pt --steps 200
"""
import argparse
import json
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
from dialdiff.metrics import decode_response, score  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="datasets/CommonsenseConversation")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--steps", type=int, default=200, help="sampling steps (<= num_steps)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-clamp", action="store_true")
    p.add_argument("--clamp-max-t", type=int, help="only snap to word embeddings at steps t <= this")
    p.add_argument("--raw-weights", action="store_true", help="use raw instead of EMA weights")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = Config.from_dict(ckpt["config"])
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    model = Denoiser(len(tok), cfg).to(device)
    model.load_state_dict(ckpt["model" if args.raw_weights else "ema"])
    model.eval()
    diffusion = TextDiffusion(cfg.num_steps, cfg.schedule, device)

    pairs = load_pairs(os.path.join(args.data, f"{args.split}.jsonl"), args.limit)
    loader = DataLoader(DialogueDataset(pairs, tok, cfg.seq_len), batch_size=args.batch_size)

    hyps, refs, start = [], [], time.time()
    for b in loader:
        ids, mask = b["ids"].to(device), b["mask"].to(device)
        out = diffusion.sample(model, ids, mask, steps=args.steps, clamp=not args.no_clamp,
                               clamp_max_t=args.clamp_max_t)
        for o, r, m in zip(out, ids, mask):
            hyps.append(decode_response(o, m, tok))
            refs.append(decode_response(r, m, tok))
    elapsed = time.time() - start

    metrics = {**score(hyps, refs), "steps": args.steps, "clamp": not args.no_clamp, "clamp_max_t": args.clamp_max_t,
               "weights": "raw" if args.raw_weights else "ema", "seconds": round(elapsed, 1)}
    run_dir = os.path.dirname(args.ckpt)
    clamp_tag = "noclamp" if args.no_clamp else (f"clamp{args.clamp_max_t}" if args.clamp_max_t is not None else "clamp")
    tag = f"{args.split}_steps{args.steps}_{clamp_tag}"
    with open(os.path.join(run_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(run_dir, f"samples_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for (src, _), h, r in zip(pairs, hyps, refs):
            f.write(json.dumps({"source": src, "reference": r, "generated": h}) + "\n")

    print(json.dumps(metrics, indent=2))
    for (src, _), h, r in list(zip(pairs, hyps, refs))[:5]:
        print(f"\nSOURCE: {src}\nREF:    {r}\nGEN:    {h}")


if __name__ == "__main__":
    main()
