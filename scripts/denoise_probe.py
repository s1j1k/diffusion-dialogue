"""Token accuracy of one denoising step at fixed noise levels.

Noises real responses to step t, asks the model for x0, and checks how many
response words the nearest embedding recovers. This separates "the network
has not learned" from "the sampler is going wrong": a model can score well
here and still generate poorly. Saves probe.json next to the checkpoint.

    python scripts/denoise_probe.py --ckpt runs/cpu-small/checkpoint.pt
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dialdiff import Config, Denoiser, TextDiffusion  # noqa: E402
from dialdiff.data import DialogueDataset, load_pairs  # noqa: E402

torch.set_flush_denormal(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="datasets/CommonsenseConversation")
    p.add_argument("--limit", type=int, default=64)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = Config.from_dict(ckpt["config"])
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    model = Denoiser(len(tok), cfg)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = TextDiffusion(cfg.num_steps, cfg.schedule)
    levels = sorted({0, cfg.num_steps // 40, cfg.num_steps // 10, cfg.num_steps // 4,
                     cfg.num_steps // 2, 3 * cfg.num_steps // 4, cfg.num_steps - 1})

    result = {"levels": levels}
    for split in ("train", "test"):
        ds = DialogueDataset(load_pairs(os.path.join(args.data, f"{split}.jsonl"), args.limit), tok, cfg.seq_len)
        words = ds.mask & (ds.ids != tok.pad_token_id)
        torch.manual_seed(0)
        accs = []
        with torch.no_grad():
            x0 = model.embed(ds.ids)
            for t in levels:
                tv = torch.full((len(ds.ids),), t)
                pred = model.nearest_tokens(model(diffusion.q_sample(x0, tv, ds.mask), tv))
                accs.append(round((pred[words] == ds.ids[words]).float().mean().item(), 4))
        result[split] = accs

    with torch.no_grad():
        norms = model.embedding.weight.norm(dim=1)
    result["closest_to_origin"] = tok.convert_ids_to_tokens(norms.argsort()[:8].tolist())

    with open(os.path.join(os.path.dirname(args.ckpt), "probe.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("t:    ", levels)
    print("train:", result["train"])
    print("test: ", result["test"])
    print("tokens nearest the origin:", result["closest_to_origin"])


if __name__ == "__main__":
    main()
