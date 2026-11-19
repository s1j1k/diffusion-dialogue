"""Run a model version over the fixed benchmark prompts and save the outputs.

    python scripts/benchmark.py --ckpt runs/cpu-small/checkpoint.pt --version v0.1

Writes benchmarks/results/<version>.json with everything needed to compare
versions later: checkpoint checksum, model config, code commit, sampling
settings, seeds, the responses to every prompt, and any test-set metrics found
next to the checkpoint. Then rebuild benchmarks/README.md with
scripts/compare_benchmarks.py.

Seeds are fixed, so re-running a version on the same code reproduces its
outputs exactly (on the same hardware and library versions).
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serve import Generator  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROMPTS = os.path.join(ROOT, "benchmarks", "prompts.jsonl")
RESULTS = os.path.join(ROOT, "benchmarks", "results")

# Sampling settings run for every version. The first is the headline setting.
SETTINGS = [
    {"label": "50 steps, no clamp", "steps": 50, "clamp": False},
    {"label": "50 steps, clamp t<=500", "steps": 50, "clamp": True, "clamp_max_t": 500},
    {"label": "200 steps, no clamp", "steps": 200, "clamp": False},
]
SEEDS = [0, 1, 2]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--version", required=True, help="model version, e.g. v0.1 (see models/registry.json)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.set_flush_denormal(True)
    gen = Generator(args.ckpt, torch.device(args.device))
    prompts = [json.loads(line) for line in open(PROMPTS, encoding="utf-8")]

    outputs = []
    for s in SETTINGS:
        for pr in prompts:
            responses, start = [], time.time()
            for seed in SEEDS:
                responses += gen.generate(pr["prompt"], steps=s["steps"], samples=1, clamp=s["clamp"],
                                          clamp_max_t=s.get("clamp_max_t"), seed=seed)
            outputs.append({"id": pr["id"], "setting": s["label"], "responses": responses,
                            "seconds_per_response": round((time.time() - start) / len(SEEDS), 3)})
        print("done:", s["label"], flush=True)

    test_metrics = []
    for mpath in sorted(glob.glob(os.path.join(os.path.dirname(args.ckpt), "metrics_*.json"))):
        with open(mpath) as f:
            test_metrics.append(json.load(f))

    result = {
        "version": args.version,
        "benchmarked_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": {"path": args.ckpt, "sha256": sha256(args.ckpt), "bytes": os.path.getsize(args.ckpt)},
        "config": gen.cfg.to_dict(),
        "params": sum(p.numel() for p in gen.model.parameters()),
        "weights": "ema",
        "benchmark_code": {"commit": git("rev-parse", "--short", "HEAD"),
                           "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": args.device, "platform": platform.platform()},
        "settings": SETTINGS,
        "seeds": SEEDS,
        "outputs": outputs,
        "test_metrics": test_metrics,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"{args.version}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
