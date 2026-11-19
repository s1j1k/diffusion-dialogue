# Model versions

Every trained model that is worth keeping gets a version. Checkpoints are
too large for ordinary git history (~60 MB each), so they are stored as
**GitHub Release assets** (up to 2 GB per file, versioned by tag), while
everything needed to identify and rebuild a version is committed here.

| File | Purpose |
| --- | --- |
| `registry.json` | One entry per version: date, code commit, full config, data, rebuild command, checkpoint checksum and release tag |
| `../benchmarks/results/<version>.json` | That version's responses to the fixed benchmark prompts |
| `../docs/experiments/<run>.md` | Training curve, evaluations and samples of the run that produced it |

## Versions

| Version | Date | What it is | Checkpoint |
| --- | --- | --- | --- |
| v0 | 2024-05 | Original assignment model (DiffuSeq reproduction with the sampling bug) | Not kept; rebuildable from commit `ce8a1ff`; outputs recorded from the report |
| v0.1 | 2026-11-19 | First model after the rewrite: 7.5M parameters, trained on CPU | Release `model-v0.1`, asset `dialdiff-v0.1-cpu-small.pt` |

## Use a released version

```sh
gh release download model-v0.1 -R s1j1k/diffusion-dialogue -p "*.pt" -D runs/v0.1
sha256sum runs/v0.1/dialdiff-v0.1-cpu-small.pt     # compare with registry.json
python scripts/serve.py --ckpt runs/v0.1/dialdiff-v0.1-cpu-small.pt
```

Check out the version's `code_commit` if the current code has changed in a way
that no longer loads the checkpoint (the config is stored inside it, so this is
only needed after architecture changes).

## Rebuild a version from scratch

Each entry's `rebuild.command` checks out the exact training code and repeats
the run with the recorded seed and data. Bit-identical weights also need the
same PyTorch build and hardware (recorded in the benchmark results'
`environment`); otherwise expect the same curve and scores within noise.

## Add a new version

1. Train: `python scripts/train.py ... --out runs/<run>`; commit the code first
   and note the commit hash.
2. Evaluate and probe: `scripts/evaluate.py`, `scripts/denoise_probe.py`.
3. Record the trial: `python scripts/record_trial.py runs/<run> --title ...`.
4. Add an entry to `registry.json` (next version number, code commit, config,
   data, rebuild command, `sha256sum` of the checkpoint).
5. Benchmark: `python scripts/benchmark.py --ckpt runs/<run>/checkpoint.pt --version vX.Y`,
   then `python scripts/compare_benchmarks.py`.
6. Commit, then publish the checkpoint:
   `gh release create model-vX.Y runs/<run>/checkpoint.pt#dialdiff-vX.Y-<run>.pt --title "Model vX.Y" --notes "..."`
   (or upload under the asset name recorded in the registry).
