# Benchmarking

## Our own model versions

`benchmarks/prompts.jsonl` is a fixed set of 12 prompts: the two prompts
shown in the 2024 report, six held-out test prompts, and four everyday
prompts written for the benchmark (no reference reply). Every model version is
run on all of them with the same seeds and sampling settings:

```sh
python scripts/benchmark.py --ckpt <checkpoint> --version vX.Y
python scripts/compare_benchmarks.py        # rebuilds benchmarks/README.md
```

Each result file records the checkpoint's SHA-256, the full model config,
the code commit (and whether the tree was dirty), library versions, seeds,
every response and the time per response. With the same code and environment
the outputs are reproducible exactly (checked for v0.1). Corpus-level metrics
on 200–1,000 test prompts come from `scripts/evaluate.py` and are copied into
the result file.

Metrics and their limits: BLEU against a single reference is a weak signal
for open-domain dialogue (many good replies share no words with the
reference), so we also track distinct-1/2 (diversity), response length and
the samples themselves. If a GPU is available, BERTScore and a small
LLM-as-judge rating (fluency, relevance) would be worth adding.

## Against current diffusion language models

Not run yet. This is how it could be done, so the comparison is fair and
repeatable.

### Candidates

| System | Type | Access | Notes |
| --- | --- | --- | --- |
| LLaDA-8B-Instruct (Nie et al., 2025) | Open weights, masked diffusion | Hugging Face; needs a GPU with ~16 GB+ (Colab A100/L4, or a hosted endpoint) | Closest open counterpart; supports chat |
| Dream-7B-Instruct | Open weights, masked diffusion (adapted from an AR model) | Hugging Face; similar GPU needs | |
| Inception Mercury | Commercial diffusion LLM | Hosted API (check current docs for access and pricing) | Fast; chat and code models |
| Gemini Diffusion (Google) | Commercial, experimental | Limited access at the time of writing | |
| Small masked diffusion (MDLM-style) trained by us | Same size and data as our model | Our code, once implemented | The like-for-like comparison for the report |
| Same-size autoregressive model trained by us | Baseline | Our code, once implemented | Shows what diffusion costs or gains at this scale |

### Protocol

1. **Same prompts.** Use `benchmarks/prompts.jsonl` plus the same 1,000 test
   prompts as `scripts/evaluate.py`.
2. **Prompt format.** Instruction-tuned chat models expect a chat template
   and a system instruction; our model sees the raw prompt. Use a fixed
   minimal instruction, e.g. "Reply to this Reddit comment in one short
   sentence.", record it, and keep it identical across systems.
3. **Decoding.** Record steps, temperature/remasking strategy and seeds; use
   each system's default plus a fixed low-step setting so speed and quality
   trade-offs are visible.
4. **Metrics.** BLEU, distinct-1/2, length, latency per response, and cost per
   1,000 responses for hosted APIs. For quality beyond BLEU, a blind
   pairwise judgement (human or LLM judge) on 100 prompts.
5. **Record results in the same format.** Add a script
   (`scripts/benchmark_external.py`) that calls a system and writes
   `benchmarks/results/<system>.json` with the same `outputs` structure, so
   `compare_benchmarks.py` shows them alongside our versions. Record model
   name and version string, API date, and parameters.
6. **Context.** These systems have 7–8B parameters and trillions of training
   tokens, versus our 7.5M parameters and 64k pairs. The point of the
   comparison is to show the gap and what scale buys, not a head-to-head
   win; the like-for-like comparisons are the same-size masked diffusion and
   autoregressive baselines.
