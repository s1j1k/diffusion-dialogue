# Research log

Dated record of work since the 2024 report, for the next report. Newest
entries at the bottom. Settings and scores for every run are in
`runs/<name>/config.json` and `runs/<name>/metrics_*.json` (not committed).

## Background (already reported, May 2024)

Group assignment, AI for Engineers: a simplified reproduction of DiffuSeq
on 1,000 Commonsense Conversation pairs, 4,000 steps on one A100 (university
HPC). Test BLEU 0.00046 on 4 prompts. See the 2024 report for the background
and method; nothing below repeats it.

## 2026-11-18: diagnosis, rewrite, first CPU run

**Code audit.** Compared the 2024 code with the public DiffuSeq repository
line by line: about half of the non-comment lines appeared verbatim
(`diffusion_model.py` 75 %, `transformer_model.py` 59 %). This was expected
for a reproduction task, but it meant the code was not a distinct piece of
work, so we rewrote it rather than patching it.

**Root cause of the failed results.** Training read the network output as
the clean embedding x0; sampling read it as noise ε. With a perfect
denoiser substituted for the network, the original sampler reconstructs 0 %
of tokens; changing only that interpretation gives 100 %. Six secondary
defects were also found (timestep scaling mismatch, learning rate, embedding
scale, evaluation on 4 prompts with special tokens in references, and
others). Full analysis: `docs/TECHNICAL_NOTES.md`.

**Rewrite (`dialdiff/`).** Fresh implementation built around one
parameterisation (x0). Changes from 2024: learned 128-d unit-variance
embeddings instead of BERT's 768-d ones (BERT's are ~20x smaller than
the noise: per-dimension std 0.043); a plain PyTorch transformer denoiser
sized by config; sqrt noise schedule; DDIM-style sampler for an x0 model so
evaluation can use 100–200 steps instead of 2,000; proper evaluation
(1,000+ prompts, corpus and sentence BLEU, distinct-n, EMA weights). Unit
tests include the perfect-denoiser check, which would have caught the 2024
bug immediately.

**Data.** Full Commonsense Conversation set from the DiffuSeq release:
3,382,137 train / 2,047 valid / 9,999 test pairs.

**Compute.** No GPU available locally (4-thread CPU). Plan: small sanity runs
on CPU, full-size training on a Colab GPU.

**CPU performance bug (engineering).** The first CPU run took ~100 s per
training step. Profiling showed 91 % of the time in matrix multiplies on
denormal floats: with unit-variance 128-d embeddings the token logits are
far apart, so most softmax probabilities underflow below 1e-38, and the
backward pass of the vocabulary projection multiplies matrices full of them.
On this CPU a 2048×2048 matmul takes 0.9 s normally and 144 s with denormal
inputs. `torch.set_flush_denormal(True)` brought a step to 4.3 s, and
computing the rounding NLL on a random 25 % of positions (unbiased) to 2.3 s,
about 46× faster overall. The test suite went from 90 s to 29 s.

Observation: the rounding NLL is ~0 from the first step, because each
unit-variance embedding is already far closer to itself than to any other
token. It still matters: it is the term that stops the embeddings collapsing
towards zero, which would trivially shrink the MSE.

**Run `cpu-small`.** Preset `cpu-small` (seq_len 64, d_model 256, 4 layers,
7.5M parameters), first 100k training pairs, 2,000 steps (≈64k samples
seen), batch 32, lr 3e-4, warm-up 200, rounding NLL on 25 % of positions,
T = 2,000, sqrt schedule. A sanity check that the fixed pipeline learns, not
a tuned result. Recorded as model version **v0.1** (`models/registry.json`)
and trial `docs/experiments/cpu-small.md`.

Training loss 1.075 → 0.265, validation 1.079 → 0.328 (still falling).

## 2026-11-19: evaluating v0.1, the "traveler" effect, benchmarking

**Denoising probe.** Share of response words recovered in one step from
noise level t, on unseen test prompts: 0.85 (t = 0), 0.74 (200), 0.55
(1000), 0.40 (1500), 0.01 (1999). Training prompts score the same, so the
network has learned to denoise and generalises; it fails only from pure
noise, where it predicts a near-average embedding.

**Generation, 200 test prompts, 50 steps:**

| Sampling | Sentence BLEU | Corpus BLEU | Distinct-2 | Output |
| --- | --- | --- | --- | --- |
| Clamp every step | 0.0029 | 0.0003 | 0.26 | "traveler enclosure walker traveler…" |
| Clamp t ≤ 1000 | 0.0059 | 0.0002 | 0.59 | mixed |
| Clamp t ≤ 500 | 0.0062 | 0.0001 | 0.67 | mixed |
| No clamp | **0.0082** | 0.0002 | 0.61 | common words in no order ("did be like will they…") |

**Why "traveler".** The repeated nonsense words are the tokens whose random
initial embeddings lie closest to the origin (`traveler` has the shortest
vector: norm 8.5 vs a mean of 11.2). Rare words never occur in 64k training
replies, so their embeddings stay at their random starting points. Under
uncertainty the MSE-optimal prediction is the mean of plausible embeddings,
near the origin; rounding it picks whichever rare word drew the shortest
vector, and clamping at every step locks that in from the first step. With
another seed a different word would win. Remedies to try: clamp only late
(done, helps), equal-norm embeddings, cosine rounding, more training.

**Inference server.** `scripts/serve.py` loads a checkpoint once and serves
`POST /generate`. Example: "I let the waffles get mouldy last night. Did you
eat one?" → "right couple willving do theyaus like the be it was morning are
they a" (50 steps, ~0.5 s per response on CPU).

**Version control and benchmarking.** Model versions are registered in
`models/registry.json` (code commit, config, data, checksum, rebuild
command), checkpoints are attached to GitHub Releases, and every version
answers the fixed prompt set in `benchmarks/` with fixed seeds (re-running
v0.1 reproduced its outputs exactly). v0 (2024) is recorded from the report.
How to extend this to current diffusion products: `docs/BENCHMARKING.md`.

## Training budget compared with DiffuSeq

"Samples seen" = training steps × batch size. DiffuSeq settings are from its
published training script for this dataset (`scripts/train.sh`, README).

| | DiffuSeq (paper) | Our 2024 run | `cpu-small` (2026) |
| --- | --- | --- | --- |
| Hardware | 7 GPUs | 1 A100, 50 min job | 4-thread laptop CPU |
| Unique training pairs | 3,382,137 | 1,000 | 100,000 |
| Steps × batch | 140,000 × 2,048 | 4,000 × 32 | 2,000 × 32 |
| Samples seen | 286.7M | 128k (0.045 %) | 64k (0.02 %) |
| Passes over its data | ~85 | 128 (memorising 1,000 pairs) | 0.64 |
| Model | ~91M (BERT-base-sized), 128-d embeddings | BERT-base encoder, 768-d BERT embeddings | 7.5M, 128-d embeddings |
| Decoding | MBR over 10 samples (10 seeds) | 1 sample | 1 sample |

Two separate problems held back the 2024 result: the sampler bug (which no
amount of training could fix), and a training budget about 2,000× smaller
than DiffuSeq's, on a dataset too small to learn general dialogue from.

## Next

- Full-size run on Colab (default config, 24M parameters, ≥50k steps).
- Steps-vs-BLEU curve (10 / 50 / 200 / 2,000 sampling steps).
- Comparison with newer methods: masked diffusion (MDLM-style), self-
  conditioning, same-size autoregressive baseline. Notes in
  `docs/LITERATURE.md`.
