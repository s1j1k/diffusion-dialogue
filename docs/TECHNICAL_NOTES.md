# Technical notes: why the 2024 model failed, and the rewrite

These notes cover the original model's near-zero BLEU score, and how the
rewritten code differs. Line references are to the original code at commit
`ce8a1ff` (`src/`), which is kept in the git history.

## 1. The original result

The 2024 report (AI for Engineers, Assignment 3) trained a DiffuSeq-style
model on 1,000 Commonsense Conversation pairs for 4,000 steps and evaluated it
on 4 test prompts:

| Metric | Reported |
| --- | --- |
| Average sentence BLEU (test) | 0.00046 |
| Average loss (test) | 17.19 |

The generated responses were unrelated words with no grammar ("survivor
instruments shot screens proper dirt 1970 ..."). The report put this down to
the small dataset, short training, pretrained BERT embeddings and the output
dimension. Those were real limits, but they were not the cause.

## 2. Root cause: training and sampling read the network output differently

A diffusion denoiser can be trained to predict either the clean data `x0` or
the added noise `ε`; the two are interchangeable through

    x_t = sqrt(ᾱ_t)·x0 + sqrt(1 − ᾱ_t)·ε   ⇔   x0 = (x_t − sqrt(1 − ᾱ_t)·ε) / sqrt(ᾱ_t)

but every part of the code must agree on which one the network outputs.

**Training treated the output as `x0`.** `training_losses_seq2seq` sets
`target = x_start` (line 448) and minimises `‖target − model_output‖²`
(line 450). The network therefore learns to return the clean embedding.

**Sampling treated the same output as `ε`.** `p_mean_variance` passes the
network output to `_predict_xstart_from_eps` (line 248):

    pred_xstart = sqrt(1/ᾱ_t)·x_t − sqrt(1/ᾱ_t − 1)·model_output

When `model_output` is actually an `x0` estimate, this "reconstruction" is a
large multiple of the noisy input minus a multiple of the clean estimate.
At most timesteps `sqrt(1/ᾱ_t)` is huge (ᾱ_t → 0 near t = T), so the noise
term dominates and the estimate carries almost no information about the
target. The posterior mean built from it then drives each reverse step. The
same conversion is used inside the training loss for the `t = 0` anchor term
and the token NLL (`_x0_helper`, lines 424 and 452), so those terms were
also computed from a meaningless `x0`.

### Evidence: a perfect denoiser

To separate the sampler from the model, the original sampling loop was run
with a fake "network" that always returns the *true* clean embedding, the
best output any trained model could give. A correct sampler must then
reproduce the target tokens exactly. (`scripts/legacy_oracle_check.py` loads
the original file from git and runs this; `scripts/legacy_ablation.py`
patches one defect at a time.) 2,000 steps, 8 sequences of 64 tokens, 40
response tokens each, 30,522-word vocabulary:

| Sampler | Output read as | `clip_denoised` | Token accuracy |
| --- | --- | --- | --- |
| Original | ε (as written) | on (as written) | **0.000** |
| Original | ε | off | 0.000 |
| Original, one-line fix | x0 | on | 1.000 |
| Original, one-line fix | x0 | off | 1.000 |
| Rewrite (`dialdiff`, 200 steps) | x0 | n/a | 1.000 |

With the original sampler, **even a perfect model produces 0 % correct
tokens**, so no amount of data or training could have raised the BLEU score.
Interpreting the output as `x0` alone restores perfect reconstruction; the
`[-1, 1]` clamp makes no difference here. This mismatch is the cause of the
reported failure.

## 3. Other defects found

These did not cause the failure on their own, but each would have limited a
working model.

| # | Defect | Where | Effect | Rewrite |
| --- | --- | --- | --- | --- |
| 2 | Timestep scaling differs: training passes `t·1000/T` (= `0.5t` for T = 2000) to the network, sampling passes raw `t` | lines 449 vs 245 | At test time the model is queried at timesteps it never saw | Integer `t` everywhere |
| 3 | Learning rate 1e-3 while fine-tuning a pretrained BERT encoder | `run_train.py` line 58 | Ten times higher than DiffuSeq's 1e-4; prone to destroying the pretrained weights | 1e-4 with warm-up and linear decay (3e-4 for the small CPU model trained from scratch) |
| 4 | Diffusion in BERT's 768-d embedding space, whose values are far smaller than the unit-variance noise (per-dimension std 0.043; mean vector norm 1.4 vs about 27.7 for the noise) | `transformer_model.py` | The signal is about 20x smaller than the noise, so it is buried for most of the chain; DiffuSeq instead learns a 128-d embedding | Learned 128-d embedding initialised at unit variance |
| 5 | Dataset embedded inputs with a separate frozen copy of BERT's embeddings, which the loss then ignored | `data_utils.py` `TextDataset` | Wasted work; confusing | Dataset returns token ids only |
| 6 | Evaluation on 4 prompts; references still contained `[CLS]`/`[SEP]`; loss averaged over padding; raw weights saved, EMA weights never used | `run_test.py` lines 107, 148, 160 | BLEU of 4 sentences is noise; special tokens in references depress BLEU further | 1,000+ test prompts, responses cut at the first `[SEP]`, corpus and sentence BLEU, distinct-n, EMA weights |
| 7 | Committed `training_config.json` came from a 100-step debug run, not the reported 4,000-step run | `src/training_config.json` | Results not reproducible from the repository | Config stored inside each checkpoint |

## 4. The rewrite

The new code (`dialdiff/`) is a fresh implementation of the same idea:
continuous diffusion over word embeddings, conditioned by partial noising
(Gong et al., 2023). It is written around a single parameterisation, so
the class of bug in Section 2 cannot recur.

| | 2024 code | Rewrite |
| --- | --- | --- |
| Network output | Ambiguous (x0 in training, ε in sampling) | Always x0 |
| Embedding | Pretrained BERT word embeddings, 768-d | Learned, 128-d, unit-variance init, tied to the output head |
| Denoiser | BERT-base encoder (110M) | `nn.TransformerEncoder`, pre-norm; size set by config (7.5M parameters for the CPU preset, 24M default) |
| Noise schedule | Linear | `sqrt` (Diffusion-LM) by default; linear and cosine available |
| Loss | MSE (+ NLL, prior), source positions included | MSE on response positions + rounding NLL + prior |
| Sampler | 2,000 ancestral steps, x0 misread as ε | DDIM-style deterministic update for an x0 prediction; any number of steps; clamping to the nearest word embedding |
| Evaluation | 4 prompts, sentence BLEU | Corpus and sentence BLEU, distinct-1/2, length, sample dump |
| Tests | None | Schedules, partial noising, perfect-denoiser reconstruction, data layout, tiny overfit |

### Loss

For response positions (mask = 1), with `emb` the word embedding of the
target tokens and `x0 = emb + sqrt(1 − ᾱ_0)·z`:

    L = ‖f(x_t, t) − x0‖²   (x0 replaced by emb when t = 0)
      + CE(softmax(x0·Eᵀ), tokens)                (rounding / decoder NLL)
      + ‖sqrt(ᾱ_{T−1})·x0‖²                       (prior; keeps x_T ≈ N(0, I))

Source positions are never noised (`q_sample` keeps them), so they condition
the response through self-attention, with no separate encoder or classifier.

### Sampling

Starting from `x_T ~ N(0, I)` on response positions and the clean source
embedding elsewhere, for a decreasing subsequence of timesteps
`t_1 > t_2 > … > t_K`:

    x0_hat  = clamp(f(x_t, t))                  (snap to nearest word embedding)
    ε_hat   = (x_t − sqrt(ᾱ_t)·x0_hat) / sqrt(1 − ᾱ_t)
    x_s     = sqrt(ᾱ_s)·x0_hat + sqrt(1 − ᾱ_s)·ε_hat

and the final `x0_hat` is rounded to tokens. This is DDIM with η = 0 written
for an x0-predicting network (Song et al., 2021), so K can be far smaller than
T. That is what makes evaluation practical on a CPU.

## 5. Results

See `docs/RESEARCH_LOG.md` for each run's settings and scores.

## References

- Gong, S. et al. (2023). DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models. ICLR.
- Li, X. L. et al. (2022). Diffusion-LM Improves Controllable Text Generation. NeurIPS.
- Ho, J., Jain, A. & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
- Song, J., Meng, C. & Ermon, S. (2021). Denoising Diffusion Implicit Models. ICLR.
- Nichol, A. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. ICML.
