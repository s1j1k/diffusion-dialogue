# Literature notes: where text diffusion has gone since DiffuSeq

Working notes for the next stage of the project (comparison with newer
methods). Summarised in our own words from a literature search on
2026-11-18; check each reference before citing.

## Search prompt used

> What are the most significant advances in diffusion language models for
> text generation since 2024, particularly for sequence-to-sequence tasks and
> open-domain dialogue? Compare continuous embedding-space diffusion (e.g.
> DiffuSeq, Diffusion-LM, Plaid) with discrete/masked diffusion (e.g. SEDD,
> MDLM, LLaDA, block diffusion). Which approaches give the best quality per
> unit of training compute, which work at small scale (under 100M
> parameters, single GPU), and what techniques most improve sample quality and
> speed? Prioritise 2025–2026 papers.

## Main threads

**Continuous → discrete.** DiffuSeq-style models diffuse in a continuous
embedding space and need a rounding step back to tokens; the survey by Li et
al. (2025) identifies the training/sampling mismatch and optimisation
overhead as their main weaknesses. Most recent work instead diffuses over
tokens directly with an absorbing `[MASK]` state. MDLM (Sahoo et al., 2024)
and MD4 (Shi et al., 2024) show the training objective reduces to a weighted
masked cross-entropy, which is simple to implement and strong at GPT-2 scale.
The survey's Figure 4 compares AR, discrete, continuous and block diffusion
training and inference side by side, and is a useful framing figure for our
report (cite, do not reproduce).

**Scale.** LLaDA (Nie et al., 2025) trains an 8B masked diffusion model
from scratch that is competitive with LLaMA3-8B, including dialogue after
instruction tuning. Converting pretrained AR models is the cheaper route:
DiffuLLaMA/DiffuGPT (Gong et al., 2024), PreDiff-LM's hybrid attention
(Yao et al., 2026) and SDAR (Cheng et al., 2026).

**Block diffusion.** Generate block by block left to right, with parallel
masked denoising inside each block (BD3-LM; SDAR; Karami & Ghodsi, 2026).
This trades some parallelism for coherence and KV caching. E2D2 (Arriola et
al., 2025) splits the model into an encoder for clean tokens and a light
decoder for noisy ones, reporting better throughput and BLEU on
summarisation and translation, which is the closest setting to ours.

**Seq2seq continuous diffusion.** SeqDiffuSeq (Yuan et al., 2024): an
encoder–decoder transformer with self-conditioning and adaptive noise
schedules, keeping fluency with fewer steps. Continuous latent models
(AURORA-LM, Liang et al., 2026) use flow matching over learned decodable
latents.

**Sampling speed.** Trajectory-aware training (Madeira et al., 2026) cuts
function evaluations by up to 26 % on LLaDA-8B; training-free caching and
adaptive parallel decoding (Wu et al., 2026). Feng et al. (2025) show
theoretically that few steps suffice for low perplexity, but exact
sequence correctness needs steps growing with length.

## Candidate comparisons for this project

Feasible on one Colab GPU at the ~25M-parameter scale of the rewrite, same
data split and metrics:

1. **Masked diffusion (MDLM-style) vs the fixed continuous model.** Same
   transformer, swap the corruption process and loss. Most direct test of
   the field's main shift.
2. **Self-conditioning** (feed the previous x0 estimate back in) for the
   continuous model; cheap, often a large gain.
3. **Steps vs quality curve** for both, since sampling cost is the main
   practical argument against diffusion.
4. **Autoregressive baseline** of the same size, for an honest reference
   point.
5. Stretch: block diffusion, or an encoder–decoder split (E2D2-style).

## References

- Arriola, M. et al. (2025). Encoder-Decoder Diffusion Language Models for Efficient Training and Inference. arXiv:2510.22852.
- Cheng, S. et al. (2026). SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation. Findings of ACL.
- Feng, G. et al. (2025). Theoretical Benefit and Limitation of Diffusion Language Model. NeurIPS. arXiv:2502.09622.
- Gong, S. et al. (2023). DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models. ICLR.
- Gong, S. et al. (2024). Scaling Diffusion Language Models via Adaptation from Autoregressive Models. arXiv:2410.17891.
- Karami, M. & Ghodsi, A. (2026). Auto-Regressive Masked Diffusion Models. arXiv:2601.16971.
- Li, T. et al. (2025). A Survey on Diffusion Language Models. arXiv:2508.10875.
- Liang, J. et al. (2026). AURORA-LM: Autoencoding Unified Representation for Continuous-Latent Diffusion Language Modeling. arXiv:2608.02602.
- Madeira, M. et al. (2026). On Trajectory-Aware Training for Masked Diffusion Language Models. arXiv:2609.37974.
- Nie, S. et al. (2025). Large Language Diffusion Models. NeurIPS. arXiv:2502.09992.
- Sahoo, S. et al. (2024). Simple and Effective Masked Diffusion Language Models. NeurIPS. arXiv:2406.07524.
- Shi, J. et al. (2024). Simplified and Generalized Masked Diffusion for Discrete Data. NeurIPS. arXiv:2406.04329.
- Wu, T. et al. (2026). Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM. arXiv:2606.26120.
- Yao, Z. et al. (2026). PreDiff-LM: Pretrained Discrete Masked Diffusion Language Modeling with Hybrid Attention. arXiv:2607.25157.
- Yuan, H. et al. (2024). Text Diffusion Model with Encoder-Decoder Transformers for Sequence-to-Sequence Generation. NAACL.
