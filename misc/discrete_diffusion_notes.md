# Masked Discrete Diffusion — experiment notes

Reference notes for `examples/demo_discretediffusion.cu`: a masked / absorbing-state
discrete-diffusion character language model (the D3PM-absorbing / MDLM recipe),
trained on text8 / enwik8. Records what we tried, the numbers, and the lessons.

## What we built

- **Absorbing-state masked diffusion.** Vocab = corpus charset + one extra `[MASK]`
  token. Forward (noising) process: sample a mask level `t ~ U(t_min, 1)`, replace
  each token with `[MASK]` independently with probability `t`. The denoiser predicts
  the original characters at masked positions.
- **Bidirectional transformer denoiser.** Added a `causal` flag to
  `TransformerLayer` (`ml/layer.hpp`, default `true` so the autoregressive demo is
  unchanged); the diffusion demo builds blocks with `causal=false`.
- **Custom masked cross-entropy** (`MaskedCELayer`): stable softmax CE, supervised
  **only at masked positions**, each weighted by `1/t` — the continuous-time MDLM
  ELBO weight for a linear schedule. Reported as bits-per-character (an upper bound
  on the true NLL).
- **MaskGIT-style generation**: start from all-`[MASK]`, iteratively denoise over
  T steps, committing the highest-confidence positions per a cosine schedule with
  annealed Gumbel noise. Plus a bidirectional **infilling** demo.

## Results (bits-per-character; lower is better)

All char-level, held-out val set. Diffusion BPC is an ELBO **upper bound**; AR BPC
is the **exact** likelihood (so AR has an inherent metric advantage).

| run | model | batch × seq | dataset | BPC | acc |
|---|---|---|---|---|---|
| v1 baseline | 256d/4L | 64×64 | enwik8 | 2.879 | 34% |
| v2 + variance-reduction + sampler | 256d/4L | 64×64 | enwik8 | 2.598 | 38% |
| v3 wider+deeper | 384d/6L | 64×64 | enwik8 | 2.394 | 41% |
| v4 wider+deeper, tiny batch | 512d/8L | 64×64 | enwik8 | ~2.40 | ~40% |
| **v5 + batch + context** | **512d/8L** | **128×128** | enwik8 | **2.239** | 43% |
| **text8 (best overall)** | 512d/8L | 128×128 | **text8** | **2.123** | 42% |
| lit-scale (undertrained) | 768d/12L | 48×256 | text8 | 3.335 | 24% |

Comparison points (same hardware where noted, else published):

- **Autoregressive transformer, same scale** (512d/8L, seq 128, enwik8): **1.503 BPC**,
  69.5% next-char accuracy. AR wins on pure BPC — partly metric (exact NLL vs ELBO),
  partly task (next-char is easier than denoising at arbitrary noise). AR cannot do
  the infilling task at all.
- **Published masked diffusion on text8**: D3PM-absorbing **1.45**, MD4 ~1.37–1.45
  (12-layer, seq 256, 100k–1M steps). Published AR on text8 ~1.18–1.23.

## What we learned

1. **Variance reduction was the single biggest lever.** The `1/t` ELBO weight is
   very high-variance under i.i.d. `t` (rare small-`t` sequences dominate the
   gradient). **Stratified / antithetic sampling of `t`** across the batch unstuck
   training — it turned a variance-limited run that plateaued at step 16k into one
   that improved monotonically, and it's what made subsequent scaling pay off.

2. **Throughput (batch × context) beats raw capacity at a fixed step budget.**
   v4 (more layers, same tiny 4096-token batch) barely improved; v5 (same model,
   4× tokens/step via bigger batch + longer context) gave the biggest single jump.
   The bottleneck was gradient noise, not parameters.

3. **GPU memory is a diagnostic.** At 4 GB used / 99% util the model was
   *compute-bound but data-starved per step*. Scaling batch/context filled the card
   (→15 GB) and improved results; stacking depth alone did not.

4. **A framework memory win: don't allocate activation buffers you never use.** The
   early-stopping "best model" snapshot was a full `TransformerLayer` that reserved
   a duplicate set of per-batch activation caches it never touches. Building it with
   `batch=1` (weights are batch-independent) **halved activation memory**
   (15 GB → 8.5 GB for the default config).

5. **Attention memory is `O(seq_len² · batch)`** — the practical cap on context.
   The literature's seq 256 forces a small batch on a 24 GB GPU.

6. **Literature architecture ≠ literature results without literature compute.**
   Adopting the published 768d/12L/seq256 model trained *worse*, not better: at a
   tractable ~20k-step budget it is badly **undertrained** (3.34 BPC, incoherent
   samples). Big models are more sample-efficient asymptotically but need far more
   steps to get there. Reproducing ~1.45 is a compute-budget problem (many
   GPU-hours), not a config tweak.

7. **Learning rate must scale down with depth.** Peak LR 5e-4 (fine for 8 layers)
   destabilized the 12-layer model right as warmup ended (loss oscillated).
   Lowering to 2e-4 fixed the *instability* but not the *underperformance* — the
   root cause there was undertraining (#6), not LR.

8. **Match the dataset for fair comparison.** The discrete-diffusion literature
   reports **text8** (27-symbol lowercased alphabet), not raw **enwik8** (205
   byte-level symbols). We added text8 preprocessing so our BPC lands on the same
   axis as the published numbers.

## Bottom line

Best result on the target benchmark: **2.12 BPC on text8** with a model that trains
in ~15 min and 8.5 GB on one 24 GB GPU. The implementation is correct (absorbing
beats uniform, learns dataset conventions such as text8's spelled-out digits, and
infills), and the ~0.6 BPC gap to same-scale AR and the further gap to the published
1.45 are both explained (ELBO vs exact likelihood; small-model/short-budget vs the
literature's large-model/long-budget training).
