# TinyStories: historical settings through explicit training

The `T8_LEGACY_TINYSTORIES=1` mode of `demo_text8_lm` keeps the old
`demo_tinystories` model and optimization recipe while using integer token IDs,
gradient-only backward, one explicit optimizer step and complete checkpoints.
Default character and byte-BPE modes retain their previous protocol.

| Setting | Historical-compatible mode |
| --- | --- |
| Model | 8 causal PreLN ReLU blocks, width 512, FF 2048, 8 heads |
| Input | Token lookup + position lookup + learned input bias |
| Output | Independent linear projection with bias; no final LayerNorm |
| Vocabulary | Full-train Metaspace BPE, 4096 entries, EOT special token |
| Data | Entire official TinyStories V2 GPT4 train and validation |
| Batch/context | 64 / 256 (16,384 tokens per update) |
| Optimizer | Adam, beta .9/.999, epsilon 1e-8, no decay or clipping |
| Schedule | 60,000-update budget; 1,000 warmup; cosine .0005 to .00005 |
| Initialization | Token/position/projection and input/output bias std .001; block weight std 1/sqrt(fan-in); block biases zero; LN gamma1/beta0 |
| Validation | Original fixed 64-window batch, after updates 1, 501, ... and the final update |
| Early stop | 12 validations without a strict improvement |
| Precision | FP32 storage and TF32 GEMMs, matching historical speed run |

The old dense token/position one-hot projection is mathematically the two
lookups plus an input bias. The independent Torch model tests the same graph,
raw gradients, first Adam update, validation offsets, and process restart.
CUDA atomic embedding reductions allow small floating point variation.

The original model weights and tokenizer were recovered after this experiment
was trained. The run's sealed configuration retains the provenance recorded
before that recovery; its historical-availability fields describe that earlier
state. Retraining the original recipe does not establish identical historical
merges or initialization. The historical final update count remains unknown;
the 60k budget is the one documented in `misc/tinystories_loss_benchmark.md`.
The recorded historical validation CE 1.32 and 141 ms/update are reference
numbers. A separate recovered-checkpoint comparison uses the same held-out
stories and byte-normalized loss across tokenizers; see below. This is not the
recent 100k-story, byte-BPE-8192 experiment.

All data and outputs must stay below the mounted external storage root.
Download the two pinned official GPT4 text files and save their sizes/hashes in
`raw/sources.json` (see the experiment's saved `download.py`). Then run:

```bash
python scripts/tinystories_legacy_prepare.py /mnt/experiments/Backup/Juzhen-data/llm/EXPERIMENT
python scripts/tinystories_legacy_run.py verify --binary /path/to/demo_text8_lm --root /mnt/experiments/Backup/Juzhen-data/llm/EXPERIMENT
python scripts/tinystories_legacy_run.py run --binary /path/to/demo_text8_lm --root /mnt/experiments/Backup/Juzhen-data/llm/EXPERIMENT
```

Encoding uses contiguous story-aligned shards, merges them in source order and
writes little-endian int32 IDs. Universal newline conversion, story stripping,
EOT handling and tokenizer training match the old preparation recipe. This mode
intentionally retains the original full train/validation split rather than the
recent deduplicated subset. There is no independent test set.

The runner seals config, dataset hash and binary, saves best and last full
checkpoints, then automatically evaluates the selected model. It saves the old
40 prompt openings in their original order (T=.8, top-k40, seed42), full 200-token
traces and first-EOT story text. It does not invoke a paid judge. A separate
32-batch validation estimate and independent FP32 loss check accompany the
legacy selection metric. The FP32 frozen-evaluation and generation entry points
write to explicit destinations and preserve all training files.

## Completed run and recovered checkpoint

The native run completed 60,000 updates with 29,536,768 parameters. Its best
legacy fixed-batch validation loss was 1.32899022102 at update 58,501. This is
the experiment's selection metric, not a complete independent test-set score.

The recovered original weights and the new model were evaluated on 1,024 shared
held-out stories using their respective tokenizers. Bits/byte were
0.41138807 for the original and 0.41051643 for the new model. The paired
confidence interval includes zero, so this comparison does not establish a
quality improvement. The original checkpoint's recorded validation loss uses
a different protocol and must not be directly ranked against the new selection
metric. `scripts/tinystories_checkpoint_compare.py` records inputs, hashes and
the comparison protocol under the supplied external output root.

The Sequential migration retains parameter names, initialization order, optimizer
settings, batches and checkpoint layer ordering. It reuses the core interfaces
from main without adding LLM-specific behavior to layer.hpp or the optimizer.

The saved source revision is provenance rather than a resume restriction, as in
the text8/BPE runners. A source revision change preserves the original sealed
configuration and pinned training binary; data hashes and model/optimizer
configuration must still match. To train with another binary, use a fresh run
directory or an explicitly controlled native checkpoint-resume invocation.
