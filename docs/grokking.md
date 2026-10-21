# Native Juzhen grokking experiment

For the binary sparse-parity experiment, see [grokking-binary.md](grokking-binary.md).

`demo_grokking` learns `(a + b) mod 113` from a fixed subset of ordered pairs.
`GROK_MODEL=transformer` is the default; `GROK_MODEL=mlp` selects a plain MLP.
The implementation uses native `Embedding`, `TransformerLayer`, `LinearLayer`,
`Sequential`, `TokenCrossEntropy`, and `AdamW`. It changes no core training code.

The task, model dimensions, and optimizer settings follow the
[author's grokking demo](https://github.com/TransformerLensOrg/TransformerLens/blob/main/demos/Grokking_Demo.ipynb),
associated with [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217).
This is an adapted experiment: Juzhen's existing block has PreLN, biases remain
trainable, and initialization/data shuffling differ. The reference has no
LayerNorm and freezes biases. This is not an exact reproduction of its results.

## Protocol

- Inputs are `[a, b, =]`; only the final position predicts `(a+b) mod p`.
  The other two target positions are ignored, not trained to copy their inputs.
- All `113² = 12,769` ordered pairs occur exactly once in the saved split.
  A seeded shuffle assigns 3,830 pairs to training and 8,939 to the heldout set.
  Heldout examples never contribute a gradient. Evaluation includes the entire
  heldout set, including a padded final batch whose padding targets are ignored.
- One causal Transformer block: dimension 128, four heads, head dimension 32,
  ReLU feedforward dimension 512, two PreLN operations, no final LayerNorm.
  Token and position embeddings are learned; the output projection is untied.
- Initialization uses host `mt19937` seed 42, unit-normal token/position tables,
  normal linear weights with standard deviation `1/sqrt(fan_in)`, zero biases,
  and unit LayerNorm scales. Initial arrays are saved for numerical verification.
- Full-batch AdamW: constant learning rate .001, betas (.9, .98), epsilon 1e-8,
  20,000 updates, no warmup, schedule, clipping, dropout, or early stopping.
  Weight decay applies to **all** parameters, including biases and LN scales.
- Two runs change only weight decay, 1 versus 0. Their split and initial parameter
  archives have identical SHA256 hashes. TF32 is disabled. CUDA embedding scatter
  uses floating-point atomics, so training trajectories are not bitwise repeatable.
- Record train and heldout cross entropy/accuracy every 100 updates, plus total
  parameter norm and elapsed time. These are exploratory heldout measurements;
  there is no additional independent test set or multi-seed statistical claim.

## Observed result, 2026-10-21

RTX 4090, Juzhen CUDA backend, seed 42:

| Setting | Train accuracy at step 200 | Heldout at step 200 | Heldout at step 4,000 | Heldout at step 20,000 |
| --- | ---: | ---: | ---: | ---: |
| AdamW, decay 1 | 100% | 0.67% | 97.16% | 99.9888% |
| AdamW, decay 0 | 100% | 0.89% | 2.83% | 4.05% |

With decay 1 the first recorded heldout accuracy above 95% occurs at step 4,000,
and above 99% at step 4,100. Random chance is `1/113 ≈ 0.885%`. This is delayed
generalization after memorization. Without decay, this run does not reach that
criterion within the same 20,000 updates. This comparison does not establish that
weight decay is universally necessary for grokking.

The late loss curves oscillate under the constant learning rate. Accuracy with
decay 1 remains at least 95.48% from step 4,100 through the end, while the final
checkpoint gets 8,938 of 8,939 heldout examples right. Both runs take about 42
seconds inside the training/evaluation loop; this excludes initialization and
the final checkpoint write.

Results, full metrics, fixed splits, archives, checkpoints, and comparison plot:
`/mnt/experiments/Backup/Juzhen-data/grokking/20261021`.

The independent Torch checker validates data disjointness and labels, the first
backward pass and AdamW update, and final accuracy/loss on **both complete splits**.
CUDA first-gradient maximum error is `2.39e-7`; first-update maximum error is
`3.26e-6`. A small CPU smoke run also passes (gradient `7.46e-8`, update `1.50e-8`).

## Build and run

Use the repository's normal dependency setup, including its spdlog submodule.
The example is supported on the CPU float and CUDA float backends. Its storage
guard currently uses Linux mount information and the machine's external root.

```bash
git submodule update --init external/spdlog
cmake -S . -B /tmp/juzhen-grokking-build-cuda \
  -DCMAKE_BUILD_TYPE=Release -DNVIDIA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build /tmp/juzhen-grokking-build-cuda --target demo_grokking -j4

# Check the external drive before creating any output directory.
mountpoint -q /mnt/experiments || exit 1
run_root=/mnt/experiments/Backup/Juzhen-data/grokking/my-experiment
GROK_OUTPUT="$run_root/decay1" NVIDIA_TF32=0 \
  /tmp/juzhen-grokking-build-cuda/demo_grokking
GROK_OUTPUT="$run_root/decay0" GROK_DECAY=0 NVIDIA_TF32=0 \
  /tmp/juzhen-grokking-build-cuda/demo_grokking

PYTHONDONTWRITEBYTECODE=1 python3 scripts/grokking.py verify "$run_root/decay1" --device cuda
PYTHONDONTWRITEBYTECODE=1 python3 scripts/grokking.py verify "$run_root/decay0" --device cuda
PYTHONDONTWRITEBYTECODE=1 python3 scripts/grokking.py plot \
  "$run_root/decay1" "$run_root/decay0" --output "$run_root/comparison.png"
```

For CPU, configure `-DNVIDIA_CUDA=OFF` and use a small smoke configuration:
`GROK_P=7 GROK_DIM=16 GROK_STEPS=3 GROK_EVAL_EVERY=1`.
`GROK_OUTPUT` is required and must resolve beneath the mounted external storage
root. Existing run metrics are never overwritten; choose a new run directory.
Training resume is not implemented in this example; checkpoints retain native
weights, optimizer states, RNG, and progress for later inspection or extension.

Optional environment variables: `GROK_P`, `GROK_DIM`, `GROK_SEED`,
`GROK_FRACTION`, `GROK_LR`, `GROK_DECAY`, `GROK_STEPS`, `GROK_EVAL_EVERY`.
Use a prime modulus to follow the usual benchmark (the program also accepts
composite values). `config.json` records the resolved settings. Each run saves
`pairs.csv`, `metrics.csv`, `initial/first/final.params`, `first.grad`, and native
`initial/memorized/latest/final.ckpt` checkpoints. The archive format is the same
as the existing text8 checker. The program checks that the drive is mounted
before creating output and before checkpoint writes; it never creates a substitute
directory beneath an unmounted mount point.

## MLP comparison, 2026-10-21

The MLP has two affine layers: `226 → 512 → 113`, with a ReLU between them,
for 174,193 trainable parameters. Inputs concatenate the one-hot encodings of
`a` and `b`. It has no attention, embeddings, positional encoding, LayerNorm,
skip connections, or other Transformer components. The native `Sequential`
contains the two existing `LinearLayer` objects and an experiment-local ReLU
activation. No core layer or optimizer implementation was changed.

The split is byte-for-byte identical to the earlier Transformer experiment.
Seed, modulus, fraction, full-batch optimizer, all-parameter decay policy,
learning rate, and 20,000-update budget are also unchanged. Initialization uses
He normal weights for the hidden affine layer (`sqrt(2/226)` standard deviation),
normal output weights (`1/sqrt(512)`), and zero biases. Architectures and initial
parameter arrays differ from the Transformer; this comparison does not isolate
architecture from initialization or match parameter counts.

| Setting | First recorded train accuracy ≥99% | Heldout accuracy then | First heldout ≥95% | Heldout at step 20,000 |
| --- | ---: | ---: | ---: | ---: |
| MLP, decay 1 | Step 300 (100%) | 0% | Step 14,400 | 99.8658% |
| MLP, decay 0 | Step 300 (100%) | 0% | Not reached | 0% |
| Transformer, decay 1 | Step 200 (100%) | 0.67% | Step 4,000 | 99.9888% |

The MLP reaches 99% heldout accuracy at step 16,900 and gets 8,927 of 8,939
heldout pairs right at the end. Its decay-1 run clearly shows delayed
generalization after training accuracy has already saturated. Both MLP runs
take about 7.4 seconds in the measured loop on the same RTX 4090; the MLP
requires more updates to generalize, but each update is substantially cheaper.
This is one seed and split, not evidence for a general architecture ranking.

Independent Torch autograd checks all four MLP parameter tensors: maximum
first-gradient error `1.05e-9`, maximum first-update error `2.16e-6`. The checker
also reproduces final losses and exact accuracies on both complete splits,
including the zero heldout accuracy of the no-decay control. CPU MLP smoke and
the original Transformer path are separately checked.

MLP artifacts are under the `mlp/` subdirectory of the result root above.
Use the same executable and verification/plot commands, with new output paths:

```bash
GROK_MODEL=mlp GROK_OUTPUT="$run_root/mlp-decay1" NVIDIA_TF32=0 \
  /tmp/juzhen-grokking-build-cuda/demo_grokking
GROK_MODEL=mlp GROK_DECAY=0 GROK_OUTPUT="$run_root/mlp-decay0" NVIDIA_TF32=0 \
  /tmp/juzhen-grokking-build-cuda/demo_grokking
```

`GROK_HIDDEN` controls MLP hidden width (default 512). For a small CPU smoke run,
use `GROK_MODEL=mlp GROK_HIDDEN=16 GROK_P=7 GROK_STEPS=3 GROK_EVAL_EVERY=1`.
`GROK_DIM` applies to the Transformer. Older Transformer archives without a
`model` field remain readable by the verification and plotting script.
