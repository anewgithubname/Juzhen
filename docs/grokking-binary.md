# Binary classification grokking: sparse XOR

The native example also supports a binary task with `GROK_TASK=parity`.
There are 40 input bits and the label is `bit[0] XOR bit[1] XOR bit[2]` (0 or 1).
The other 37 bits are irrelevant. Inputs are encoded numerically as -1/+1;
bit 0 is the least significant bit of the saved integer example ID.

This task is inspired by [A Tale of Two Circuits](https://arxiv.org/abs/2303.11873).
It is an adapted experiment: that paper uses SGD and hinge loss, whereas this
experiment keeps Juzhen's existing AdamW and cross-entropy training path.

## Protocol

- Plain ReLU MLP, `40 → 512 → 2`, 22,018 trainable parameters. No embeddings,
  attention, normalization, skip connections, dropout, or handcrafted parity features.
- 1,000 unique training examples and 10,000 disjoint heldout examples, sampled
  uniformly from 40-bit strings with host `mt19937_64`. Duplicate IDs are rejected.
  Targets are deterministic and have no label noise. Chance accuracy is 50%.
- Full-batch, 20,000 updates, evaluate both complete splits every 100 updates.
  The two output scores use ordinary two-class cross entropy.
- He-normal hidden weights, normal output weights with standard deviation
  `1/sqrt(512)`, zero biases. Dataset and initialization seeds are recorded.
- Selected configuration: AdamW learning rate .01, betas (.9, .98), epsilon 1e-8,
  decay .1 on all parameters, including biases. No schedule, clipping, or early stop.
- A matching no-decay control uses exactly the same initial arrays and examples.
- Operational grokking criterion: after first recorded train accuracy ≥99%,
  heldout accuracy is still ≤60%, and reaches ≥95% at a later recorded update.
  This criterion is a descriptive convention, not a universal definition.

## Main result, seed 42

| Update | Train accuracy | Heldout accuracy |
| --- | ---: | ---: |
| 100 | 100% | 49.47% |
| 14,400 | 100% | 95.78% |
| 15,100 | 100% | 99.20% |
| 20,000 | 100% | 100% |

The no-decay control also reaches 100% train accuracy at update 100 but ends at
49.79% heldout accuracy. Its heldout cross entropy grows to 10.47 while its
training cross entropy rounds to zero. The successful run's final losses are
`3.139e-5` on training and `4.170e-4` on heldout data. Both take about 3.4 seconds
inside the measured loop on an RTX 4090, excluding initialization/final save.

## Exploration and fixed-configuration replications

The successful configuration was found by exploratory heldout monitoring.
All unsuccessful trials are retained; default settings do not reliably induce
grokking on this task. These are all nine full native runs, in execution order:

| Artifact directory | Seed | LR | Decay | Updates | Final heldout accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decay1` | 42 | .001 | 1 | 20,000 | 49.72% |
| `decay0` | 42 | .001 | 0 | 20,000 | 50.69% |
| `long100k` | 42 | .001 | 1 | 100,000 | 49.72% |
| `lr01-decay1` | 42 | .01 | 1 | 20,000 | 50.54% |
| `lr01-decay01` | 42 | .01 | .1 | 20,000 | 100% |
| `lr01-decay001` | 42 | .01 | .01 | 20,000 | 50.34% |
| `lr01-decay0` | 42 | .01 | 0 | 20,000 | 49.79% |
| `seed43` | 43 | .01 | .1 | 20,000 | 100% |
| `seed44` | 44 | .01 | .1 | 20,000 | 99.87% |

Seeds 43 and 44 were run after freezing the selected configuration. Both reach
100% train accuracy at update 100, with heldout accuracy 50.12% and 50.73%.
They first reach 95% heldout accuracy at updates 10,400 and 10,900 respectively.
No failed replication was omitted or extended to obtain these results.

After freezing configurations, a fresh set of 100,000 unique examples was
generated with independent NumPy seed 20261021. It excludes the union of all
33,000 prior examples across seeds 42/43/44. Frozen final parameter archives
were evaluated with the independent Torch implementation, without updates:

| Model | Correct / 100,000 fresh examples |
| --- | ---: |
| Decay .1, seed 42 | 100,000 |
| Decay .1, seed 43 | 100,000 |
| Decay .1, seed 44 | 99,896 |
| No decay, seed 42 | 49,901 |

These finite-sample results demonstrate delayed generalization on this binary
task; they do not establish that every binary problem or every seed will grok.

## Verification and artifacts

Independent Torch checks cover disjoint splits, labels/encoding, initial training
loss, gradients of all four tensors, the first AdamW update, and final full-split
losses/accuracies for the main run, its matching control, and both replications.
Main-run maximum gradient error is `3.73e-8`; maximum first-update error is
`2.71e-5` (comparison atol `3e-5`, rtol `1e-3`). CPU binary smoke and the old
modular-addition path also pass; the old addition split is byte-identical.

All models, datasets and plots are stored at:
`/mnt/experiments/Backup/Juzhen-data/grokking/20261021/parity`.

Each run contains `examples.csv`, `config.json`, `metrics.csv`, initial/first/final
parameter archives, initial/final/native checkpoints, and selected verification
reports. `source/` preserves the exact training source and executable.
`fresh-examples.csv`, `fresh-test.py`, and `fresh-test.json` preserve the independent
evaluation. `comparison.png` shows the selected matched comparison, while
`replications.png` shows all three selected-configuration seeds.

## Reproduce

Build `demo_grokking` as described in [grokking.md](grokking.md). Use fresh output
directories beneath the mounted external storage root; existing metrics are
never overwritten.

```bash
mountpoint -q /mnt/experiments || exit 1
run_root=/mnt/experiments/Backup/Juzhen-data/grokking/my-binary-run
GROK_TASK=parity GROK_LR=.01 GROK_DECAY=.1 GROK_OUTPUT="$run_root/decay01" \
  NVIDIA_TF32=0 /tmp/juzhen-grokking-build-cuda/demo_grokking
GROK_TASK=parity GROK_LR=.01 GROK_DECAY=0 GROK_OUTPUT="$run_root/decay0" \
  NVIDIA_TF32=0 /tmp/juzhen-grokking-build-cuda/demo_grokking
PYTHONDONTWRITEBYTECODE=1 python3 scripts/grokking.py verify "$run_root/decay01" --device cuda
PYTHONDONTWRITEBYTECODE=1 python3 scripts/grokking.py verify "$run_root/decay0" --device cuda
PYTHONDONTWRITEBYTECODE=1 python3 scripts/grokking.py plot \
  "$run_root/decay01" "$run_root/decay0" --output "$run_root/comparison.png"
```

Change `GROK_SEED` to reproduce the fixed-configuration replications. Additional
parity-specific options are `GROK_BITS` (default 40), `GROK_TRAIN` (1000), and
`GROK_TEST` (10000). Only the first three bits determine the label regardless of
input width. For CPU smoke: `GROK_TASK=parity GROK_BITS=8 GROK_TRAIN=32
GROK_TEST=128 GROK_HIDDEN=16 GROK_STEPS=3 GROK_EVAL_EVERY=1`.
