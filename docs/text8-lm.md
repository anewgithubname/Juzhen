# text8 character language model

This experiment connects the explicit training primitives to a real text task.
It reuses the existing `TransformerLayer`; no new core layer implementation is
required. `demo_text8_lm` supports CPU float32 and CUDA float32. The Python runner
prepares data, runs the independent PyTorch model and checks numerical results.
It does not download anything or start a training service.

## Model and experiment

The default model has 3,196,160 trainable parameters: 27 character embeddings,
128 learned position embeddings, four causal pre-LN Transformer blocks of width
256, four attention heads, ReLU feed-forward layers of width 1024, a final
LayerNorm and an output projection tied to the token embedding. There is no
dropout. This matches the existing Juzhen block rather than introducing new
activation or attention behavior.

Weights use Gaussian standard deviation 0.02, with residual output matrices
scaled by `1 / sqrt(2 * layers)`. Biases start at zero and norm scales at one.
Juzhen exports the exact initial weights for PyTorch, so framework RNG differences
cannot affect the comparison. The default run uses batch 32, context 128, 1,000
updates, AdamW (betas 0.9/0.95, epsilon 1e-8, decay 0.1), global gradient norm
clipping at 1 and learning rate 6e-4 with 100 warmup steps and cosine decay to
6e-5. Biases and normalization scales have no weight decay. These are experiment
settings, not a claim of optimal hyperparameters.

The raw text must contain only lower-case ASCII letters and spaces. Encoding is
`space=0, a=1, ..., z=26`. Contiguous 90%/5%/5% byte intervals become training,
validation and test files; neither inputs nor their shifted targets cross split
boundaries. Split hashes are validated on reuse. Both frameworks derive training
window offsets from the same stateless SplitMix64 function of seed and step.

Validation uses 32 batches (131,072 characters) and the final test uses 128
batches (524,288 characters), in evenly spaced windows within their respective
splits. Reported metrics estimate performance on those windows; they do not
cover every character in the held-out splits. Cross entropy is in nats per
character; BPC divides it by ln(2), and character perplexity exponentiates it.
A character-frequency baseline is fitted on the training split only.

## Build and run

Build `demo_text8_lm` using the normal CMake CPU or CUDA configuration. Python
requires NumPy, PyTorch and Matplotlib. Output/data locations are explicit:

```sh
python scripts/text8_lm.py run \
  --binary /path/to/build/demo_text8_lm \
  --input /path/to/text8 \
  --output /path/to/external/experiment \
  --storage-mount /path/to/external/mount \
  --device cuda
```

Use `--device cpu` with a CPU build. The runner checks the binary's backend.
On this workstation, all datasets, checkpoints, archives and generated results
must be under the mounted storage root in the local `AGENTS.md`. Pass
`--storage-mount` to refuse writes when the drive is unavailable. No fallback
storage location is selected.

`--dim`, `--layers`, `--heads`, `--seq`, `--batch`, `--steps`, `--warmup`,
`--eval-every`, `--eval-batches`, `--test-batches` and `--seed` select the protocol.
`--generate` selects the number of sampled characters after the fixed prompt.
Generation uses temperature 0.8 and top-k 10; its random stream is separate from
training. Frameworks need not produce the same sampled text.

To exercise a process restart, first add `--stop-after 100`, then repeat the
same command without that flag. The saved total `--steps` and other protocol
settings must stay unchanged because they determine the learning-rate schedule
and data order. Juzhen restores parameters, Adam moments/counters and progress;
PyTorch restores its native checkpoint. Sampler positions and scheduler values
are derived from the restored step. Metrics beyond the last saved step are
trimmed, preventing duplicate rows after a restart. Configuration/data changes
are rejected. Save points occur after optimizer updates and evaluation.

Outputs include the run configuration and split hashes, each framework's
metrics/checkpoint, initial and first-step parameter/gradient archives, generated
text, `learning-curves.png`, `results.json` and `report.txt`. First-step forward
loss, every parameter gradient and every updated parameter are checked against
PyTorch. Long FP32 trajectories are assessed by held-out losses rather than
bitwise final weight equality. Runs of at least 100 steps at width at least 64
must improve validation loss by 0.2 nats and keep the final framework validation
and test gaps below 0.08 nats; these are experiment checks, not model quality
benchmarks.

## Performance interpretation

Training logs measure full steps including batch construction/upload, forward,
backward, clipping, AdamW and reading the loss. Validation, checkpoint writes and
first-step archives are excluded from steady-state comparisons. Training runs
use matching explicit matmul/softmax attention and unfused PyTorch AdamW.
Optional nvidia-smi polling samples per-process GPU memory every 0.5 seconds;
these observations are approximate peaks, including framework/context overhead.

For a separate matched throughput measurement, set `T8_BENCHMARK=1` for the
native executable with the same `T8_*` configuration (and an existing output
directory). It writes `benchmark.json`. PyTorch benchmarks are available with:

```sh
python scripts/text8_lm.py benchmark \
  --config /path/to/experiment/config.json \
  --initial /path/to/experiment/juzhen/initial.params \
  --output /path/to/experiment/torch-benchmark.json --device cuda
```

Both benchmarks perform three rounds of 50 warmup and 100 measured updates.
They use the same initial weights, FP32 precision, input windows, clipping and
fixed 6e-4 learning rate. The Torch benchmark includes eager attention, SDPA,
and SDPA with fused AdamW; SDPA output/gradients are checked against eager first.
TF32 is disabled. SDPA uses PyTorch's available backend dispatch; the script does
not claim that FlashAttention is selected. Compare Juzhen against the optimized
baseline as well as the architecture-matched eager baseline.

## Regression checks

Configure `TEXT8_TEST_OUTPUT` to an explicit external output directory to enable
CTest `text8_pipeline`. It verifies split integrity, next-token alignment,
stateless sampling, refusal of unavailable storage and changed data/protocols,
multilayer loss/gradient/update agreement, and separate-process resume versus
uninterrupted training in both frameworks. Each invocation creates a fresh
small fixture so an old finished checkpoint cannot silently replace a test run.
The existing core LLM, CNN, JVP and checkpoint tests remain independently useful.

## Phase profiling

Set `T8_PROFILE=1` for the native executable, supplying the same configuration
and an existing output directory. The profiler writes `profile.csv` after
50 warmup steps and 100 measured steps. It separates batch construction,
zeroing gradients, forward, loss, backward, clipping, update and loss readback.
`host_ms` measures the call on the CPU; `stream_elapsed_ms` uses CUDA events.
The latter includes gaps while waiting for host launches, so it is not pure
kernel execution time. Nsight Systems can distinguish kernels, transfers and
host waits when more detail is needed.

Set `T8_PROFILE_SYNC=1` to synchronize at phase boundaries. This isolates each
phase's cost but changes overlap and adds synchronization overhead. Neither
profiling mode replaces the normal, uninstrumented throughput benchmark.
A fresh 5,000-step run should use a new output directory and `--steps 5000`;
its longer cosine schedule differs from the 1,000-step experiment. Keep each
run's schedule fixed when testing process restart.

## Local CUDA kernel optimization measurement

On the RTX 4090, the width-256, 4-layer, context-128, batch-32 FP32 experiment
was remeasured after CUDA global-norm clipping was already optimized. Cooperative
LayerNorm forward/input-gradient reductions and in-place Transformer ReLU
backward reduced median full-step latency from 7.017 to 5.567 ms (20.7% less
latency, 26.0% more tokens/s). This is five randomized, interleaved processes
per version, each with three rounds of 50 warmup plus 100 measured updates;
TF32 is disabled, with four CPU threads and the same CPU affinity.

Separate synchronized phase profiling measured forward 2.413 -> 1.788 ms and
backward 4.213 -> 3.368 ms. These phase measurements alter synchronization;
use the uninstrumented full-step benchmark for throughput comparisons. The
separately measured PyTorch SDPA + fused AdamW result was 4.922 ms, so this
optimization does not establish a speed advantage over optimized PyTorch.

## Short-prompt generation

Generation right-pads short contexts and reads logits at the last real token.
Causal attention hides the future padding, so a short prompt has the same
predictions as its unpadded causal prefix. Left-padding with ordinary space
characters shifts learned positions and creates an out-of-distribution prefix;
it previously caused repetitive samples despite valid held-out likelihoods.
This affects decoding only; the training and evaluation windows are unchanged.

Native and Torch examples write `generation.params` with the 27 logits at each
sampling step. The pipeline compares these traces to independent unpadded
forwards, covering short prefixes, a full context and sliding windows. Framework
sampling RNGs still differ, so sampled text need not be byte-identical.

For a native checkpoint, set `T8_GENERATE_ONLY=1` with its saved `T8_*` model
configuration and output directory to regenerate text without training or
rewriting parameters, metrics or checkpoints. The Python training CLI clears
this flag along with other inherited `T8_*` flags. Keep outputs on the mounted
external drive as for training.

## Native API integration

The model uses the core `Sequential` wrapper for its floating-point layer chain.
Token and position embeddings retain their integer-ID entry points. One backward
pass supplies both embedding gradients, and a separate AdamW/Adam step updates
the registered parameters. Named parameter order, initialization order and the
checkpoint layer list remain compatible with the pre-migration trainer.
