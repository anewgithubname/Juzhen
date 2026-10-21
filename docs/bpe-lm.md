# Byte-level BPE language modeling

`scripts/bpe_lm.py` trains a byte-level BPE tokenizer using the installed Python
`tokenizers` library, prepares integer token files, and runs the existing Juzhen
LM executable. Tokenization/decoding stays in Python; training and native
sampling use Juzhen. No core layer API changes are needed.

## Data and tokenizer

Raw UTF-8 text is split into contiguous 90/5/5 Unicode-character intervals
before any tokenizer training. Only the training interval learns BPE merges.
Each split is then encoded independently; next-token windows never cross splits.
The tokenizer has all 256 byte symbols, no normalization, no added prefix space,
and no special tokens. It preserves case, punctuation and arbitrary UTF-8 text.
This single-stream experiment does not add document/EOS boundaries.

Vocabulary defaults to 8192. `tokenizer.json`, source/train hashes, tokenizer
version/settings and all split hashes are sealed in `metadata.json`; reuse and
resume reject changed content. Files contain little-endian int32 IDs. The native
reader validates alignment and ranges, including IDs above 255. Legacy text8
byte files, alphabet and checkpoint protocol remain the defaults.

Large strings are chunked before whitespace runs to preserve word prefixes.
Encoding/decoding and per-token byte lengths are checked for each chunk. The
byte lengths count byte-level alphabet symbols, including partial UTF-8 symbols,
rather than UTF-8 replacement characters from decoding individual tokens.

The tokenizer API follows the official [Tokenizers quicktour](https://huggingface.co/docs/tokenizers/quicktour)
and [ByteLevel pre-tokenizer](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

## Run

Build `demo_text8_lm` normally. Choose explicit data/output locations; on this
workstation they must be on the mounted external storage root in `AGENTS.md`.

```sh
python scripts/bpe_lm.py run --input /path/to/text8 \
  --output /external/bpe-run --binary /path/to/demo_text8_lm \
  --storage-mount /external/mount --device cuda --stop-after 2
```

Repeat the same command without `--stop-after 2` to resume both frameworks, or
add `--juzhen-only` to continue only native training after first-step parity is
established. The total steps, scheduler, tokenization, architecture and data
protocol stay fixed across process restart. Native and Torch runs load identical
initial weights; first-step losses, every parameter gradient and updates are
compared. PyTorch is an independent reference, not the native training backend.

All architecture options from the character experiment are supported. Defaults:
6 layers, width 384, 6 heads, context 128, batch 16, 10000 steps, 300 warmup steps.
This retains ReLU and FP32 rather than claiming full GPT-2 architectural parity.

To sample another prompt from a native checkpoint:

```sh
python scripts/bpe_lm.py generate --output /external/bpe-run \
  --binary /path/to/demo_text8_lm --prompt 'the history of ' --tokens 128 \
  --storage-mount /external/mount
```

Python encodes the prompt, Juzhen samples integer IDs, and Python decodes them.
Short contexts are right-padded and logits are read at the last real token.
Generated IDs and logits are retained so partial UTF-8 tokens and sampled
prefixes can be checked without incorrectly re-encoding generated text.

## Metrics and checks

CE and perplexity are per BPE token. `test_bits_per_token` is deliberately not
called character BPC. Reports additionally divide sampled target log likelihood
by the exact target byte count to estimate bits/byte. Neither token perplexity
nor differently sampled byte-normalized windows is a controlled direct ranking
against previous character-model experiments.

Configure `BPE_TEST_OUTPUT` to enable the optional CPU/CUDA `bpe_pipeline` CTest.
It checks train-only merges, Unicode and chunk round trips, split boundaries,
int32 IDs above 255, invalid/truncated tokens, sealed tokenizer/data changes,
Torch gradient/update parity, separate-process resume, causal generation logits
through short/full/sliding contexts and inference without changing training files.

## Initial text8 result (2026-10-20)

The default 8192-token configuration produced 19,455,826 training tokens from
90 million training characters (4.626 characters/token). The 13,835,520-parameter
model completed 10,000 native FP32 updates on an RTX 4090 at a steady median
11.175 ms/update. Validation CE/token fell from 9.06551 to 4.56627. Sampled test
CE/token was 4.694950, token perplexity 109.393329, and byte-normalized loss
1.484300 bits/byte over 262,144 target tokens / 1,196,255 bytes.

Independent PyTorch SDPA evaluation of the final native weights differed by
3.57e-8 CE/token. First-step maximum gradient and update differences were
1.80e-5 and 3.95e-6. CPU/CUDA pipeline checks and loading a pre-BPE character
checkpoint passed. The full 10,000-step training used Juzhen only; this was not
a full-run PyTorch throughput comparison. Generated English remains repetitive.
text8 only exercises lowercase ASCII and spaces; Unicode round trips and byte
fallback are covered separately by the synthetic pipeline tests. These numbers
do not establish GPT-2-level quality or a controlled advantage over character
modeling.

## Longer training, checkpoint selection and generation evaluation

`run --prepared-data /external/frozen-data` reuses an integrity-checked tokenizer
and three encoded splits. It is mutually exclusive with `--input`; the data is
linked into the new output directory. Total steps and the cosine schedule remain
part of the sealed training protocol: a 50,000-step experiment starts in a fresh
directory rather than relabeling a completed 10,000-step checkpoint. The first
steps can still establish Torch parity before native-only training resumes.

At evaluated update steps, native training preserves `best.ckpt`, `best.params`
and `best-validation.txt` when validation CE improves. The selection is among
evaluated post-update states, including any explicit early stop used for parity;
test loss is never used to choose this checkpoint. `last.ckpt` retains the normal
resume state. Existing runs without best-state files remain readable.

Native generation accepts these optional environment settings without changing
the training protocol:

- `T8_TEMPERATURE`: nonnegative finite temperature; zero selects greedy argmax.
- `T8_TOP_K`: retained vocabulary count; zero permits the complete vocabulary.
- `T8_SAMPLE_SEED`: explicit sampling seed.
- `T8_CHECKPOINT`: checkpoint to read, defaulting to `last.ckpt`.
- `T8_GENERATION_OUTPUT`: separate existing directory for logits and generated IDs.

The defaults remain temperature 0.8, top-k 10 and the original seed offset.
`scripts/bpe_generation_eval.py` uses six fixed shared lowercase prompts, greedy
sampling and temperatures 0.6/0.8/1.0 with top-k 40 and seeds 17/42. Every sample
has 192 new BPE tokens. It records completion-only word repetition and word
counts, token repetition, EOS counts and first-story metrics. Repetition measures
neither factuality nor narrative consistency; cross-tokenizer token budgets
also produce different text lengths. All samples are retained, without selecting
only successful outputs. Independent frozen-weight Torch evaluation checks final
native test CE, best-checkpoint validation CE and selected causal generation
logits. Evaluation hashes protected training files before and after sampling.

`scripts/tinystories_bpe_prepare.py` retrieves a bounded subset from a pinned
revision of the [official TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).
Its default is 100,000 unique GPT-4 training stories plus 5,000 validation and
5,000 test stories from the official validation source. Stories are deduplicated
across all splits using whitespace-normalized hashes. Only training stories
learn BPE merges. This dataset keeps case and punctuation and uses one special
`<|endoftext|>` token; text8 has no special token. Raw subsets, split hashes,
source URLs/revision and token files are stored under the supplied mounted path.
The entire multi-GB source and any Hugging Face home-directory cache are avoided.
The two datasets and tokenizers differ, so their token perplexities are not a
controlled ranking of model quality.

For one reusable sample, the Python generation CLI exposes the same controls:

```sh
python scripts/bpe_lm.py generate --output /external/bpe-run \
  --binary /external/bin/demo_bpe_lm_cuda --checkpoint best \
  --prompt 'Once upon a time, there was a rabbit ' --tokens 192 \
  --temperature 0.8 --top-k 40 --sample-seed 17 \
  --destination /external/samples/rabbit --storage-mount /external/mount
```

`--temperature 0` is greedy. `--checkpoint last` and the original sampling
settings remain defaults; `--destination` keeps the run's original sample intact.
The sample directory includes its IDs, logits and sampling/checkpoint provenance.

A prompt ending in whitespace can leave a rare standalone ByteLevel space token
at the end of the prefix, rather than the common space-prefixed next word.
`bpe_generation_eval.py --trim-prompt-end` runs the same suite with terminal
whitespace removed. The October 20 experiment retains the original suite and
this additional post-hoc boundary check separately; it does not replace the
initial results with whichever prompt variant looks better.
