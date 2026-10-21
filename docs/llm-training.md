# Explicit token-model training

Include `ml/llm.hpp`. The first implementation supports CPU `float` and CUDA
`CUDAfloat`. Existing `eval`, `value`, and list-based `forward` retain their behavior.
`backward` computes and accumulates gradients without updating parameters.
The former updating layer `backward` and list-based `backprop` functions are now
named `backward_update`; migrate existing updating calls to that name.
Gradient buffers are lazy, so
legacy training does not allocate a second set of parameter-sized buffers.

## Sequential training

The general gradient-only API, native LossLayer example and ownership contract
are documented in [Training API](training-api.md).

LLM examples use `Sequential` for their output-first floating-point layer chain:
Transformer blocks, final normalization when present, and output projection.
Integer Embedding entry points run before this chain. TokenCrossEntropy consumes
the chain's logits and integer targets; its gradient seeds `Sequential::backward`.
The returned activation gradient is passed to both embedding tables.

## Integer input and loss

`TokenIds<D>` owns immutable int32 IDs and uploads them lazily on CUDA. An
`Embedding<D>(vocabulary_size, embedding_dimension)` looks up columns of its
`[dimension, vocabulary]` weight table. Call `eval(ids)`, then `value()` to obtain
`[dimension, token_count]` activations. `backward(upstream)` accumulates
repeated IDs into the dense weight gradient; CUDA uses atomic addition.

`TokenCrossEntropy<D>` accepts `[vocabulary, token_count]` logits and integer
targets with `eval(logits, targets)`. `value()` is the mean loss over valid tokens;
`valid_tokens()` reports the denominator. `grad(scale)` returns a dense logits
gradient. Default `ignore_index` is -100. Empty/all-ignored inputs return zero
loss and zero gradients. Skip an optimizer update if the entire accumulation
window has no valid tokens; calling backward on zeros still creates defined
gradients, which AdamW can use to update momentum and apply weight decay.

The loss borrows the logits matrix: keep it alive and unchanged until `grad()`.
Rvalue logits are rejected. The implementation caches only per-token maximum
and log-normalizer, not a full softmax or a logits copy. Large positive logits
are handled with shifted log-sum-exp. Each backward output is still dense.
For causal language modeling, prepare targets by shifting the input tokens one
position; the loss does not shift them implicitly.

## Parameters, sharing, and updates

`LinearLayer` and `TransformerLayer` provide `parameters()` and
`backward(input, upstream)`. Standalone `LayerNorm` provides `parameters()`
and `backward(upstream)`. Unsupported layers fail explicitly; the network
traversal checks the entire list before starting backward. List order remains
output-first, matching the existing `forward` function.

A `ParameterRef` borrows a stable matrix object, gradient buffer, trainable flag,
and the layer's existing Adam state. Keep parameter-owning layers alive and do
not move or assign whole layer objects while an optimizer uses their references.
`W()` and `b()` continue to expose the existing matrices. Gradients are available
through each optimizer parameter's `grad()` and `has_grad()`.

`TiedProjection<D>(embedding)` computes `embedding.weight().T() * hidden`.
It contributes to the same gradient as input lookup. Its input JVP uses the
shared linear map; integer Embedding inputs reject input JVPs explicitly. Registering that same
parameter twice is deduplicated; inconsistent registrations or distinct matrix
objects aliasing the same live storage are rejected. Do not implement tying by
copying matrices or independently registering a transposed matrix.

```cpp
Embedding<D> token(vocab, width);
TransformerLayer<D> block(width, width, 4*width, seq, batch, heads, true);
TiedProjection<D> head(token);
Sequential<D> network{&head, &block};
AdamW<D> optimizer(network.parameters()); // Includes the shared token weight.
TokenCrossEntropy<D> loss;

optimizer.zero_grad();
token.eval(input_ids);
loss.eval(network.eval(token.value()), target_ids);
auto dx = network.backward(loss.grad());
token.backward(dx);
optimizer.clip_grad_norm(1.0);
optimizer.step();
```

A real GPT model also adds position embeddings; the runnable
`examples/demo_llm_tokens.cu` shows this and two-microbatch accumulation.
Its cyclic synthetic token stream is a learning check, not a language benchmark.

`AdamWOptions` configures learning rate, betas, epsilon and decoupled weight
decay. Weight decay applies to weight tables/matrices; bias and LayerNorm groups
are excluded. Set `weight_decay=0` for Adam. `step()` preserves gradient values,
skips frozen/absent gradients and advances each updated parameter's existing
Adam counter once. It returns the number of distinct parameters updated.
Call `zero_grad()` once per accumulation window; it zeros allocated buffers and
marks gradients absent. `clip_grad_norm` returns the pre-clip global norm and
rejects nonfinite norms before modifying gradients.

When every microbatch has the same valid-token count, use
`loss.grad(1.0f / number_of_microbatches)`. Otherwise scale each mean gradient by
`microbatch_valid_tokens / total_valid_tokens_in_the_window`. A mean of
microbatch means is incorrect when valid-token counts differ.

## Checkpoints and compatibility

States remain in their existing layer locations, preserving the version-2
checkpoint format and old model loading. Construct the optimizer before loading
so its initialization does not overwrite restored learning-rate/beta settings.
Recreate it with the same weight decay and parameter-group options: those are
caller configuration, not fields in the existing checkpoint format. Save at
optimizer-step boundaries and call `zero_grad()` before resumed training.
Gradient accumulation buffers are not checkpointed. Data iterator and scheduler
restoration retain the existing caller-managed contract.

Include both Embedding owners in a checkpoint network if using token/position
embeddings. The tied output head serializes no duplicate weights or moments.
Use the same layer list and shared-weight configuration when loading.
Old `freeze`/`unfreeze` behavior remains; a frozen layer's explicit backward
still propagates input gradients but does not accumulate parameter gradients.
Embedding provides `set_trainable`; standalone LayerNorm has `trainable`.
The float-only `forward` traversal intentionally does not accept an Embedding
as its integer input layer. Call the integer entry point first, as above.

## Validation

`testLLMTraining` runs scalar/operator checks, repeated IDs, transposed layouts,
invalid inputs, empty/ignored targets, non-power-of-two and 50k vocabularies,
sharing, transposed optimizer storage, frozen embeddings and token-weighted
gradient accumulation. General Adam, clipping and LayerNorm checks remain in
`testTrainingAPI`. It writes no files by default.

Configure `LLM_TEST_OUTPUT` to an explicit fixture/checkpoint directory to enable
`llm_torch_compare`. This runs the binary with checkpoint recovery checks and
compares loss, all 15 parameter gradients, clipping and three AdamW steps with
an independent float64 PyTorch reference. On this workstation, that directory
must be under the mounted external data root specified in local `AGENTS.md`.

`demo_llm_tokens` can optionally save/resume a caller-selected
`JUZHEN_LLM_CHECKPOINT` path. `JUZHEN_LLM_STEPS` is the total target step count,
including steps restored from that file. Neither example nor unit checks
implicitly create a model cache.

## CUDA global gradient norm

CUDA clipping reduces all active gradient buffers together, using FP64 partial
sums to avoid FP32 square overflow/underflow, and reads one checked global norm
back to the host. When clipping is needed, one tiled launch scales the active
buffers. CPU clipping and the per-parameter AdamW update formula stay the same.
The norm can differ in the last bits from the previous sequence of FP32 cuBLAS
norm results; gradients and parameter updates are validated against PyTorch.

Scratch allocations grow on demand and hold descriptors/partial sums only;
they are not parameter or optimizer state and are not checkpointed. Gradient
addresses are uploaded on every call, so replaced storage, changed active
parameters and checkpoint recovery do not leave stale pointers. Optimizer
copies may share the scratch, under the existing single-thread/default-stream
contract. Frozen/absent groups remain excluded, and NaN/Inf is rejected before
any gradient is scaled. Transposed storage is supported. Unusual registrations
sharing one gradient allocation retain the previous sequential clipping behavior.

## CUDA normalization and activation backward

For contiguous CUDA tensors with at least 32 features, LayerNorm forward and
input-gradient kernels use one warp per token. Adjacent lanes access adjacent
features and compute two-pass mean/variance or gradient reductions together.
Narrow columns keep the existing serial kernels; transposed tensors and other
backends retain the generic paths. Epsilon remains 1e-5, and cached normalized
values, JVP, parameter-gradient accumulation and checkpoint layouts are unchanged.
Reduction order changes FP32 rounding, so equality is checked with tolerances.

Transformer ReLU backward applies the existing `x > 0` derivative to its owned
upstream buffer in place, avoiding the activation copy and derivative temporary.
The cached activation is read-only. Transposed layouts retain the generic path;
zero, signed zero and nonfinite upstream behavior match the original product.

`testLLMTraining` checks normalization against an independent FP64 reference,
including narrow/partial-warp/large dimensions, constant and small-variance inputs,
transposed layouts, frozen parameters and accumulation. Parameter reductions are
also checked against an FP64 reduction of the actual cached normalization. The
text8 pipeline's width-32 model exercises the new CUDA paths against PyTorch
and through separate-process checkpoint recovery.
