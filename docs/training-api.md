# Training API

`backward` computes input and parameter gradients without updating weights.
`backward_update` retains the previous combined backpropagation and Adam update.
Existing calls to `backprop(net, x)` migrate to `backward_update(net, x)`;
existing layer calls that update weights migrate to `layer.backward_update(...)`.

`Sequential` borrows a native Juzhen list, in its established output-first order:

```cpp
#include "ml/layer.hpp"
#include "ml/optimizer.hpp"
using namespace Juzhen;

LinearLayer<float> linear(1, 2, batch_size);
LossLayer<float> loss(batch_size, targets);
Sequential<float> network{&loss, &linear};
AdamWOptions options;
options.weight_decay = 0;  // Adam without weight decay.
AdamW<float> optimizer(network.parameters(), options);

for (int step = 0; step < steps; ++step) {
    optimizer.zero_grad();
    network.eval(inputs);
    network.backward();
    optimizer.step();
}
```

`inputs` is an lvalue matrix with two rows and `batch_size` columns; `targets`
has one row and `batch_size` columns. Both matrices and all layers must outlive
their use by the network. `LossLayer` keeps a copy of its targets, as before.
The loss sums squared errors over output features and divides by batch size.

`eval` returns the final layer value. `backward()` seeds a scalar output with
one; a non-scalar output requires `backward(upstream_gradient)`. Backward uses
cached values and does not reevaluate layers. Repeated backward calls accumulate
gradients until `optimizer.zero_grad()`. Input and cached layer values must remain
unchanged between evaluation and backward. A failed evaluation invalidates the
wrapper's cached input. The wrapper does not own layers or the input.

The list-level forms remain available as `Juzhen::forward(net, inputs)`,
`Juzhen::backward(net, inputs, std::move(upstream))`, and
`Juzhen::parameters(net)`. The gradient-only path supports the fully connected
`Layer` (tanh), `ReluLayer`, `LinearLayer`, `ConvLayer`, `ConvTransLayer`,
`TransformerLayer`, and the `LossLayer`, `LogisticLayer`, and `MaskedCELayer` loss
heads. `LayerNormLayer<D>` is the single normalization implementation used both
inside `TransformerLayer` and directly inside `Sequential`.
`MaskedCELayer` is provided by `ml/masked_cross_entropy.hpp` and shared by the
three discrete-diffusion demos. Its per-column weights include the desired
normalization, so the layer applies no additional batch averaging. Targets must
be one-hot columns or normalized probability distributions; weights must be
finite and nonnegative. Loss heads accept a scalar upstream gradient and
register no trainable parameters. Their legacy checkpoint layouts are retained.

`ZeroOneLayer` reports classification error for evaluation only; it is
nondifferentiable and cannot be used in a training network. Direct subclasses of
`Layer` must explicitly implement and advertise gradient-only backward before
joining this path. Subclasses that change an adapted layer's computation must
also provide the corresponding backward implementation.
Networks containing unsupported layers are rejected before
gradient traversal. Existing `backward_update` training loops remain supported.
Convolution inputs, upstream gradients and JVP tangents must match the configured
dimensions. GPU convolution paths require non-transposed tensors, including
weights and biases; CPU convolution supports matrix transpose views.

`AdamW` supports CPU `float`, CUDA `CUDAfloat`, Metal `MPSfloat`, and ROCm
`ROCMfloat`, with parameter freezing, shared parameter registrations,
learning-rate changes and optional global gradient clipping. CPU and CUDA use
their optimized updates; Metal and ROCm use device matrix operations. The latter
two backends have not been exercised on hardware for this migration. It borrows existing layer Adam states so checkpoint storage stays
compatible. Weight decay excludes parameters registered with `decay=false`.

The `demo_training` target provides a complete regression example.
The `training_api` test checks finite-difference gradients, native-list and
Sequential equivalence, accumulation, legacy Adam equivalence, error handling,
gradient clipping and normalization against scalar references.

The `all_layer_training_api` regression exercises activation, convolution and transpose
convolution layers through the explicit training API. `masked_loss_api` checks
masked cross-entropy against independent scalar loss and finite-difference
references, zero-weight columns, scalar upstream scaling, extreme logits, shape
and weight validation, and legacy Adam parity on CPU and CUDA.

`LayerNormLayer<D>(features, columns)` supports both standalone and layer-list
use. `forward(input)` returns a new output matrix by value and updates the
normalization statistics needed for backward and JVP; it does not store that
output in `value()`. `eval(input)` computes the same result and moves it into the
layer's output cache for `Sequential` and `value()`. Each entry point performs a
new normalization. Backward and JVP always use the statistics from the most
recent successful normalization, so do not mix entries between a network's
`eval` and its backward pass.

Standalone calls remain `backward(std::move(upstream), enabled)`,
`backward_update(std::move(upstream), enabled)`, and `jvp(tangent)`; layer-list
calls use the usual `backward(input, std::move(upstream))` and
`jvp(input, tangent)` overloads. Gradient-only backward accumulates gamma and
beta gradients; updating backward keeps the previous Adam behavior. Freezing a
layer disables its parameter gradients and updates while preserving input
gradients. The public `trainable` flag can also disable gamma and beta training.

The previous `LayerNorm<D>` name is removed without an alias. Rename standalone
instances to `LayerNormLayer<D>`, and replace wrapper member access such as
`normalization.norm.gamma` with `normalization.gamma`. Parameter and optimizer
checkpoint names remain unchanged, including the normalization fields inside
Transformer checkpoints.

## Examples and custom networks

Layer-based training demos now separate gradient calculation from updates.
They explicitly set `options.weight_decay = 0` to preserve their original Adam
training rule; `AdamWOptions` otherwise defaults to nonzero weight decay.
Existing learning-rate schedules use `optimizer.set_lr(...)`.

When targets change every batch, a persistent prediction network can be combined
with a temporary loss head:

```cpp
Sequential<float> network{&linear};
AdamWOptions options;
options.weight_decay = 0;
AdamW<float> optimizer(network.parameters(), options);

// inputs and targets remain alive through this step.
optimizer.zero_grad();
const auto& prediction = network.eval(inputs);
LossLayer<float> loss(inputs.num_col(), targets);
network.backward(loss.backward(prediction, Matrix<float>::ones(1, 1)));
optimizer.step();
```

Call `loss.eval(prediction)` to read the loss via `loss.value()` when needed.
The loss backward uses the supplied prediction directly and does not repeat the
network's forward pass. Finish backward before running validation or inference
through the same layers, because those calls replace the forward caches.

The CNN rectified-flow example has skip connections and therefore keeps its
explicit graph traversal: `net.eval(input)`, `net.backward(input, upstream)` and
`optimizer.step()`. Its backward method accumulates the two skip-branch gradients
before propagating into each encoder layer. `demo_word2vec` and the standalone
WebAssembly examples use their own matrix calculations rather than the layer
training API. `demo_gui` remains an optional source with external GUI dependencies.
