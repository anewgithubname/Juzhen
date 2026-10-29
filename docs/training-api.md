# Training API

`backward` computes input and parameter gradients without updating weights; an
optimizer (`AdamW` in `ml/optimizer.hpp`) then applies the update. This is the
only training path. The earlier `backward_update`, which updated each layer's
weights during backpropagation, has been removed together with `adam_update`,
`setlr` and the per-layer learning-rate fields.

`Sequential` borrows a native Juzhen list, in its established output-first order.
The network contains only layers; the loss is a function applied to its output:

```cpp
#include "ml/layer.hpp"
#include "ml/loss.hpp"
#include "ml/optimizer.hpp"
using namespace Juzhen;

LinearLayer<float> linear(1, 2, batch_size);
Sequential<float> network{&linear};
AdamWOptions options;
options.weight_decay = 0;  // Adam without weight decay.
AdamW<float> optimizer(network.parameters(), options);

for (int step = 0; step < steps; ++step) {
    optimizer.zero_grad();
    auto [loss, dprediction] = mse(network.eval(inputs), targets);
    network.backward(std::move(dprediction));
    optimizer.step();
}
```

`inputs` is an lvalue matrix with two rows and `batch_size` columns; `targets`
has one row and `batch_size` columns. Both matrices and all layers must outlive
their use by the network.

## Loss functions

`ml/loss.hpp` provides each loss as a plain function. It takes the network output
(one column per sample) and returns a `LossResult` holding `value`, the scalar
loss as a 1x1 matrix that stays on the device, and `grad`, the gradient with
respect to that output, ready to pass to `backward`:

| Function | Value | Gradient |
|---|---|---|
| `mse(prediction, target)` | sum of squared errors / N | 2 (prediction - target) / N |
| `softmax_cross_entropy(logits, labels)` | mean cross-entropy over the batch | (softmax(logits) - labels) / N |
| `masked_softmax_cross_entropy(logits, labels, weights)` | sum over columns of w × cross-entropy | (softmax(logits) - labels) × w |
| `error_rate(scores, labels)` | fraction of misclassified columns | none; evaluation only |

N is the number of columns. Labels are one-hot columns or normalized probability
distributions. Masked weights have shape (1, N), must be finite and nonnegative,
and already contain any desired normalization: no batch averaging is applied, and
a zero weight removes its column from both the loss and the gradient. When only
the gradient is needed, write `network.backward(softmax_cross_entropy(logits, labels).grad)`.

`eval` returns the final layer value. `backward(upstream)` takes the gradient of
the loss with respect to that value, for example a loss function's `grad`; scaling
it scales every gradient. Backward uses cached values and does not reevaluate layers. Repeated backward calls accumulate
gradients until `optimizer.zero_grad()`. Input and cached layer values must remain
unchanged between evaluation and backward. A failed evaluation invalidates the
wrapper's cached input. The wrapper does not own layers or the input.

The list-level forms remain available as `Juzhen::forward(net, inputs)`,
`Juzhen::backward(net, inputs, std::move(upstream))`, and
`Juzhen::parameters(net)`. The gradient-only path supports the fully connected
`Layer` (tanh), `ReluLayer`, `LinearLayer`, `ConvLayer`, `ConvTransLayer`,
`TransformerLayer` and `LayerNormLayer`. `LayerNormLayer<D>` is the single
normalization implementation used both inside `TransformerLayer` and directly
inside `Sequential`.

Direct subclasses of `Layer` must explicitly implement and advertise
gradient-only backward before joining this path. Subclasses that change an
adapted layer's computation must also provide the corresponding backward
implementation. Networks containing unsupported layers are rejected before
gradient traversal.

Losses used to be layers (`LossLayer`, `LogisticLayer`, `MaskedCELayer`,
`ZeroOneLayer`) placed at the front of the network list; they have been replaced
by the functions above. Checkpoints saved from a network list that contained one
of those layers no longer load, because the list no longer includes the loss.
Convolution inputs, upstream gradients and JVP tangents must match the configured
dimensions. GPU convolution paths require non-transposed tensors, including
weights and biases; CPU convolution supports matrix transpose views.

`AdamW` supports CPU `float`, CUDA `CUDAfloat`, Metal `MPSfloat`, and ROCm
`ROCMfloat`, with parameter freezing, shared parameter registrations,
learning-rate changes and optional global gradient clipping. CPU and CUDA use
their optimized updates; Metal and ROCm use device matrix operations. ROCm has
been run on hardware; Metal has not. The optimizer keeps its moments in the
layers' Adam states, so they are saved and restored by checkpoints. Weight decay
excludes parameters registered with `decay=false`.

The `demo_training` target provides a complete regression example.
The `training_api` test checks finite-difference gradients, native-list and
Sequential equivalence, accumulation, error handling,
gradient clipping and normalization against scalar references.

The `all_layer_training_api` regression exercises activation, convolution and transpose
convolution layers through the explicit training API. `loss_functions` checks
every loss function against independent FP64 references and finite differences,
including soft labels, extreme logits, zero-weight columns, shape and weight
validation, and a short training run, on CPU, CUDA and ROCm.

`LayerNormLayer<D>(features, columns)` supports both standalone and layer-list
use. `forward(input)` returns a new output matrix by value and updates the
normalization statistics needed for backward and JVP; it does not store that
output in `value()`. `eval(input)` computes the same result and moves it into the
layer's output cache for `Sequential` and `value()`. Each entry point performs a
new normalization. Backward and JVP always use the statistics from the most
recent successful normalization, so do not mix entries between a network's
`eval` and its backward pass.

Standalone calls are `backward(std::move(upstream), enabled)` and
`jvp(tangent)`; layer-list calls use the usual `backward(input, std::move(upstream))`
and `jvp(input, tangent)` overloads. Backward accumulates gamma and beta
gradients. Freezing a layer disables its parameter gradients while preserving
input gradients. The public `trainable` flag can also disable gamma and beta training.

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

Because a loss is a function of the prediction and the targets, targets can
change every batch without constructing anything:

```cpp
// inputs and targets remain alive through this step.
optimizer.zero_grad();
const auto& prediction = network.eval(inputs);
auto [loss, dprediction] = softmax_cross_entropy(prediction, targets);
network.backward(std::move(dprediction));
optimizer.step();
```

The loss uses the supplied prediction directly and does not repeat the network's
forward pass. Finish backward before running validation or inference through the
same layers, because those calls replace the forward caches. To evaluate, apply
a loss function to the network output, for example
`error_rate(forward(net, test_inputs), test_labels)`.

The CNN rectified-flow example has skip connections and therefore keeps its
explicit graph traversal: `net.eval(input)`, `net.backward(input, upstream)` and
`optimizer.step()`. Its backward method accumulates the two skip-branch gradients
before propagating into each encoder layer. `demo_word2vec` and the standalone
WebAssembly examples use their own matrix calculations rather than the layer
training API. `demo_gui` remains an optional source with external GUI dependencies.
