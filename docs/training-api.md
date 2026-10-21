# Training API

`backward` computes input and parameter gradients without updating weights.
`backward_update` retains the previous combined backpropagation and Adam update.
Existing calls to `backprop(net, x)` migrate to `backward_update(net, x)`;
existing layer calls that update weights migrate to `layer.backward_update(...)`.

`Sequential` borrows a native Juzhen list, in its established output-first order:

```cpp
#include "ml/layer.hpp"
#include "ml/optimizer.hpp"

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
`Juzhen::parameters(net)`. The gradient-only path currently supports
`LinearLayer`, `TransformerLayer`, and `LossLayer`; other layer types retain
`backward_update` and must explicitly implement gradient-only backward before
joining this path. `LayerNorm` also exposes gradient-only backward and parameters.
Networks containing unsupported layers are rejected before gradient traversal.

`AdamW` supports CPU `float` and CUDA `CUDAfloat`, parameter freezing, shared
parameter registrations, learning-rate changes and optional global gradient
clipping. It borrows existing layer Adam states so checkpoint storage stays
compatible. Weight decay excludes parameters registered with `decay=false`.

The `demo_training` target provides a complete regression example.
The `training_api` test checks finite-difference gradients, native-list and
Sequential equivalence, accumulation, legacy Adam equivalence, error handling,
gradient clipping and normalization against scalar references.
