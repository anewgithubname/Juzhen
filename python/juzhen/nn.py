"""Neural networks: layers, Sequential, losses.

The layers are Juzhen's C++ layers (ml/layer.hpp) with their hand-written
backward passes; there is no computation graph. A training step is::

    opt.zero_grad()
    pred = net(X_i)                               # forward, keeps X_i for backward
    loss = nn.softmax_cross_entropy(pred, Y_i)    # value and gradient
    net.backward(loss.grad)                       # gradients of every parameter
    opt.step()                                    # AdamW update, in C++

Data is one sample per row. Inside, the C++ layers see the same storage as
features x batch, the layout they were written for.

A layer object here is a description; Sequential creates the C++ layer on
its device once the input size is known.
"""

from typing import NamedTuple

import numpy as np

import juzhen as jz

__all__ = [
    "Layer", "Dense", "Linear", "LayerNorm", "Conv2d", "ConvTranspose2d", "Transformer",
    "Sequential", "Loss", "softmax_cross_entropy", "mse", "error_rate", "one_hot",
    "save_checkpoint", "load_checkpoint",
]


class Layer:
    """Base class of the layer descriptions.

    Subclasses define in_features (None if it comes from the previous layer),
    out_features(n_in) and create(core, n_in), which makes the C++ layer.
    """

    in_features = None

    def out_features(self, n_in):
        raise NotImplementedError

    def create(self, core, n_in):
        raise NotImplementedError


class Dense(Layer):
    """Fully connected layer: activation(W x + b).

    ``Dense(out)`` takes its input size from the previous layer; ``Dense(in, out)``
    gives it. activation is "tanh" (the C++ Layer<D>, the default), "relu"
    (ReluLayer<D>) or None (LinearLayer<D>). Weights start as N(0, 0.001^2).
    """

    _classes = {"tanh": "Dense", "relu": "ReLUDense", None: "Linear"}

    def __init__(self, *features, activation="tanh"):
        if len(features) == 1:
            self.in_features, self.out = None, int(features[0])
        elif len(features) == 2:
            self.in_features, self.out = int(features[0]), int(features[1])
        else:
            raise TypeError("Dense(out_features) or Dense(in_features, out_features)")
        if activation not in self._classes:
            raise ValueError(f"activation must be 'tanh', 'relu' or None, not {activation!r}")
        self.activation = activation

    def out_features(self, n_in):
        return self.out

    def create(self, core, n_in):
        return getattr(core, self._classes[self.activation])(self.out, n_in)

    def __repr__(self):
        inner = f"{self.in_features}, {self.out}" if self.in_features else f"{self.out}"
        return f"Dense({inner}, activation={self.activation!r})"


class Linear(Dense):
    """Fully connected layer without activation: W x + b."""

    def __init__(self, *features):
        super().__init__(*features, activation=None)

    def __repr__(self):
        inner = f"{self.in_features}, {self.out}" if self.in_features else f"{self.out}"
        return f"Linear({inner})"


class LayerNorm(Layer):
    """Normalizes each sample over its features, then scales and shifts."""

    def __init__(self, features=None, batch_size=1):
        self.in_features, self.batch_size = features, batch_size

    def out_features(self, n_in):
        return n_in

    def create(self, core, n_in):
        return core.LayerNorm(n_in, self.batch_size)


class Conv2d(Layer):
    """2-D convolution (optionally followed by ReLU) on a fixed batch size.

    Each sample is an image flattened channel by channel (C x H x W, as numpy
    gives for an N x C x H x W array). The output is the C_out x H_out x W_out
    image, flattened the same way.
    """

    _core_name = "Conv2d"

    def __init__(self, in_channels, out_channels, kernel_size, image_size, batch_size,
                 padding=0, stride=1, relu=True):
        self.c_in, self.c_out = in_channels, out_channels
        self.kh, self.kw = _pair(kernel_size)
        self.h, self.w = _pair(image_size)
        self.batch_size, self.padding, self.stride, self.relu = batch_size, padding, stride, relu
        self.in_features = in_channels * self.h * self.w

    def output_size(self):
        return ((self.h + 2 * self.padding - self.kh) // self.stride + 1,
                (self.w + 2 * self.padding - self.kw) // self.stride + 1)

    def out_features(self, n_in):
        h, w = self.output_size()
        return self.c_out * h * w

    def create(self, core, n_in):
        if not hasattr(core, self._core_name):
            raise ValueError(f"{self._core_name} is not available on device "
                             f"'{core.__name__.rsplit('.', 1)[-1]}' in this build")
        return getattr(core, self._core_name)(self.batch_size, self.c_in, self.h, self.w, self.c_out,
                                              self.kh, self.kw, self.padding, self.stride, self.relu)


class ConvTranspose2d(Conv2d):
    """Transposed 2-D convolution (upsampling), the counterpart of Conv2d."""

    _core_name = "ConvTranspose2d"

    def output_size(self):
        return ((self.h - 1) * self.stride - 2 * self.padding + self.kh,
                (self.w - 1) * self.stride - 2 * self.padding + self.kw)


class Transformer(Layer):
    """Pre-norm transformer block on batch_size sequences of seq_len tokens.

    Each row is one token of d_model features; the rows of a batch are the
    tokens of its sequences, one sequence after another.
    """

    def __init__(self, d_model, d_k, d_ff, seq_len, batch_size, heads=1, causal=True):
        self.in_features = d_model
        self.args = (d_model, d_k, d_ff, seq_len, batch_size, heads, causal)

    def out_features(self, n_in):
        return self.in_features

    def create(self, core, n_in):
        return core.Transformer(*self.args)


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


class Sequential:
    """Layers applied one after another, input layer first.

    The C++ layers are created when the input size is known: right away if
    the first layer gives in_features, otherwise on the first call.
    """

    def __init__(self, *layers, device="cpu"):
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = tuple(layers[0])
        if not layers:
            raise ValueError("Sequential needs at least one layer")
        self.layers = list(layers)
        self.device = device
        self._core = jz._device(device)
        self._impls = None
        self._net = None
        if self.layers[0].in_features is not None:
            self.build(self.layers[0].in_features)

    @property
    def built(self):
        return self._net is not None

    def build(self, in_features):
        """Create the C++ layers for inputs of in_features features."""
        if self.built:
            if in_features != self.in_features:
                raise ValueError(f"the network takes {self.in_features} features, got {in_features}")
            return self
        impls, n = [], in_features
        for layer in self.layers:
            if layer.in_features is not None and layer.in_features != n:
                raise ValueError(f"{layer!r} takes {layer.in_features} features, but gets {n}")
            impls.append(layer.create(self._core, n))
            n = layer.out_features(n)
        self.in_features, self.out_features = in_features, n
        self._impls = impls  # the Python objects own the C++ layers
        self._net = self._core.Network(impls)
        return self

    def _check(self, x):
        if not isinstance(x, jz.Matrix):
            raise TypeError(f"expected a juzhen matrix, got {type(x).__name__}; use jz.array()")
        if x.device != self.device:
            raise ValueError(f"the input is on '{x.device}' but the network is on '{self.device}'; "
                             f"use x.to('{self.device}')")
        self.build(x.shape[1])

    def __call__(self, x):
        """The output for a batch x (batch x features); keeps x for backward()."""
        self._check(x)
        return self._net.forward(x)

    forward = __call__

    def predict(self, x):
        """The output without keeping x (no backward() afterwards)."""
        self._check(x)
        return self._net.predict(x)

    def backward(self, grad):
        """Back-propagate grad, the loss gradient for the last output.

        Adds to the gradient of every parameter and returns the gradient with
        respect to the input.
        """
        return self._net.backward(grad)

    def jvp(self, x, tangent):
        """J(x) @ tangent: how the output changes when x moves along tangent."""
        self._check(x)
        return self._net.jvp(x, tangent)

    def parameters(self):
        """The trainable parameters (named "layer<k>.<name>"), for an optimizer."""
        if not self.built:
            raise RuntimeError("the network is not built yet: give the first layer its input size, "
                               "call build(in_features) or call the network once")
        return self._net.parameters()

    def __repr__(self):
        inner = ",\n".join(f"    {layer!r}" for layer in self.layers)
        return f"Sequential(\n{inner},\n    device={self.device!r})"

    # ------------------------------------------------------------------
    # The five-line interface: fit() and evaluate()

    def fit(self, X, Y, epochs=1, batch_size=32, lr=1e-3, loss="cross_entropy",
            weight_decay=0.0, shuffle=True, verbose=True):
        """Train on X (samples x features) and Y (samples x outputs) with AdamW.

        loss is "cross_entropy" (Y one-hot) or "mse". Returns the mean loss of
        each epoch.
        """
        loss_fn = _loss_function(loss)
        loader = jz.data.DataLoader(X, Y, batch_size=batch_size, shuffle=shuffle,
                                    drop_last=True, device=self.device)
        self.build(loader.features)
        if not hasattr(self, "optimizer"):
            self.optimizer = jz.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer.lr = lr
        history = []
        for epoch in range(epochs):
            total = None
            for X_i, Y_i in loader:
                self.optimizer.zero_grad()
                result = loss_fn(self(X_i), Y_i)
                self.backward(result.grad)
                self.optimizer.step()
                total = result.value if total is None else total + result.value
            history.append(total.item() / len(loader))  # one GPU wait per epoch
            if verbose:
                print(f"epoch {epoch + 1}/{epochs}: loss {history[-1]:.4f}")
        return history

    def evaluate(self, X, Y, batch_size=1000, loss="cross_entropy"):
        """Mean loss (and accuracy for cross_entropy) on X, Y."""
        loss_fn = _loss_function(loss)
        loader = jz.data.DataLoader(X, Y, batch_size=batch_size, shuffle=False,
                                    drop_last=False, device=self.device)
        total_loss, total_errors, n = 0.0, 0.0, 0
        for X_i, Y_i in loader:
            scores = self.predict(X_i)
            b = X_i.shape[0]
            total_loss += loss_fn(scores, Y_i).value.item() * b
            if loss == "cross_entropy":
                total_errors += error_rate(scores, Y_i).item() * b
            n += b
        result = {"loss": total_loss / n}
        if loss == "cross_entropy":
            result["accuracy"] = 1 - total_errors / n
        return result


# ----------------------------------------------------------------------
# Losses (ml/loss.hpp). Each returns the value (1 x 1, averaged over the
# batch) and its gradient with respect to the prediction.


class Loss(NamedTuple):
    value: object  # 1 x 1 matrix
    grad: object   # same shape as the prediction

    def item(self):
        """The loss as a number (waits for the GPU)."""
        return self.value.item()


def _core_of(m):
    return jz._device(m.device)


def softmax_cross_entropy(logits, labels):
    """Softmax cross-entropy; labels are one-hot rows (or probabilities)."""
    return Loss(*_core_of(logits).softmax_cross_entropy(logits, labels))


def mse(prediction, target):
    """Squared error summed over outputs, averaged over the batch."""
    return Loss(*_core_of(prediction).mse(prediction, target))


def error_rate(scores, labels):
    """Fraction of rows whose largest score is not the labelled class (1 x 1)."""
    return _core_of(scores).error_rate(scores, labels)


def _loss_function(name):
    losses = {"cross_entropy": softmax_cross_entropy, "mse": mse}
    if name not in losses:
        raise ValueError(f"loss must be one of {sorted(losses)}, not {name!r}")
    return losses[name]


def one_hot(labels, num_classes, device="cpu"):
    """Integer class labels (length N) as an N x num_classes one-hot matrix."""
    labels = np.asarray(labels.numpy() if isinstance(labels, jz.Matrix) else labels).astype(np.int64).ravel()
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(f"labels must be in [0, {num_classes})")
    out = np.zeros((labels.size, num_classes), dtype=np.float32)
    out[np.arange(labels.size), labels] = 1
    return jz.array(out, device=device)


# ----------------------------------------------------------------------
# Checkpoints (ml/checkpoint.hpp): parameters, Adam states and the CPU
# random number generator, written atomically.


def save_checkpoint(net, path, epoch=0, step=0, data_position=0):
    """Save net's parameters and optimizer states to path."""
    if not net.built:
        raise RuntimeError("build the network before saving it")
    net._core.save_checkpoint(net._net, str(path), epoch, step, data_position)


def load_checkpoint(net, path):
    """Load a checkpoint saved from the same architecture; returns its progress."""
    if not net.built:
        raise RuntimeError("build the network before loading into it")
    epoch, step, data_position = net._core.load_checkpoint(net._net, str(path))
    return {"epoch": epoch, "step": step, "data_position": data_position}
