"""Neural networks: layers, Sequential, losses.

The layers are Juzhen's C++ layers (ml/layer.hpp, ml/embedding.hpp) with
their hand-written backward passes; there is no computation graph. A
training step is::

    opt.zero_grad()
    out = net(x)                                  # forward, keeps x for backward
    loss = nn.softmax_cross_entropy(out, y)       # value and gradient
    net.backward(loss.grad)                       # gradients of every parameter
    opt.step()                                    # AdamW update, in C++

Data is one sample per row. Inside, the C++ layers see the same storage as
features x batch, the layout they were written for.

Numbers are juzhen matrices. Whole numbers that name things (token ids for
an Embedding, class numbers for a loss) are numpy integer arrays or lists.

A layer object here is a description; Sequential creates the C++ layer on
its device once the input size is known.
"""

import copy
import math
import warnings

import numpy as np

import juzhen as jz

__all__ = [
    "Layer", "Dense", "Linear", "LayerNorm", "Conv2d", "ConvTranspose2d", "Transformer",
    "Embedding", "PositionEmbedding",
    "Sequential", "Loss", "softmax_cross_entropy", "masked_softmax_cross_entropy", "mse",
    "error_rate", "one_hot",
    "save_checkpoint", "load_checkpoint",
]


def _init_args(init, what):
    """init as the (rule, std) pair of the C++ Init: "auto" or a number."""
    if isinstance(init, str):
        if init != "auto":
            raise ValueError(f"{what}: init must be 'auto' (the layer's own rule) or a number (the "
                             f"standard deviation), not {init!r}")
        return init, 0.0
    if isinstance(init, bool) or not isinstance(init, (int, float, np.integer, np.floating)):
        raise TypeError(f"{what}: init must be 'auto' (the layer's own rule) or a number (the "
                        f"standard deviation), got {type(init).__name__}")
    if not math.isfinite(init) or init < 0:
        raise ValueError(f"{what}: init is a standard deviation and must be 0 or more, got {init}")
    return "normal", float(init)


def _size(value, what):
    """A positive whole number given as an int (a float such as 64.0 is also accepted)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
            or value != int(value) or value < 1:
        raise ValueError(f"{what} must be a positive whole number, got {value!r}")
    return int(value)


class Layer:
    """Base class of the layer descriptions.

    Subclasses define in_features (None if it comes from the previous layer),
    out_features(n_in) and create(core, n_in), which makes the C++ layer.
    """

    in_features = None
    takes_ids = False  # the layer reads token ids (only Embedding)
    rows = None        # the exact number of rows a batch must have, when the layer fixes it
    seq_len = None     # the sequence length the layer was made for, if any

    def out_features(self, n_in):
        raise NotImplementedError

    def create(self, core, n_in):
        raise NotImplementedError

    def _with_batch(self, batch_size):
        """A copy of this description for another batch size (see Sequential.copy)."""
        return copy.copy(self)


class Dense(Layer):
    """Fully connected layer: activation(W x + b).

    ``Dense(out)`` takes its input size from the previous layer; ``Dense(in, out)``
    gives it. activation is "tanh" (the C++ Layer<D>, the default), "relu"
    (ReluLayer<D>) or None (LinearLayer<D>).

    Weights start as N(0, 2 / in) in front of a ReLU and N(0, 1 / in)
    otherwise; biases start at zero. init=s gives N(0, s^2) instead (0 gives
    zeros).
    """

    _classes = {"tanh": "Dense", "relu": "ReLUDense", None: "Linear"}
    tied = False

    def __init__(self, *features, activation="tanh", init="auto"):
        # Dense(256, "relu") means Dense(256, activation="relu")
        if len(features) >= 2 and (isinstance(features[-1], str) or features[-1] is None):
            activation, features = features[-1], features[:-1]
        if len(features) == 1:
            self.in_features, self.out = None, _size(features[0], "Dense: the number of outputs")
        elif len(features) == 2:
            self.in_features = _size(features[0], "Dense: the number of inputs")
            self.out = _size(features[1], "Dense: the number of outputs")
        else:
            raise TypeError("Dense(out_features) or Dense(in_features, out_features)")
        if activation not in self._classes:
            raise ValueError(f"activation must be 'tanh', 'relu' or None, not {activation!r}")
        self.activation = activation
        self.init = init
        self._init = _init_args(init, type(self).__name__)

    def out_features(self, n_in):
        return self.out

    def create(self, core, n_in):
        return getattr(core, self._classes[self.activation])(self.out, n_in, *self._init)

    def _extras(self):
        return "" if self.init == "auto" else f", init={self.init!r}"

    def __repr__(self):
        inner = f"{self.in_features}, {self.out}" if self.in_features else f"{self.out}"
        return f"Dense({inner}, activation={self.activation!r}{self._extras()})"


class Linear(Dense):
    """Fully connected layer without activation: W x + b.

    ``Linear(vocab, tied=True)`` is the output layer of a language model with
    tied weights: it has no weight or bias of its own and computes x @ E.T
    with E the table of the network's Embedding (C++ TiedLinearLayer<D>).
    """

    def __init__(self, *features, tied=False, init="auto"):
        super().__init__(*features, activation=None, init=init)
        self.tied = bool(tied)
        if self.tied and init != "auto":
            raise ValueError("Linear(tied=True) has no weight of its own to initialize; "
                             "give init to the Embedding instead")

    def create(self, core, n_in, embedding=None):
        if self.tied:
            return core.TiedLinear(embedding)
        return super().create(core, n_in)

    def __repr__(self):
        inner = f"{self.in_features}, {self.out}" if self.in_features else f"{self.out}"
        return f"Linear({inner}{', tied=True' if self.tied else ''}{self._extras()})"


class Embedding(Layer):
    """Turns token ids into vectors: row i of the output is the table row of id i.

    The first layer of a network; the network then takes whole numbers in
    0 .. vocab - 1 (a numpy integer array or a list) instead of a matrix. The
    table ``weight`` is vocab x dim and starts as N(0, init^2) (C++ EmbeddingLayer<D>).
    """

    takes_ids = True

    def __init__(self, vocab, dim, init=0.02):
        self.vocab, self.dim = _size(vocab, "Embedding: vocab"), _size(dim, "Embedding: dim")
        self.init = init
        self._init = _init_args(init, "Embedding")

    def out_features(self, n_in):
        return self.dim

    def create(self, core, n_in):
        return core.Embedding(self.vocab, self.dim, *self._init)

    def __repr__(self):
        return f"Embedding({self.vocab}, {self.dim}{'' if self.init == 0.02 else f', init={self.init!r}'})"


class PositionEmbedding(Layer):
    """Adds a learned vector to each token according to its position in its sequence.

    The rows of a batch are sequences of seq_len tokens, one after another;
    any number of sequences is accepted. The table ``weight`` is seq_len x
    features (C++ PositionEmbeddingLayer<D>). It also works after layers that
    are not an Embedding.
    """

    def __init__(self, seq_len, init=0.02):
        self.seq_len = _size(seq_len, "PositionEmbedding: seq_len")
        self.init = init
        self._init = _init_args(init, "PositionEmbedding")

    def out_features(self, n_in):
        return n_in

    def create(self, core, n_in):
        return core.PositionEmbedding(self.seq_len, n_in, *self._init)

    def __repr__(self):
        return f"PositionEmbedding({self.seq_len}{'' if self.init == 0.02 else f', init={self.init!r}'})"


class LayerNorm(Layer):
    """Normalizes each sample over its features, then scales and shifts.

    Works with any number of rows.
    """

    def __init__(self, features=None, batch_size=None):
        if batch_size is not None:
            raise TypeError("LayerNorm no longer takes batch_size: it works with any number of rows; "
                            "write nn.LayerNorm()")
        self.in_features = None if features is None else _size(features, "LayerNorm: features")

    def out_features(self, n_in):
        return n_in

    def create(self, core, n_in):
        return core.LayerNorm(n_in)

    def __repr__(self):
        return f"LayerNorm({self.in_features or ''})"


class Conv2d(Layer):
    """2-D convolution (optionally followed by ReLU) on a fixed batch size.

    Each sample is an image flattened channel by channel (C x H x W, as numpy
    gives for an N x C x H x W array). The output is the C_out x H_out x W_out
    image, flattened the same way. init as in Dense, with the kernel's own
    input count (in_channels x kernel height x kernel width).
    """

    _core_name = "Conv2d"

    def __init__(self, in_channels, out_channels, kernel_size, image_size, batch_size,
                 padding=0, stride=1, relu=True, init="auto"):
        self.c_in, self.c_out = in_channels, out_channels
        self.kh, self.kw = _pair(kernel_size)
        self.h, self.w = _pair(image_size)
        self.batch_size, self.padding, self.relu = batch_size, padding, relu
        self.stride = _size(stride, f"{type(self).__name__}: stride")
        self.in_features = in_channels * self.h * self.w
        self.rows = _size(batch_size, f"{type(self).__name__}: batch_size")
        self.init = init
        self._init = _init_args(init, type(self).__name__)
        h, w = self.output_size()
        if h < 1 or w < 1:
            raise ValueError(f"{type(self).__name__}: a {self.kh} x {self.kw} kernel does not fit a "
                             f"{self.h} x {self.w} image with padding={padding}")

    def output_size(self):
        return ((self.h + 2 * self.padding - self.kh) // self.stride + 1,
                (self.w + 2 * self.padding - self.kw) // self.stride + 1)

    def _with_batch(self, batch_size):
        other = copy.copy(self)
        other.batch_size = other.rows = _size(batch_size, "copy: batch_size")
        return other

    def out_features(self, n_in):
        h, w = self.output_size()
        return self.c_out * h * w

    def create(self, core, n_in):
        if not hasattr(core, self._core_name):
            raise ValueError(f"{self._core_name} is not available on device "
                             f"'{core.__name__.rsplit('.', 1)[-1]}' in this build")
        return getattr(core, self._core_name)(self.batch_size, self.c_in, self.h, self.w, self.c_out,
                                              self.kh, self.kw, self.padding, self.stride, self.relu,
                                              *self._init)

    def __repr__(self):
        kernel = self.kh if self.kh == self.kw else (self.kh, self.kw)
        image = self.h if self.h == self.w else (self.h, self.w)
        return (f"{type(self).__name__}({self.c_in}, {self.c_out}, {kernel}, image_size={image}, "
                f"batch_size={self.batch_size}, padding={self.padding}, stride={self.stride}, "
                f"relu={self.relu}{'' if self.init == 'auto' else f', init={self.init!r}'})")


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
        names = ("d_model", "d_k", "d_ff", "seq_len", "batch_size", "heads")
        d_model, d_k, d_ff, seq_len, batch_size, heads = (
            _size(v, f"Transformer: {n}") for v, n in zip((d_model, d_k, d_ff, seq_len, batch_size, heads), names))
        if d_k % heads:
            raise ValueError(f"Transformer: heads ({heads}) must divide d_k ({d_k})")
        self.in_features = d_model
        self.seq_len, self.batch_size = seq_len, batch_size
        self.rows = seq_len * batch_size
        self.args = (d_model, d_k, d_ff, seq_len, batch_size, heads, bool(causal))

    def out_features(self, n_in):
        return self.in_features

    def create(self, core, n_in):
        return core.Transformer(*self.args)

    def _with_batch(self, batch_size):
        other = copy.copy(self)
        other.batch_size = _size(batch_size, "copy: batch_size")
        other.rows = other.seq_len * other.batch_size
        other.args = self.args[:4] + (other.batch_size,) + self.args[5:]
        return other

    def __repr__(self):
        d_model, d_k, d_ff, seq_len, batch_size, heads, causal = self.args
        return (f"Transformer({d_model}, {d_k}, {d_ff}, seq_len={seq_len}, batch_size={batch_size}, "
                f"heads={heads}, causal={causal})")


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


class Sequential:
    """Layers applied one after another, input layer first.

    The C++ layers are created when the input size is known: right away if
    the first layer gives in_features (or is an Embedding), otherwise on the
    first call.
    """

    def __init__(self, *layers, device="cpu"):
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = tuple(layers[0])
        if not layers:
            raise ValueError("Sequential needs at least one layer")
        for k, layer in enumerate(layers):
            if isinstance(layer, Sequential):
                raise TypeError(f"layer {k} is a Sequential. A network is one list of layers: write "
                                "nn.Sequential(*inner.layers, ...) to reuse the descriptions "
                                "(the new network gets its own weights)")
            if not isinstance(layer, Layer):
                raise TypeError(f"layer {k} is a {type(layer).__name__}, not a layer description such as "
                                "nn.Dense(64)")
            for j in range(k):
                if layers[j] is layer:
                    raise ValueError(f"the same {layer!r} object is layer {j} and layer {k}. Each place needs "
                                     "its own layer object (they would not share weights): for example "
                                     "[nn.Dense(64) for _ in range(2)]")
        self.layers = list(layers)
        self._core = jz._device(device)
        self.device = "cpu" if self._core is jz._core.cpu else "gpu"  # "cuda", "rocm", "metal" mean "gpu"
        self._impls = None
        self._net = None
        self._out_shape = None

        self.takes_ids = self.layers[0].takes_ids
        for k, layer in enumerate(self.layers[1:], 1):
            if layer.takes_ids:
                raise ValueError(f"{layer!r} is layer {k}, but it reads token ids, which only the first "
                                 "layer of a network receives; put it first")
            if getattr(layer, "tied", False) and not isinstance(self.layers[0], Embedding):
                raise ValueError(f"{layer!r} shares the table of the network's Embedding, but the first "
                                 f"layer is {self.layers[0]!r}; start the network with nn.Embedding(...)")
        # layers with a fixed batch must agree on the number of rows; so must the sequence lengths
        self.batch_rows = self._common("rows", "rows per batch")
        self.seq_len = self._common("seq_len", "tokens per sequence")
        if self.batch_rows and self.seq_len and self.batch_rows % self.seq_len:
            raise ValueError(f"{self.batch_rows} rows per batch do not fill whole sequences of "
                             f"{self.seq_len} tokens")
        if self.takes_ids:
            first_block = next((k for k, l in enumerate(self.layers) if isinstance(l, Transformer)), None)
            if first_block is not None and not any(isinstance(l, PositionEmbedding)
                                                   for l in self.layers[:first_block]):
                warnings.warn("this network sends token vectors into a Transformer without positions, so "
                              "the model cannot tell where in the sequence a token is; add "
                              "nn.PositionEmbedding(seq_len) after the Embedding", stacklevel=2)
            self.build(None)
        elif self.layers[0].in_features is not None:
            self.build(self.layers[0].in_features)

    def _common(self, name, what):
        found = [(getattr(layer, name), layer) for layer in self.layers if getattr(layer, name) is not None]
        for value, layer in found[1:]:
            if value != found[0][0]:
                raise ValueError(f"{found[0][1]!r} and {layer!r} need different numbers of {what} "
                                 f"({found[0][0]} and {value}); the layers of one network must agree")
        return found[0][0] if found else None

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
            if getattr(layer, "tied", False):
                first = self.layers[0]
                if (layer.out, n) != (first.vocab, first.dim):
                    raise ValueError(f"{layer!r} uses the table of {first!r} as its weight, so it needs "
                                     f"{first.vocab} outputs and {first.dim} inputs; it has {layer.out} "
                                     f"outputs and gets {n} inputs")
                impls.append(layer.create(self._core, n, embedding=impls[0]))
            else:
                impls.append(layer.create(self._core, n))
            n = layer.out_features(n)
        self.in_features, self.out_features = in_features, n
        self._impls = impls  # the Python objects own the C++ layers
        self._net = self._core.Network(impls)
        return self

    # ------------------------------------------------------------------
    # Input checks: every message says what arrived and what to do

    def _fixed_rows(self):
        """Why the batch size is fixed, in words."""
        layer = next(l for l in self.layers if l.rows is not None)
        if self.seq_len:
            return (f"exactly {self.batch_rows} rows per batch ({self.batch_rows // self.seq_len} sequences "
                    f"of {self.seq_len} tokens, fixed by {layer!r})")
        return f"exactly {self.batch_rows} rows per batch (fixed by {layer!r})"

    def _check_rows(self, rows):
        if self.batch_rows and rows != self.batch_rows:
            raise ValueError(f"the network takes {self._fixed_rows()}, got {rows}; for another batch size "
                             "use net.copy(batch_size=...)")
        if self.seq_len and rows % self.seq_len:
            raise ValueError(f"{rows} rows do not fill whole sequences of {self.seq_len} tokens")

    def _check(self, x):
        if self.takes_ids:
            return self._ids(x)
        if not isinstance(x, jz.Matrix):
            raise TypeError(f"expected a juzhen matrix, got {type(x).__name__}; use jz.array()")
        if x.device != self.device:
            raise ValueError(f"the input is on '{x.device}' but the network is on '{self.device}'; "
                             f"use x.to('{self.device}')")
        if self.built and x.shape[1] != self.in_features and x.shape[1] == 1:
            raise ValueError(f"the network takes {self.in_features} features, got 1: the input has shape "
                             f"{x.shape}. A 1-D array becomes one column; for one sample of "
                             f"{self.in_features} features use a 1 x {self.in_features} array, "
                             "for example X[0:1]")
        self.build(x.shape[1])
        self._check_rows(x.shape[0])
        return None

    def _ids(self, x):
        """Token ids as a flat int32 array (row after row) and the sequence length they came with."""
        if isinstance(x, jz.Matrix):
            raise TypeError("this network starts with an Embedding, so its input is token ids (whole "
                            "numbers), not a juzhen matrix; pass a numpy integer array or a list, "
                            "for example x.numpy().astype(int)")
        a = np.asarray(x)
        if a.dtype.kind not in "iu":
            raise TypeError(f"token ids must be whole numbers, got {a.dtype} values; use ids.astype(int)")
        if a.size == 0:
            raise ValueError("no token ids were given")
        vocab = self.layers[0].vocab
        if a.min() < 0 or a.max() >= vocab:
            where = tuple(int(i) for i in np.argwhere((a < 0) | (a >= vocab))[0])
            at = f"ids[{', '.join(map(str, where))}]" if a.ndim else "the id"
            raise ValueError(f"{at} is {int(a[where])}, but the vocabulary has {vocab} tokens (ids 0 to "
                             f"{vocab - 1})")
        if self.seq_len and a.ndim == 2 and a.shape[1] != self.seq_len:
            raise ValueError(f"the ids have shape {a.shape}, {a.shape[1]} tokens per sequence, but the "
                             f"network was built for sequences of {self.seq_len} tokens")
        self._check_rows(a.size)
        return np.ascontiguousarray(a, dtype=np.int32).reshape(-1), (self.seq_len or 1)

    def __call__(self, x):
        """The output for a batch x; keeps x for backward().

        x is a matrix (batch x features), or token ids when the first layer
        is an Embedding: integers of shape (batch, seq_len), or flat. The
        output then has one row per token, sequence after sequence.
        """
        ids = self._check(x)
        out = self._net.forward(x) if ids is None else self._net.forward_ids(*ids)
        self._out_shape = out.shape
        return out

    forward = __call__

    def predict(self, x):
        """The output without keeping x (no backward() afterwards)."""
        ids = self._check(x)
        self._out_shape = None
        return self._net.predict(x) if ids is None else self._net.predict_ids(*ids)

    def backward(self, grad):
        """Back-propagate grad, the loss gradient for the last output.

        Adds to the gradient of every parameter and returns the gradient with
        respect to the input. Token ids have no gradient: a network that
        starts with an Embedding returns the gradient with respect to the
        embedded tokens (tokens x dim) instead.
        """
        if isinstance(grad, Loss):
            raise TypeError("backward() takes the gradient, not the loss: write net.backward(loss.grad)")
        if not isinstance(grad, jz.Matrix):
            raise TypeError(f"backward() takes a juzhen matrix (loss.grad), got {type(grad).__name__}")
        if self._out_shape is None:
            raise RuntimeError("call the network on a batch before backward(): out = net(x) "
                               "(predict() does not keep what backward needs)")
        if grad.device != self.device:
            raise ValueError(f"the gradient is on '{grad.device}' but the network is on '{self.device}'; "
                             f"use grad.to('{self.device}')")
        if grad.shape != self._out_shape:
            raise ValueError(f"the gradient has shape {grad.shape} but the network's output has shape "
                             f"{self._out_shape}; pass loss.grad of a loss computed from that output")
        return self._net.backward(grad)

    def jvp(self, x, tangent):
        """J(x) @ tangent: how the output changes when x moves along tangent."""
        if self.takes_ids:
            raise TypeError("jvp() is not defined for a network whose input is token ids: whole numbers "
                            "cannot move along a tangent")
        self._check(x)
        if not isinstance(tangent, jz.Matrix):
            raise TypeError(f"the tangent must be a juzhen matrix shaped like x, got {type(tangent).__name__}")
        if tangent.device != self.device:
            raise ValueError(f"the tangent is on '{tangent.device}' but the network is on '{self.device}'; "
                             f"use tangent.to('{self.device}')")
        if tangent.shape != x.shape:
            raise ValueError(f"the tangent has shape {tangent.shape} but x has shape {x.shape}; "
                             "they must be the same")
        self._out_shape = None
        return self._net.jvp(x, tangent)

    def parameters(self):
        """The trainable parameters (named "layer<k>.<name>"), for an optimizer."""
        if not self.built:
            raise RuntimeError("the network is not built yet: give the first layer its input size, "
                               "call build(in_features) or call the network once")
        return self._net.parameters()

    def state_dict(self):
        """The parameters as numpy arrays (copies), by name.

        load_state_dict() puts them back, into this network or another one
        with the same layers: one built for a different batch size or
        device, for example.
        """
        return {p.name: p.value.numpy() for p in self.parameters()}

    def load_state_dict(self, state):
        """Set the parameters from a state_dict() of a network with the same layers."""
        params = self.parameters()
        missing = [p.name for p in params if p.name not in state]
        if missing or len(state) != len(params):
            raise ValueError(f"the state does not match this network: it has {sorted(state)}, "
                             f"the network needs {[p.name for p in params]}")
        # check every shape before changing anything
        for p in params:
            shape = np.shape(state[p.name])
            if tuple(shape) != tuple(p.shape):
                raise ValueError(f"the state does not match this network: '{p.name}' has shape "
                                 f"{tuple(shape)} in the state but {tuple(p.shape)} in the network; "
                                 "nothing was loaded")
        for p in params:
            p.assign(jz.array(state[p.name], device=self.device))

    def copy(self, batch_size=None, device=None):
        """A new network with the same layers and a copy of the weights.

        batch_size rebuilds the layers that fix it (Transformer, Conv2d) for
        that many sequences or images: model.copy(batch_size=1) is the network
        for generating one sequence at a time. device moves the copy. The
        copy does not follow later training of this network, and Adam's
        moments are not copied.
        """
        if not self.built:
            raise RuntimeError("the network is not built yet, so it has no weights to copy: call it once "
                               "or call build(in_features) first")
        layers = [copy.copy(l) if batch_size is None else l._with_batch(batch_size) for l in self.layers]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the original was warned already
            other = Sequential(*layers, device=self.device if device is None else device)
        other.build(self.in_features)
        other.load_state_dict(self.state_dict())
        return other

    def __reduce__(self):
        raise TypeError("a network cannot be pickled; save it with jz.save_checkpoint(net, path) "
                        "or keep net.state_dict()")

    def __repr__(self):
        inner = ",\n".join(f"    {layer!r}" for layer in self.layers)
        return f"Sequential(\n{inner},\n    device={self.device!r})"

    # ------------------------------------------------------------------
    # The five-line interface: fit() and evaluate()

    def _data(self, X, Y, loss, what):
        """X and Y as host arrays: float32 2-D, or integers for token ids and class numbers."""
        if self.takes_ids:
            X = np.asarray(X.numpy() if isinstance(X, jz.Matrix) else X)
            if X.dtype.kind not in "iu":
                raise TypeError(f"{what}: this network starts with an Embedding, so X holds token ids "
                                f"(whole numbers); got {X.dtype} values, use X.astype(int)")
            if self.seq_len and (X.ndim != 2 or X.shape[1] != self.seq_len):
                raise ValueError(f"{what}: X must have shape (sequences, {self.seq_len}), one row of token "
                                 f"ids per sequence; got {X.shape}")
        else:
            X = jz.data._host(X)
        if not isinstance(Y, jz.Matrix):
            Y = np.asarray(Y)
            if Y.dtype.kind in "USO":
                raise TypeError(f"{what}: Y holds text or objects ({Y.dtype}); labels must be class numbers, "
                                "for example np.unique(Y, return_inverse=True)[1]")
            if loss == "cross_entropy":
                per_row = Y.ndim == 1 or (Y.ndim == 2 and Y.shape[1] == 1)
                if Y.dtype.kind in "iu":
                    if not self.takes_ids and not per_row:
                        raise ValueError(f"{what}: Y is an integer array of shape {Y.shape}. Whole numbers "
                                         "are class numbers, one per sample; for one-hot rows pass "
                                         "Y.astype('float32'), or pass the class numbers Y.argmax(1)")
                    if len(Y) != len(X):
                        raise ValueError(f"X has {len(X)} rows but Y has {len(Y)}")
                    return X, Y  # class numbers, one per row (or per token)
                if per_row:
                    raise TypeError(f"{what}: Y holds {Y.dtype} values, one per sample, but cross_entropy "
                                    "needs class numbers (whole numbers) or one-hot rows; for class "
                                    "numbers use Y.astype(int)")
        return X, jz.data._host(Y)

    def _batch(self, X, batch_size, default, what):
        """Samples per batch and rows per sample, respecting layers with a fixed batch."""
        per_sample = self.seq_len if (self.takes_ids and X.ndim == 2 and self.seq_len) else 1
        fixed = self.batch_rows // per_sample if self.batch_rows else None
        if batch_size is None:
            return fixed or default, per_sample
        if fixed and batch_size != fixed:
            unit = "sequences" if per_sample > 1 else "samples"
            raise ValueError(f"{what}(batch_size={batch_size}) does not fit the network, which takes "
                             f"{self._fixed_rows()}: leave batch_size out, or use {fixed} {unit}")
        return _size(batch_size, f"{what}: batch_size"), per_sample

    def fit(self, X, Y, epochs=1, batch_size=None, lr=1e-3, loss="cross_entropy",
            weight_decay=0.0, shuffle=True, verbose=True):
        """Train on X and Y with AdamW; returns the mean loss of each epoch.

        X is samples x features, or token ids (sequences x seq_len) for a
        network that starts with an Embedding. loss is "cross_entropy" (Y
        holds class numbers, or one-hot rows) or "mse". batch_size defaults
        to 32, or to the size the network's layers were built for.
        """
        loss_fn = _loss_function(loss)
        X, Y = self._data(X, Y, loss, "fit")
        batch, _ = self._batch(X, batch_size, 32, "fit")
        loader = jz.data.DataLoader(X, Y, batch_size=batch, shuffle=shuffle,
                                    drop_last=True, device=self.device)
        if not self.takes_ids:
            self.build(loader.features)
        # a new optimizer each call: lr and weight_decay take effect, and the
        # Adam moments are kept, because they live in the layers
        self.optimizer = jz.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
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
            if not math.isfinite(history[-1]):
                raise ValueError(f"fit(): the loss became {history[-1]} in epoch {epoch + 1}. Usual causes: "
                                 f"a learning rate that is too large (lr={lr:g}), or nan or inf in X or Y. "
                                 "The weights are no longer usable: build the network again")
            if verbose:
                print(f"epoch {epoch + 1}/{epochs}: loss {history[-1]:.4f}")
        return history

    def evaluate(self, X, Y, batch_size=None, loss="cross_entropy"):
        """Mean loss (and accuracy for cross_entropy) on all of X, Y.

        batch_size defaults to 1000, or to the size the network's layers were
        built for; then a last, smaller batch is filled up with copies that
        are not counted, so every sample counts exactly once.
        """
        loss_fn = _loss_function(loss)
        X, Y = self._data(X, Y, loss, "evaluate")
        batch, per_sample = self._batch(X, batch_size, 1000, "evaluate")
        if not self.takes_ids:
            self.build(X.shape[1])
        n, fixed = len(X), bool(self.batch_rows)
        labels = Y.dtype.kind in "iu"
        total_loss, total_errors, counted = 0.0, 0.0, 0
        for start in range(0, n, batch):
            new = min(batch, n - start)
            rows = np.arange(start, start + new)
            if fixed and new < batch:
                # a fixed batch is filled up with copies of its first sample; only the real ones count
                rows = np.concatenate([rows, np.full(batch - new, start)])
            X_i, Y_i = X[rows], Y[rows]
            scores = self.predict(X_i if self.takes_ids else jz.array(X_i, device=self.device))
            if not labels:
                Y_i = jz.array(Y_i, device=self.device)
            keep = new * per_sample                 # rows of the real samples: the first ones
            if keep < scores.shape[0]:
                scores = scores[:keep]
                Y_i = Y_i.reshape(-1)[:keep] if labels else Y_i[:keep]
            weight = int(np.count_nonzero(np.asarray(Y_i) != -100)) if labels else keep
            if weight:
                total_loss += loss_fn(scores, Y_i).value.item() * weight
                if loss == "cross_entropy":
                    total_errors += error_rate(scores, Y_i).item() * weight
            counted += weight
        result = {"loss": total_loss / max(counted, 1)}
        if loss == "cross_entropy":
            result["accuracy"] = 1 - total_errors / max(counted, 1)
        return result


# ----------------------------------------------------------------------
# Losses (ml/loss.hpp). Each returns the value (1 x 1, averaged over the
# batch) and its gradient with respect to the prediction.


class Loss:
    """What a loss function returns: value (1 x 1) and grad (shaped like the prediction).

    It can be used where its number is wanted: ``f"{loss:.4f}"``,
    ``float(loss)``, ``loss < 0.1``; each of them waits for the GPU, as
    loss.item() does. ``value, grad = loss`` unpacks it. To keep a history,
    store loss.item(): a stored Loss keeps its whole gradient alive.
    """

    __slots__ = ("value", "grad")

    def __init__(self, value, grad):
        self.value = value  # 1 x 1 matrix
        self.grad = grad    # same shape as the prediction

    def __iter__(self):
        yield self.value
        yield self.grad

    def item(self):
        """The loss as a number (waits for the GPU)."""
        return self.value.item()

    def __float__(self):
        return self.value.item()

    def __round__(self, digits=None):
        return round(self.value.item(), digits)

    def __format__(self, spec):
        return format(self.value.item(), spec) if spec else repr(self)

    def __repr__(self):
        rows, cols = self.grad.shape
        return f"Loss({self.value.item():.6g}, grad: {rows} x {cols} on '{self.grad.device}')"

    def __lt__(self, other):
        return self.value.item() < float(other)

    def __le__(self, other):
        return self.value.item() <= float(other)

    def __gt__(self, other):
        return self.value.item() > float(other)

    def __ge__(self, other):
        return self.value.item() >= float(other)

    def _not_a_number(self, *args):
        raise TypeError("a Loss is not a number in arithmetic; use loss.item() (waits for the GPU) or "
                        "loss.value (a 1 x 1 matrix that stays on the device), and loss.grad / k to "
                        "scale the gradient")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _not_a_number
    __truediv__ = __rtruediv__ = __neg__ = __iadd__ = _not_a_number

    def backward(self, *args, **kwargs):
        raise RuntimeError("juzhen has no automatic differentiation: the loss does not know the network. "
                           "Write net.backward(loss.grad), then opt.step()")


def _core_of(m):
    return jz._device(m.device)


def _prediction(m, what, name="logits"):
    if not isinstance(m, jz.Matrix):
        raise TypeError(f"{what}: the {name} must be a juzhen matrix (the output of the network), "
                        f"got {type(m).__name__}")
    return m


def _are(name):
    return "are" if name.endswith("s") else "is"


def _have(name):
    return "have" if name.endswith("s") else "has"


def _same(a, b, what, a_name, b_name, remedy=""):
    """b is a juzhen matrix on a's device with a's shape."""
    if not isinstance(b, jz.Matrix):
        raise TypeError(f"{what}: the {b_name} must be a juzhen matrix, got {type(b).__name__}; "
                        f"use jz.array({b_name}, device='{a.device}')")
    if a.device != b.device:
        raise ValueError(f"{what}: the {a_name} {_are(a_name)} on '{a.device}' but the {b_name} "
                         f"{_are(b_name)} on '{b.device}'; use {b_name}.to('{a.device}')")
    if a.shape != b.shape:
        raise ValueError(f"{what}: the {b_name} {_have(b_name)} shape {b.shape} but the {a_name} "
                         f"{_have(a_name)} shape {a.shape}{remedy}")


def _labels_remedy(logits, labels):
    if labels.shape == (logits.shape[0], 1):
        return (f". A matrix of labels must hold one-hot rows (or probabilities), {logits.shape[0]} x "
                f"{logits.shape[1]} here; for class numbers pass whole numbers instead: "
                "labels.numpy().astype(int)")
    return ". A matrix of labels must hold one-hot rows (or probabilities) with the same shape"


def _class_numbers(labels, scores, ignore_index, what):
    """Class numbers as a flat int32 array, one per row of scores, checked."""
    a = np.asarray(labels)
    n, classes = scores.shape
    if a.dtype.kind == "f":
        if a.ndim == 2 and a.shape == (n, classes):
            raise TypeError(f"{what}: one-hot labels must be a juzhen matrix; "
                            f"use jz.array(labels, device='{scores.device}')")
        raise TypeError(f"{what}: class numbers must be whole numbers, got {a.dtype} values; "
                        "use labels.astype(int)")
    if a.dtype.kind not in "iu":
        raise TypeError(f"{what}: labels must be class numbers (a numpy integer array or a list of ints) "
                        f"or a juzhen matrix of one-hot rows, got {type(labels).__name__}")
    if a.ndim == 2 and a.shape == (n, classes) and classes > 1:
        raise ValueError(f"{what}: the labels are an integer array of shape {a.shape}, the shape of the "
                         "logits. Whole numbers are class numbers, one per row: pass labels.argmax(1), "
                         f"or one-hot rows as a matrix: jz.array(labels, device='{scores.device}')")
    a = a.reshape(-1)  # row after row: (batch, seq_len) targets are one sequence after another
    if a.size != n:
        raise ValueError(f"{what}: there are {a.size} labels for {n} rows of logits; "
                         "class numbers need exactly one label per row")
    wrong = (a != ignore_index) & ((a < 0) | (a >= classes))
    if wrong.any():
        row = int(np.argmax(wrong))
        hint = " If your classes are numbered from 1, subtract 1." if a[row] == classes else ""
        raise ValueError(f"{what}: label {int(a[row])} at row {row} is outside the {classes} classes "
                         f"(0 to {classes - 1}).{hint} A label of {ignore_index} skips its row")
    return np.ascontiguousarray(a, dtype=np.int32)


def _one_per_class(logits, what):
    if logits.shape[1] == 1:
        raise ValueError(f"{what} needs one output per class, but the logits have 1 column. For two "
                         "classes end the network with Linear(2) and use the labels 0 and 1")


def softmax_cross_entropy(logits, labels, ignore_index=-100):
    """Softmax cross-entropy, averaged over the batch.

    labels are class numbers, one per row of logits (a numpy integer array or
    a list; rows labelled ignore_index are skipped), or a juzhen matrix of
    one-hot rows (or probabilities) shaped like logits. If every row is
    skipped the value is 0 and the gradient is zero.
    """
    what = "softmax_cross_entropy"
    _one_per_class(_prediction(logits, what), what)
    core = _core_of(logits)
    if isinstance(labels, jz.Matrix):
        _same(logits, labels, what, "logits", "labels", _labels_remedy(logits, labels))
        return Loss(*core.softmax_cross_entropy(logits, labels))
    numbers = _class_numbers(labels, logits, ignore_index, what)
    return Loss(*core.softmax_cross_entropy_labels(logits, numbers, int(ignore_index)))


def mse(prediction, target):
    """Squared error summed over outputs, averaged over the batch."""
    _same(_prediction(prediction, "mse", "prediction"), target, "mse", "prediction", "target",
          ". The target needs one row per sample and one column per output")
    return Loss(*_core_of(prediction).mse(prediction, target))


def masked_softmax_cross_entropy(logits, labels, weights, ignore_index=-100):
    """Softmax cross-entropy weighted per sample, without batch averaging.

    weights (length N) must be finite and nonnegative and already include any
    scaling; a zero weight drops its sample from the loss and the gradient.
    labels are class numbers or one-hot rows, as in softmax_cross_entropy.
    """
    what = "masked_softmax_cross_entropy"
    _one_per_class(_prediction(logits, what), what)
    core = _core_of(logits)
    if isinstance(weights, jz.Matrix):
        if weights.device != logits.device:
            raise ValueError(f"{what}: the logits are on '{logits.device}' but the weights are on "
                             f"'{weights.device}'; use weights.to('{logits.device}')")
    else:
        weights = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        if len(weights) != logits.shape[0]:
            raise ValueError(f"{what}: weights must have shape ({logits.shape[0]}, 1), one weight per row "
                             f"of logits; got {len(weights)} weights")
        weights = jz.array(weights, device=logits.device)
    if isinstance(labels, jz.Matrix):
        _same(logits, labels, what, "logits", "labels", _labels_remedy(logits, labels))
        return Loss(*core.masked_softmax_cross_entropy(logits, labels, weights))
    numbers = _class_numbers(labels, logits, ignore_index, what)
    if weights.shape != (logits.shape[0], 1):
        raise ValueError(f"{what}: weights must have shape ({logits.shape[0]}, 1), got {weights.shape}")
    return Loss(*core.masked_softmax_cross_entropy_labels(logits, numbers, weights, int(ignore_index)))


def error_rate(scores, labels, ignore_index=-100):
    """Fraction of rows whose largest score is not the labelled class (1 x 1).

    labels are class numbers or one-hot rows, as in softmax_cross_entropy.
    """
    what = "error_rate"
    _prediction(scores, what, "scores")
    if isinstance(labels, jz.Matrix):
        _same(scores, labels, what, "scores", "labels", _labels_remedy(scores, labels))
        return _core_of(scores).error_rate(scores, labels)
    numbers = _class_numbers(labels, scores, ignore_index, what)
    return _core_of(scores).error_rate_labels(scores, numbers, int(ignore_index))


def _loss_function(name):
    losses = {"cross_entropy": softmax_cross_entropy, "mse": mse}
    if name not in losses:
        raise ValueError(f"loss must be one of {sorted(losses)}, not {name!r}")
    return losses[name]


def one_hot(labels, num_classes, device="cpu"):
    """Integer class labels (length N) as an N x num_classes one-hot matrix.

    The matrix is built on the device: only the N labels are copied there.
    The losses also take the class numbers themselves; one_hot is for code
    that needs the matrix.
    """
    values = np.asarray(labels.numpy() if isinstance(labels, jz.Matrix) else labels).ravel()
    if values.size == 0:
        raise ValueError("one_hot needs at least one label")
    if values.dtype.kind == "f" and not (values == np.round(values)).all():
        bad = values[values != np.round(values)][0]
        raise ValueError(f"labels must be whole numbers from 0 to {num_classes - 1}, found {bad:g}")
    values = values.astype(np.int64)
    if values.min() < 0 or values.max() >= num_classes:
        bad = int(values[(values < 0) | (values >= num_classes)][0])
        hint = " If your classes are numbered from 1, subtract 1." if bad == num_classes else ""
        raise ValueError(f"labels must be in [0, {num_classes}): whole numbers from 0 to {num_classes - 1}, "
                         f"found {bad}.{hint}")
    column = jz.array(values.astype(np.float32).reshape(-1, 1), device=device)
    return jz._device(device).one_hot(column, int(num_classes))


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
