# juzhen for Python

Python interface to [Juzhen](https://github.com/anewgithubname/Juzhen), a small,
readable C++ library of matrices and neural networks.

```python
import juzhen as jz

X, Y, X_test, Y_test = jz.datasets.mnist()        # one sample per row

model = jz.Sequential(jz.Dense(256, activation="relu"), jz.Linear(10))
model.fit(X, Y, epochs=3, lr=1e-3)
print(model.evaluate(X_test, Y_test))
```

- `@` is the matrix product and `*` is elementwise, as in numpy.
- Data is one sample per row (N x d). Inside, the C++ layers see the same bytes
  as d x N, the layout they are written for, so nothing is copied or rearranged.
- Matrices live in Juzhen's memory pool, on the CPU or with `device="gpu"` on the
  GPU backend of the build. Converting to or from numpy copies.
- Reading values (`.numpy()`, `.item()`, `print`) waits for the GPU.
- Comparisons work element by element and give a matrix of 1s and 0s:
  `(A > 0) * A`, `jz.mean(A == B)`. A numpy array has to be made a matrix
  first (`jz.array(y)`); for the accuracy of a classifier use
  `1 - jz.nn.error_rate(scores, labels)`.
- Numbers are juzhen matrices. Whole numbers that name things are numpy integer
  arrays or lists: class numbers for a loss (`model.fit(X, y)` with `y` holding
  0..9 works; one-hot rows still do too) and token ids for a network that starts
  with `nn.Embedding`.
- Weights start at a scale that trains: N(0, 2/fan_in) in front of a ReLU,
  N(0, 1/fan_in) otherwise, zero biases. `init=` on a layer overrides it with a
  standard deviation.

A language model on token ids is an ordinary `Sequential`. This one learns
to continue a repeating sequence, with the training step written out:

```python
import numpy as np
import juzhen as jz
from juzhen import nn, optim

vocab, seq_len, batch = 16, 8, 32
device = "gpu" if jz.has_gpu() else "cpu"
model = nn.Sequential(
    nn.Embedding(vocab, 64),             # token ids -> vectors
    nn.PositionEmbedding(seq_len),       # + a learned vector per position
    nn.Transformer(64, 64, 256, seq_len, batch, heads=4),
    nn.LayerNorm(),
    nn.Linear(vocab, tied=True),         # scores from the embedding table
    device=device)
opt = optim.AdamW(model.parameters(), lr=3e-3)

rng = np.random.default_rng(0)
for step in range(200):
    start = rng.integers(0, vocab, (batch, 1))
    ids = (start + np.arange(seq_len + 1)) % vocab     # integers, shape (batch, seq_len + 1)
    out = model(ids[:, :-1])                           # one row of scores per token
    loss = nn.softmax_cross_entropy(out, ids[:, 1:])   # the targets are the next ids
    opt.zero_grad()
    model.backward(loss.grad)                          # no autograd: the loss hands over its gradient
    opt.step()
print(f"loss {loss:.3f}")

# After training: a copy of the weights as they are now, for one sequence at a
# time. The copy does not follow later training of `model`.
generator = model.copy(batch_size=1)
prompt = np.arange(3, 3 + seq_len)[None, :] % vocab
print("next token:", int(jz.argmax(generator.predict(prompt))[-1, 0]))   # 11
```

New to the library? Start with the notebook `examples/tutorial.ipynb`.

## Performance notes

Every operator returns a new matrix: `A @ B / n` computes `A @ B`, then a
second matrix for the division. The C++ library reuses the temporary in that
case; Python cannot tell that `A @ B` is a temporary. On large matrices,
scale or accumulate in place instead:

```python
n = 2000
A1, A2, A3 = jz.randn(n, n), jz.randn(n, n), jz.zeros(n, n)

A3 += A1 @ A2 / n        # one extra matrix and pass over memory for "/ n"

C = A1 @ A2              # the same result without it
C /= n                   # in place: *=, /=, += and -= do not allocate
A3 += C
```

On an RTX 4090 with 5000 x 5000 matrices the first form runs at 43 TFLOPS,
the second at 46 TFLOPS, the speed of the C++ demo. Matrix products and the
layers themselves run at the speed of C++.

Reading a value back (`.item()`, `.numpy()`) waits for the GPU to finish, so
in a training loop read the loss every few hundred steps rather than every
step when speed matters.

## Examples

The scripts in `examples/` run on the GPU when the build has one; `--device cpu`
(or `cpu` as the first argument of the `fit_` ones) runs them on the CPU.

| file | what it shows | C++ original |
|---|---|---|
| `tutorial.ipynb` | a guided tour of the whole library, step by step | |
| `cifar_rectified_flow.ipynb` | an image generator on CIFAR-10: a UNet built from `Sequential` pieces, with a speed comparison against C++ | `demo_cnn_rectified.cu` |
| `helloworld.py` | matrix arithmetic on both devices | `helloworld.cu` |
| `fit_mnist.py` | `fit()` and `evaluate()` | |
| `fit_sin.py` | a hand-written training loop | |
| `demo_mnist.py` | an MLP on MNIST, step by step | `demo_mnist.cu` |
| `cnn_mnist.py` | convolutions on MNIST | `demo_cnn_mnist.cu` |
| `rectified_flow.py` | a generative flow; `hstack`, Euler sampling | `demo_rectified.cu` |
| `transformer_charlm.py` | a character transformer LM on token ids: `Embedding`, `PositionEmbedding`, a tied output, `copy(batch_size=1)` for generation | `demo_transformer.cu` (one-hot), `demo_llm_tokens.cu` |
| `knn.py` | k nearest neighbours with `topk` | `knn.cu` |

See `misc/python_interface_plan.md` for the design.

## Building

From the repository root:

```
pip install ./python                                   # CPU
pip install ./python -C cmake.define.NVIDIA_CUDA=ON    # CUDA
pip install ./python -C cmake.define.ROCM_HIP=ON       # ROCm
pip install ./python -C cmake.define.APPLE_SILICON=ON  # Metal
pytest python/tests
```

On Windows build from a Visual Studio developer prompt (MSVC).
