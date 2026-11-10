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

New to the library? Start with the notebook `examples/tutorial.ipynb`.

## Performance notes

Every operator returns a new matrix: `A @ B / n` computes `A @ B`, then a
second matrix for the division. The C++ library reuses the temporary in that
case; Python cannot tell that `A @ B` is a temporary. On large matrices,
scale or accumulate in place instead:

```python
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
| `transformer_charlm.py` | a character transformer LM; `one_hot` on the device, `state_dict()` into a batch-1 network | `demo_transformer.cu` |
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
