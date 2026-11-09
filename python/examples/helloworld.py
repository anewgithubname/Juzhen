"""examples/helloworld.cu in Python; prints the same numbers."""

import numpy as np

import juzhen as jz

A = jz.array([[1, 2, 3],
              [4, 5, 6]])                 # 2 x 3, on the CPU
B = jz.array([[.1, .2],
              [.3, .4],
              [.5, .6]])                  # 3 x 2

C = jz.log(jz.exp(A @ B) + 1) / 5         # @ is the matrix product
print(C)

if jz.has_gpu():
    # the same code on the GPU: only device changes
    Ag = A.to("gpu")
    Bg = B.to("gpu")
    Cg = jz.log(jz.exp(Ag @ Bg) + 1) / 5
    print(np.allclose(C.numpy(), Cg.numpy(), atol=1e-5))

print(jz.backend())
