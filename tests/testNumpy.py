import numpy as np
import time

np.random.seed(0)  # Equivalent to 'rng default'

n = 1001

A = np.random.randn(n, n)
B = np.random.randn(n, n)

C = np.zeros((n, n))

start_time = time.time()  # Equivalent to 'tic'

for i in range(50):  # Python indexing starts at 0
    C = C + (np.dot(A, B.T) / n * A) + B - A

end_time = time.time()  # Equivalent to 'toc'

print(f"Elapsed time: {end_time - start_time} seconds")
