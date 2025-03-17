import time
import torch
import torchvision
import torchvision.transforms as transforms

device = 'cpu'

# gemm on cpu for 10000 by 10000 matrix 10 times
A = torch.randn(10000, 10000, device=device)
B = torch.randn(10000, 10000, device=device)
C = torch.zeros(10000, 10000, device=device)

start = time.time()
for _ in range(10):
    C = C + A @ B/10000
print("Time: ", time.time() - start)

# gemm on gpu for 5000 by 5000 matrix 10 times

device = "mps"
A = torch.randn(10000, 10000, device=device)
B = torch.randn(10000, 10000, device=device)
C = torch.zeros(10000, 10000, device=device)

start = time.time()
for _ in range(10):
    C = A @ B

print(C[:5,:5])
print("Time: ", time.time() - start)

