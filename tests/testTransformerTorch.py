"""Cross-check ml/layer.hpp TransformerLayer against PyTorch.

Reads the dump written by testTransformerTorchDump (weights, input x, upstream
gradient g, forward output, input gradient dx), rebuilds the identical block
with examples/demo_transformer.py's TransformerBlock (pre-LN, multi-head,
causal, 1/sqrt(d_h) scaling), loads the same weights, and compares the forward
output and dL/dx numerically. PyTorch runs in float64, so it acts as an exact
oracle and the reported error is the C++ float32 rounding.

Usage:
    <build dir>/testTransformerTorchDump     # writes res/transformer_torch_dump.bin
    python3 tests/testTransformerTorch.py [dump_path]
"""

import os
import struct
import sys

try:
    import numpy as np
    import torch
except ImportError as e:
    print(f"SKIP: {e} (install torch/numpy to run this test)")
    sys.exit(77)  # ctest SKIP_RETURN_CODE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from demo_transformer import TransformerBlock  # noqa: E402

FWD_TOL = 1e-4   # float32 forward pass vs float64 oracle
BWD_TOL = 1e-4


def read_mat(f):
    r, c = struct.unpack("<ii", f.read(8))
    data = np.frombuffer(f.read(4 * r * c), dtype="<f4")
    return data.reshape((c, r)).T.copy()  # file stores column-major


def compare(got, ref, tag, tol):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    max_abs = float(np.abs(got - ref).max())
    rel_l2 = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-12))
    status = "PASS" if (max_abs <= tol and rel_l2 <= tol) else "FAIL"
    print(f"[{status}] {tag}: max_abs={max_abs:.3e} rel_l2={rel_l2:.3e}")
    return status == "PASS"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "res", "transformer_torch_dump.bin")
    if not os.path.exists(path):
        print(f"Dump not found: {path}\nRun testTransformerTorchDump first.")
        return 1

    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"JZTFDMP1":
            print(f"Bad magic {magic!r}: rerun testTransformerTorchDump.")
            return 1
        d_model, d_k, d_ff, seq, batch, heads = struct.unpack("<6i", f.read(24))
        names = ["Wq", "Wk", "Wv", "Wo", "bo", "W1", "b1", "W2", "b2",
                 "ln1_g", "ln1_b", "ln2_g", "ln2_b", "x", "g", "out", "dx"]
        m = {n: read_mat(f) for n in names}

    print(f"config: d_model={d_model} d_k={d_k} d_ff={d_ff} "
          f"seq={seq} batch={batch} heads={heads}")

    torch.manual_seed(0)
    blk = TransformerBlock(d_model, d_k, d_ff, seq, heads).double()
    with torch.no_grad():
        blk.Wq.weight.copy_(torch.from_numpy(m["Wq"]))
        blk.Wk.weight.copy_(torch.from_numpy(m["Wk"]))
        blk.Wv.weight.copy_(torch.from_numpy(m["Wv"]))
        blk.Wo.weight.copy_(torch.from_numpy(m["Wo"]))
        blk.Wo.bias.copy_(torch.from_numpy(m["bo"].ravel()))
        blk.ffn1.weight.copy_(torch.from_numpy(m["W1"]))
        blk.ffn1.bias.copy_(torch.from_numpy(m["b1"].ravel()))
        blk.ffn2.weight.copy_(torch.from_numpy(m["W2"]))
        blk.ffn2.bias.copy_(torch.from_numpy(m["b2"].ravel()))
        blk.ln1.weight.copy_(torch.from_numpy(m["ln1_g"].ravel()))
        blk.ln1.bias.copy_(torch.from_numpy(m["ln1_b"].ravel()))
        blk.ln2.weight.copy_(torch.from_numpy(m["ln2_g"].ravel()))
        blk.ln2.bias.copy_(torch.from_numpy(m["ln2_b"].ravel()))

    # layer.hpp uses columns as tokens (d_model, seq*batch); the PyTorch block
    # uses rows (seq*batch, d_model) with the same b*seq+s ordering: transpose.
    x = torch.from_numpy(m["x"].T.copy()).double().requires_grad_(True)
    g = torch.from_numpy(m["g"].T.copy()).double()

    out = blk(x)
    (out * g).sum().backward()

    ok = compare(out.detach().numpy().T, m["out"], "forward vs PyTorch", FWD_TOL)
    ok &= compare(x.grad.numpy().T, m["dx"], "backward (dx) vs PyTorch", BWD_TOL)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
