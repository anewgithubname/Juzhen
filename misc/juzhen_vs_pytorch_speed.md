# Juzhen vs PyTorch training-speed comparison

Single-GPU (RTX 4090) training-step throughput for the **same** transformer
architecture, so per-step FLOPs are equivalent and only the framework/kernels
differ. Measured 2026-07-05.

- Model: 8 layers, d_model=512, d_k=512, d_ff=2048, 8 heads, seq_len=256,
  batch=64, vocab=4096 (~30M params), causal AR LM on TinyStories.
- Juzhen: `demo_tinystories` (CUDA backend, TF32 via NVIDIA_TF32=1), measured
  over the real training run (141 ms/step).
- PyTorch: `scripts/tinystories_pytorch_bench.py` (PyTorch 2.10.0+cu128),
  architecture matched (pre-norm, ReLU FFN, learned token+pos embeddings,
  `scaled_dot_product_attention` = FlashAttention). 30 warmup + 150 timed steps.

## Results

| Configuration | ms/step | throughput | peak mem | vs Juzhen |
|---------------|--------:|-----------:|---------:|----------:|
| **Juzhen (TF32)** | 141 | 116k tok/s | ~13.5 GB | 1.0× (baseline) |
| PyTorch fp32 + TF32 (same precision) | 61.8 | 265k tok/s | 4.5 GB | **2.3× faster** |
| PyTorch AMP bf16 | 37.4 | 438k tok/s | 3.2 GB | **3.8× faster** |
| PyTorch AMP bf16 + torch.compile | 32.6 | 503k tok/s | 3.2 GB | **4.3× faster** |

## Interpretation

- **Same-precision (TF32 vs TF32): PyTorch is ~2.3× faster.** This is the
  fairest single comparison. The gap comes from PyTorch using a fused
  FlashAttention kernel (`scaled_dot_product_attention`) and better kernel
  scheduling/fusion, whereas Juzhen's attention is hand-written with more
  separate kernel launches and materialised intermediate tensors.
- **PyTorch at full tilt (bf16 + compile): ~4.3× faster.** The extra speedup
  beyond 2.3× is mixed precision (the 4090's bf16 tensor cores far outrun TF32)
  plus `torch.compile` graph optimisation / operator fusion — neither of which
  Juzhen currently has.
- **Memory: Juzhen 13.5 GB vs PyTorch 3–4.5 GB (3–4× more).** Juzhen builds a
  dense one-hot embedding input (V+seq = 4352-row matrix) and materialises more
  activations; PyTorch uses a true embedding lookup and bf16 activations.

## Takeaway

For a self-built teaching framework, trailing a mature, heavily-optimised
PyTorch by ~2.3× at equal precision is a respectable result — not an
order-of-magnitude gap. The core GEMMs (which route through cuBLAS) are
evidently efficient; the gap is concentrated in known, addressable engineering
optimisations: fused attention, mixed precision, and graph-level fusion.

## Reproduce

```bash
# Juzhen (during/after training):
NVIDIA_TF32=1 ./build/demo_tinystories     # reports ms/step in its log

# PyTorch:
python3 scripts/tinystories_pytorch_bench.py
```
