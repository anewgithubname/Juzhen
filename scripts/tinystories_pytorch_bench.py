#!/usr/bin/env python3
"""PyTorch speed benchmark matching demo_tinystories.cu (Juzhen).

Architecture is matched to the Juzhen TransformerLayer so the FLOPs-per-step
are equivalent and the ms/step comparison is fair:
  - token embedding (V x d_model) + learned absolute position embedding
    (seq x d_model)  [Juzhen: one-hot(token) ⊕ one-hot(pos) -> Linear]
  - N pre-norm blocks: x = x + Attn(LN1(x)); x = x + FFN(LN2(x))
  - multi-head causal attention, scale 1/sqrt(d_h)
  - FFN: Linear(d_model->d_ff) -> ReLU -> Linear(d_ff->d_model)   [ReLU, not GELU]
  - final LayerNorm-free projection d_model -> V, softmax cross-entropy
  - Adam, same dims: 8L / d512 / dff2048 / 8h / seq256 / batch64 / vocab4096

We do NOT train to convergence — just enough warmup + timed steps to measure
throughput. Juzhen reference: ~141 ms/step (TF32) at batch 64, seq 256.

Runs several PyTorch configs so we see it at its best:
  fp32+TF32, AMP-bf16, AMP-bf16 + torch.compile.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = "/mnt/external_hdd/data/nlp/tinystories"
V = 4096
SEQ = 256
BATCH = 64
D_MODEL = 512
D_K = 512
D_FF = 2048
HEADS = 8
BLOCKS = 8
WARMUP = 30
TIMED = 150
LR = 5e-4

dev = torch.device("cuda")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.Wq = nn.Linear(D_MODEL, D_K, bias=False)
        self.Wk = nn.Linear(D_MODEL, D_K, bias=False)
        self.Wv = nn.Linear(D_MODEL, D_K, bias=False)
        self.Wo = nn.Linear(D_K, D_MODEL, bias=True)
        self.W1 = nn.Linear(D_MODEL, D_FF, bias=True)
        self.W2 = nn.Linear(D_FF, D_MODEL, bias=True)

    def forward(self, x):  # x: (B, T, d_model)
        B, T, _ = x.shape
        h = self.ln1(x)
        q = self.Wq(h).view(B, T, HEADS, D_K // HEADS).transpose(1, 2)
        k = self.Wk(h).view(B, T, HEADS, D_K // HEADS).transpose(1, 2)
        v = self.Wv(h).view(B, T, HEADS, D_K // HEADS).transpose(1, 2)
        # fused scaled dot-product attention (flash) with causal mask
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, D_K)
        x = x + self.Wo(a)
        h = self.ln2(x)
        x = x + self.W2(F.relu(self.W1(h)))
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D_MODEL)
        self.pos = nn.Embedding(SEQ, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(BLOCKS))
        self.proj = nn.Linear(D_MODEL, V)
        self.register_buffer("pos_ids", torch.arange(SEQ).unsqueeze(0))

    def forward(self, idx):                       # idx: (B, T)
        x = self.tok(idx) + self.pos(self.pos_ids)
        for b in self.blocks:
            x = b(x)
        return self.proj(x)                       # (B, T, V)


def make_batch(data):
    n = len(data)
    ix = np.random.randint(0, n - SEQ - 1, size=BATCH)
    x = np.stack([data[i:i + SEQ] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + SEQ + 1] for i in ix]).astype(np.int64)
    return (torch.from_numpy(x).to(dev, non_blocking=True),
            torch.from_numpy(y).to(dev, non_blocking=True))


def run(tag, use_amp, compile_model):
    torch.manual_seed(0)
    model = Model().to(dev)
    if compile_model:
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    data = np.memmap(f"{BASE}/train.bin", dtype=np.uint16, mode="r")

    def step():
        x, y = make_batch(data)
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        loss.backward()
        opt.step()
        return loss.item()

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    last = 0.0
    for _ in range(TIMED):
        last = step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1024**2
    per = dt / TIMED * 1000
    toks = BATCH * SEQ / (per / 1000)
    print(f"{tag:28s} {per:7.1f} ms/step  {toks/1e3:6.1f}k tok/s  "
          f"peak {mem:6.0f} MB  (loss {last:.2f})")
    del model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return per


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"PyTorch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"config: {BLOCKS}L d_model={D_MODEL} d_ff={D_FF} heads={HEADS} "
          f"seq={SEQ} batch={BATCH} vocab={V}")
    print(f"timing {TIMED} steps after {WARMUP} warmup\n")
    print(f"{'config':28s} {'ms/step':>10s}  {'throughput':>12s}  {'peak mem':>10s}")
    run("fp32 + TF32", use_amp=False, compile_model=False)
    run("AMP bf16", use_amp=True, compile_model=False)
    run("AMP bf16 + torch.compile", use_amp=True, compile_model=True)
    print("\nJuzhen reference: ~141 ms/step  (116.4k tok/s)  peak ~13.5 GB (TF32)")


if __name__ == "__main__":
    main()
