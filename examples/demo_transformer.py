"""
demo_transformer.py

Tiny character language model on a toy corpus.

Mirrors examples/demo_transformer.cu:
  LinearEmbed(V+seq_len -> d_model=48)
    -> two causal TransformerBlock layers(d_model=48, d_k=48, d_ff=96)
  -> LinearProj(d_model=48 -> V)
"""

import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


SEQ_LEN = 16
STRIDE = 1
D_MODEL = 48
D_K = 48
D_FF = 96
EPOCHS = 2000
LOG_EVERY = 200
GEN_LEN = 48
TOP_K = 3
TEMPERATURE = 0.8
LR = 1e-4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_toy_corpus() -> str:
    phrases = [
        "hello world. ",
        "hello moon. ",
        "good night. ",
        "good day. ",
    ]
    return "".join(phrases[i % len(phrases)] for i in range(48))


def build_vocab(corpus: str):
    chars = sorted(set(corpus))
    c2i = {ch: i for i, ch in enumerate(chars)}
    return c2i, chars


def build_dataset(corpus: str, c2i: dict, vocab_size: int):
    in_dim = vocab_size + SEQ_LEN
    windows = [(i, i + SEQ_LEN) for i in range(0, len(corpus) - SEQ_LEN, STRIDE)]
    batch = len(windows)
    n = batch * SEQ_LEN

    x = torch.zeros(n, in_dim)
    y = torch.zeros(n, dtype=torch.long)
    starts = []
    for b, (off, _) in enumerate(windows):
        starts.append(off)
        for s in range(SEQ_LEN):
            col = b * SEQ_LEN + s
            x[col, c2i[corpus[off + s]]] = 1.0
            x[col, vocab_size + s] = 1.0
            y[col] = c2i[corpus[off + s + 1]]
    return x, y, starts


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_ff: int, seq_len: int):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.Wv = nn.Linear(d_model, d_k, bias=False)
        self.Wo = nn.Linear(d_k, d_model, bias=True)
        self.ffn1 = nn.Linear(d_model, d_ff, bias=True)
        self.ffn2 = nn.Linear(d_ff, d_model, bias=True)
        self.scale = 1.0 / math.sqrt(d_k)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0] // self.seq_len
        h = x.view(batch, self.seq_len, -1)
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0), -1e9)
        a = torch.softmax(scores, dim=-1)
        attn = torch.matmul(a, v)
        r = h + self.Wo(attn)
        out = r + self.ffn2(F.relu(self.ffn1(r)))
        return out.reshape(batch * self.seq_len, -1)


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, in_dim: int):
        super().__init__()
        self.embed = nn.Linear(in_dim, D_MODEL, bias=True)
        self.tf1 = TransformerBlock(D_MODEL, D_K, D_FF, SEQ_LEN)
        self.tf2 = TransformerBlock(D_MODEL, D_K, D_FF, SEQ_LEN)
        self.proj = nn.Linear(D_MODEL, vocab_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.tf1(h)
        h = self.tf2(h)
        return self.proj(h)


def sample_top_k(logits: torch.Tensor, recent: str, idx_to_char: list[str]) -> int:
    scores = logits.clone()
    recent_window = recent[-12:]
    for idx, ch in enumerate(idx_to_char):
        scores[idx] -= 0.02 * recent_window.count(ch)
    top_scores, top_indices = torch.topk(scores, min(TOP_K, len(idx_to_char)))
    probs = torch.softmax(top_scores / TEMPERATURE, dim=-1)
    chosen = torch.multinomial(probs, 1).item()
    return int(top_indices[chosen].item())


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(SEED)

    corpus = build_toy_corpus()
    c2i, i2c = build_vocab(corpus)
    vocab_size = len(i2c)
    in_dim = vocab_size + SEQ_LEN
    x, y, starts = build_dataset(corpus, c2i, vocab_size)
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    model = TinyTransformerLM(vocab_size, in_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    print("=== Tiny Transformer Demo ===")
    print(f'Corpus: "{corpus[:96]}"')
    print(f"Vocabulary ({vocab_size} chars): {''.join(ch if ch != ' ' else '_' for ch in i2c)}")
    print(f"Network: embed({in_dim}->{D_MODEL}) -> Transformer({D_MODEL},{D_K},{D_FF}) [x2] -> proj({D_MODEL}->{vocab_size})")
    print(f"Data: {len(starts)} windows x {SEQ_LEN} tokens = {x.shape[0]} predictions")
    print(f"Epochs: {EPOCHS}   Device: {DEVICE}\n")

    for ep in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())

        if ep % LOG_EVERY == 0 or ep == EPOCHS - 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean().item() * 100.0
            print(f"epoch {ep:4d}   loss = {loss_value:.4f}   ppl = {math.exp(loss_value):6.2f}   acc = {acc:5.1f}%")

    model.load_state_dict(best_state)

    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=-1).cpu()

    print("\n--- Sample predictions ---")
    for k in range(min(3, len(starts))):
        off = starts[k * max(1, len(starts) // 3)]
        start_col = (k * max(1, len(starts) // 3)) * SEQ_LEN
        inp = corpus[off: off + SEQ_LEN]
        tgt = corpus[off + 1: off + SEQ_LEN + 1]
        pred = ''.join(i2c[int(preds[start_col + s].item())] for s in range(SEQ_LEN))
        print(f'  in:   "{inp}"')
        print(f'  want: "{tgt}"')
        print(f'  got:  "{pred}"\n')

    seed_pos = corpus.find("hello world. ")
    if seed_pos == -1 or seed_pos + SEQ_LEN >= len(corpus):
        seed_pos = 0
    window = corpus[seed_pos: seed_pos + SEQ_LEN]
    generated = window

    for _ in range(GEN_LEN):
        enc = torch.zeros(SEQ_LEN, in_dim, device=DEVICE)
        for s, ch in enumerate(window):
            enc[s, c2i[ch]] = 1.0
            enc[s, vocab_size + s] = 1.0
        with torch.no_grad():
            last_logits = model(enc)[-1].cpu()
        next_idx = sample_top_k(last_logits, generated, i2c)
        next_ch = i2c[next_idx]
        generated += next_ch
        window = window[1:] + next_ch

    print("--- Generated text ---")
    print(f'Seed:      "{corpus[seed_pos: seed_pos + SEQ_LEN]}"')
    print(f'Generated: "{generated}"')


if __name__ == "__main__":
    main()