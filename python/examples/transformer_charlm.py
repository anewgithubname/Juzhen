"""examples/demo_transformer.cu in Python: a character-level transformer LM.

Each token is a character and its position, both one-hot (built on the
device with nn.one_hot), embedded by a linear layer, passed through causal
transformer blocks and projected to next-character scores. Training samples
random windows from the first 90% of the corpus, evaluates on the last 10%
every 1000 steps, keeps the best weights and stops early when the
validation loss stalls. Then it shows predictions and generates text.

The corpus is --corpus, else datasets/enwik8, else datasets/corpus.txt,
else a short built-in passage (the opening of Alice in Wonderland), too
short to learn more than memorizing it. For enwik8:

    curl -L https://mattmahoney.net/dc/enwik8.zip -o datasets/enwik8.zip
    unzip -o datasets/enwik8.zip -d datasets/

    python transformer_charlm.py [--device gpu] [--steps 30000] [--corpus FILE]
"""

import argparse
import math
import os
import time

import numpy as np

import juzhen as jz
from juzhen import nn, optim

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="gpu" if jz.has_gpu() else "cpu")
parser.add_argument("--steps", type=int, default=30000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--corpus", help="a text file to train on")
args = parser.parse_args()
device = args.device

ALICE = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of having "
    "nothing to do: once or twice she had peeped into the book her sister was reading, but it had "
    "no pictures or conversations in it, and what is the use of a book, thought Alice, without "
    "pictures or conversations? So she was considering in her own mind, as well as she could, for "
    "the hot day made her feel very sleepy and stupid, whether the pleasure of making a daisy-chain "
    "would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit "
    "with pink eyes ran close by her. There was nothing so very remarkable in that; nor did Alice "
    "think it so very much out of the way to hear the Rabbit say to itself, Oh dear! Oh dear! I "
    "shall be late! But when the Rabbit actually took a watch out of its waistcoat-pocket, and "
    "looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind "
    "that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out "
    "of it, and burning with curiosity, she ran across the field after it, and fortunately was just "
    "in time to see it pop down a large rabbit-hole under the hedge. In another moment down went "
    "Alice after it, never once considering how in the world she was to get out again. ")


def load_corpus():
    folder = os.path.join(jz._core.project_dir, "datasets")
    candidates = [args.corpus] if args.corpus else [os.path.join(folder, n) for n in ("enwik8", "corpus.txt")]
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 64:
            return np.fromfile(path, dtype=np.uint8), path
    if args.corpus:
        raise SystemExit(f"cannot read the corpus {args.corpus} (or it is too short)")
    return np.frombuffer(ALICE.encode(), dtype=np.uint8), "built in (Alice in Wonderland)"


def readable(codes):
    text = bytes(int(c) for c in codes).decode("latin-1")
    return "".join("/" if ch == "\n" else ch if " " <= ch < "\x7f" else "." for ch in text)


jz.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

corpus, source = load_corpus()
symbols = np.unique(corpus)                      # the byte values that occur
V = len(symbols)
code_to_id = np.zeros(256, dtype=np.int64)
code_to_id[symbols] = np.arange(V)
ids = code_to_id[corpus]

seq_len, d_model, d_k, d_ff, heads, blocks = 64, 256, 256, 1024, 8, 4
log_every, patience = 1000, 6
peak_lr, warmup = 5e-4, 1000
min_lr = peak_lr * 0.1

split = len(ids) * 9 // 10
train_lo, train_hi = 0, split - seq_len          # window starts [lo, hi)
val_lo, val_hi = split, len(ids) - seq_len
if train_hi <= train_lo or val_hi <= val_lo:
    raise SystemExit(f"the corpus is too small for seq_len={seq_len}")
batch = min(64, train_hi - train_lo, val_hi - val_lo)
positions = np.tile(np.arange(seq_len), batch)   # token b * seq_len + s is at position s


def lr_at(step):
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, args.steps - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * t))


def tokens(chars, pos):
    """One-hot character and position side by side: one row per token."""
    return jz.hstack([nn.one_hot(chars, V, device=device), nn.one_hot(pos, seq_len, device=device)])


def make_batch(starts):
    window = starts[:, None] + np.arange(seq_len)
    inputs, targets = ids[window].ravel(), ids[window + 1].ravel()
    return tokens(inputs, positions), nn.one_hot(targets, V, device=device), targets


def make_model(n_sequences):
    return nn.Sequential(
        nn.Linear(V + seq_len, d_model),
        *[nn.Transformer(d_model, d_k, d_ff, seq_len, n_sequences, heads=heads) for _ in range(blocks)],
        nn.Linear(V),
        device=device)


def evaluate(fixed):
    X, Y, targets = fixed
    logits = model.predict(X)
    loss = nn.softmax_cross_entropy(logits, Y).item()
    accuracy = np.mean(jz.argmax(logits).numpy()[:, 0] == targets)
    return loss, 100 * accuracy


print("=== Transformer Char-LM Demo ===")
print(f"Corpus: {source}, {len(ids)} characters, split at {split} (90/10)")
print(f"Preview: \"{readable(corpus[:80])}...\"")
print(f"Vocabulary: {V} symbols")
print(f"Network: embed({V + seq_len}->{d_model}) -> Transformer({d_model},{d_k},{d_ff},{heads}h) "
      f"x{blocks} -> proj({d_model}->{V})")
print(f"Training: {batch} sequences x {seq_len} tokens per step, up to {args.steps} steps, "
      f"device {device}\n")

model = make_model(batch)
opt = optim.AdamW(model.parameters(), lr=lr_at(0), weight_decay=0)

# fixed, evenly spaced windows for comparable train and validation numbers
spread = lambda lo, hi: lo + (np.arange(batch) * (hi - lo)) // batch
train_fixed = make_batch(spread(train_lo, train_hi))
val_fixed = make_batch(spread(val_lo, val_hi))

best, best_loss, best_step, stale = model.state_dict(), math.inf, 0, 0
start = time.perf_counter()
for step in range(args.steps):
    opt.lr = lr_at(step)
    X, Y, _ = make_batch(rng.integers(train_lo, train_hi, batch))
    opt.zero_grad()
    model.backward(nn.softmax_cross_entropy(model(X), Y).grad)
    opt.step()

    if step % log_every == 0 or step == args.steps - 1:
        train_loss, _ = evaluate(train_fixed)
        val_loss, val_acc = evaluate(val_fixed)
        improved = val_loss < best_loss
        if improved:
            best, best_loss, best_step, stale = model.state_dict(), val_loss, step, 0
        else:
            stale += 1
        print(f"step {step:5d}  train_loss = {train_loss:.4f}  val_loss = {val_loss:.4f}  "
              f"val_ppl = {math.exp(val_loss):6.2f}  val_acc = {val_acc:5.1f}%  lr = {lr_at(step):.2e}"
              f"{'  *best' if improved else ''}  ({time.perf_counter() - start:.0f} s)")
        if stale >= patience:
            print(f"Early stopping at step {step}: best val_loss = {best_loss:.4f} at step {best_step}.")
            break

# one sequence at a time for inference: the same weights in a batch-1 network
generator = make_model(1)
generator.load_state_dict(best)
one_position = np.arange(seq_len)


def next_scores(window):
    return generator.predict(tokens(window, one_position)).numpy()


print("\n--- Sample predictions (held-out text) ---")
for k in range(3):
    off = val_lo + k * (val_hi - val_lo) // 3
    predicted = next_scores(ids[off:off + seq_len]).argmax(1)
    print(f"  in:   \"{readable(corpus[off:off + seq_len])}\"")
    print(f"  want: \"{readable(corpus[off + 1:off + seq_len + 1])}\"")
    print(f"  got:  \"{readable(symbols[predicted])}\"\n")


def sample_top_k(scores, recent, temperature=0.8, top_k=5):
    """One of the top_k characters, recently seen ones slightly penalized."""
    scores = scores.copy()
    for c in recent[-12:]:
        scores[c] -= 0.02
    top = np.argsort(-scores)[:top_k]
    weights = np.exp((scores[top] - scores[top[0]]) / temperature)
    return top[rng.choice(top_k, p=weights / weights.sum())]


window = list(ids[:seq_len])
generated = list(window)
for _ in range(200):
    nxt = sample_top_k(next_scores(np.array(window))[-1], generated)
    generated.append(nxt)
    window = window[1:] + [nxt]

print("--- Generated text ---")
print(f"Seed:      \"{readable(corpus[:seq_len])}\"")
print(f"Generated: \"{readable(symbols[generated])}\"")
