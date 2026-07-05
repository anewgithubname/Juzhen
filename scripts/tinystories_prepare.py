#!/usr/bin/env python3
"""Prepare TinyStories for Juzhen training.

1. Trains a BPE-4096 tokenizer (Metaspace pre-tokenizer, llama-style "▁word"
   pieces) on the TinyStories V2 corpus.
2. Encodes train/valid splits into flat uint16 token-id files (train.bin/val.bin).
3. Exports the tokenizer in a simple line-based format (tokenizer.txt) that the
   C++ demo can parse: vocab strings + merges as id triples.

All inputs/outputs live on the external scratch disk:
    /mnt/external_hdd/data/nlp/tinystories/
"""

import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

BASE = "/mnt/external_hdd/data/nlp/tinystories"
TRAIN_TXT = f"{BASE}/TinyStoriesV2-GPT4-train.txt"
VALID_TXT = f"{BASE}/TinyStoriesV2-GPT4-valid.txt"
VOCAB_SIZE = 4096
EOT = "<|endoftext|>"


def train_tokenizer():
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOT],
        show_progress=True,
    )
    tok.train([TRAIN_TXT], trainer)
    tok.save(f"{BASE}/tokenizer.json")
    return tok


def encode_split(tok, txt_path, bin_path):
    """Stream-encode a text file story by story (separated by EOT lines).

    Tokens are flushed to disk in fixed-size uint16 chunks so memory stays
    bounded regardless of corpus size (a 2 GB corpus is ~600M tokens; holding
    them all as a Python list would need tens of GB of RAM and OOM the box).
    Stories are batched through tok.encode_batch for speed.
    """
    eot_id = tok.token_to_id(EOT)
    n_stories = 0
    n_tokens = 0
    pending = bytearray()          # uint16 little-endian bytes not yet written
    stories = []                   # small batch of raw story strings
    BATCH = 1024
    FLUSH_BYTES = 8 << 20          # write ~8 MB at a time

    def encode_batch(fout):
        nonlocal n_stories, n_tokens
        if not stories:
            return
        for enc in tok.encode_batch(stories):
            ids = enc.ids
            ids.append(eot_id)
            arr = np.asarray(ids, dtype=np.uint16)
            pending.extend(arr.tobytes())
            n_tokens += len(ids)
            n_stories += 1
        stories.clear()
        if len(pending) >= FLUSH_BYTES:
            fout.write(pending)
            pending.clear()

    buf = []
    with open(txt_path, encoding="utf-8", errors="replace") as f, \
         open(bin_path, "wb") as fout:
        for line in f:
            if line.strip() == EOT:
                story = "".join(buf).strip()
                buf.clear()
                if story:
                    stories.append(story)
                    if len(stories) >= BATCH:
                        encode_batch(fout)
            else:
                buf.append(line)
        story = "".join(buf).strip()
        if story:
            stories.append(story)
        encode_batch(fout)
        if pending:
            fout.write(pending)

    print(f"{bin_path}: {n_tokens:,} tokens, {n_stories:,} stories")
    return n_tokens


def export_for_cpp(tok):
    """tokenizer.txt: VOCAB n / n escaped tokens / MERGES m / m id triples."""
    tj = json.loads(open(f"{BASE}/tokenizer.json", encoding="utf-8").read())
    vocab = tj["model"]["vocab"]            # token -> id
    merges = tj["model"]["merges"]          # ["a b", ...] or [[a, b], ...]
    id_to_tok = [None] * len(vocab)
    for t, i in vocab.items():
        id_to_tok[i] = t

    def esc(s):
        return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")

    with open(f"{BASE}/tokenizer.txt", "w", encoding="utf-8") as f:
        f.write(f"VOCAB {len(id_to_tok)}\n")
        for t in id_to_tok:
            f.write(esc(t) + "\n")
        lines = []
        for m in merges:
            a, b = m if isinstance(m, (list, tuple)) else m.split(" ", 1)
            lines.append(f"{vocab[a]} {vocab[b]} {vocab[a + b]}")
        f.write(f"MERGES {len(lines)}\n")
        f.write("\n".join(lines) + "\n")
    print(f"{BASE}/tokenizer.txt: {len(id_to_tok)} tokens, {len(lines)} merges")


def main():
    for p in (TRAIN_TXT, VALID_TXT):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — download TinyStories first")
    # reuse an existing tokenizer (training is fast but deterministic; the
    # encoded .bin files must match whatever tokenizer produced tokenizer.txt)
    if os.path.exists(f"{BASE}/tokenizer.json"):
        print("loading existing tokenizer.json")
        tok = Tokenizer.from_file(f"{BASE}/tokenizer.json")
    else:
        print("training BPE tokenizer...")
        tok = train_tokenizer()
    export_for_cpp(tok)
    encode_split(tok, VALID_TXT, f"{BASE}/val.bin")
    encode_split(tok, TRAIN_TXT, f"{BASE}/train.bin")
    # quick round-trip sanity check
    sample = "Once upon a time, there was a little fox who loved to play."
    ids = tok.encode(sample).ids
    print("round-trip:", repr(tok.decode(ids)))
    print(f"sample encodes to {len(ids)} tokens: {ids[:12]}...")


if __name__ == "__main__":
    main()
