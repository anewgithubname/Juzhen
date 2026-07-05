#!/usr/bin/env python3
"""Resume step: encode the TinyStories train split with the existing tokenizer."""
import numpy as np
from tokenizers import Tokenizer

BASE = "/mnt/external_hdd/data/nlp/tinystories"
EOT = "<|endoftext|>"
tok = Tokenizer.from_file(f"{BASE}/tokenizer.json")
eot_id = tok.token_to_id(EOT)

out, buf, n = [], [], 0
def flush():
    global n
    s = "".join(buf).strip()
    buf.clear()
    if s:
        out.extend(tok.encode(s).ids)
        out.append(eot_id)
        n += 1
        if n % 100000 == 0:
            print(f"{n} stories, {len(out):,} tokens", flush=True)

with open(f"{BASE}/TinyStoriesV2-GPT4-train.txt", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.strip() == EOT:
            flush()
        else:
            buf.append(line)
flush()
arr = np.array(out, dtype=np.uint16)
arr.tofile(f"{BASE}/train.bin")
print(f"DONE train.bin: {len(arr):,} tokens, {n:,} stories")
