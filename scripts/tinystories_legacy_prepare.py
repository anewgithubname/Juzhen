#!/usr/bin/env python3
"""Rebuild the historical full-corpus Metaspace BPE protocol as int32 IDs.

Unlike the byte-BPE experiments this intentionally retains the legacy split and
normalization choices. Validation is the full official validation corpus; there
is no separately held-out test split. All artifacts require the external drive.
"""
from pathlib import Path
import argparse
import json
import os
import io
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
os.environ.setdefault('RAYON_NUM_THREADS', '8')
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import tokenizers
import text8_lm as lm

MOUNT = '/mnt/experiments'
STORAGE = Path(MOUNT)/'Backup/Juzhen-data'
EOT = '<|endoftext|>'


def checked_root(root):
    lm.require_mount(MOUNT)
    root = root.resolve()
    if not root.is_relative_to(STORAGE.resolve()):
        raise ValueError('Models and data must remain on the external drive')
    return root


class BoundedReader(io.RawIOBase):
    def __init__(self,path,start,end):
        self.file=path.open('rb',buffering=0)
        self.file.seek(start)
        self.remaining=end-start
    def readable(self):
        return True
    def readinto(self,buffer):
        count=self.file.readinto(memoryview(buffer)[:min(len(buffer),self.remaining)])
        self.remaining-=count
        return count
    def close(self):
        self.file.close()
        super().close()


def stories(path,start=0,end=None):
    buf = []
    raw=BoundedReader(path,start,path.stat().st_size if end is None else end)
    with io.TextIOWrapper(io.BufferedReader(raw), encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip() == EOT:
                story = ''.join(buf).strip()
                buf.clear()
                if story:
                    yield story
            else:
                buf.append(line)
    story = ''.join(buf).strip()
    if story:
        yield story


def encode(tok, source, destination, start=0, end=None):
    eot = tok.token_to_id(EOT)
    total = count = 0
    pending = []
    temp = destination.with_suffix('.bin.part')
    def flush(out):
        nonlocal total, count
        payload = bytearray()
        for encoded in tok.encode_batch(pending):
            ids = np.asarray(encoded.ids+[eot], dtype='<i4')
            if len(ids) == 1 or ids.min() < 0 or ids.max() >= tok.get_vocab_size():
                raise ValueError('Invalid encoded story')
            payload.extend(ids.tobytes())
            total += len(ids)
            count += 1
        out.write(payload)
        pending.clear()
    with temp.open('wb') as out:
        for story in stories(source,start,end):
            pending.append(story)
            if len(pending) == 1024:
                flush(out)
                if count % 102400 == 0:
                    print(source.name, count, total, flush=True)
        if pending:
            flush(out)
    temp.replace(destination)
    return dict(stories=count, tokens=total, bytes=destination.stat().st_size,
                sha256=lm.sha256(destination))


def encode_shard(args):
    tokenizer,source,destination,start,end=args
    tok=Tokenizer.from_file(str(tokenizer))
    return encode(tok,source,destination,start,end)


def parallel_encode(tokpath,source,destination,workers=8):
    boundaries=[0]
    size=source.stat().st_size
    with source.open('rb') as f:
        for part in range(1,workers):
            f.seek(size*part//workers)
            while line:=f.readline():
                if line.strip()==EOT.encode():
                    break
            boundaries.append(f.tell())
    boundaries.append(size)
    args=[(tokpath,source,destination.with_name('train-shard-%02d.bin'%i),boundaries[i],boundaries[i+1])
          for i in range(workers)]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        records=list(pool.map(encode_shard,args))
    temporary=destination.with_suffix('.bin.part')
    with temporary.open('wb') as out:
        for arg in args:
            with arg[2].open('rb') as f:
                shutil.copyfileobj(f,out,8<<20)
    temporary.replace(destination)
    for arg in args:
        arg[2].unlink()
    return dict(stories=sum(r['stories'] for r in records),tokens=sum(r['tokens'] for r in records),
                bytes=destination.stat().st_size,sha256=lm.sha256(destination))


def prepare(root):
    root = checked_root(root)
    raw = root/'raw'
    sources = json.loads((raw/'sources.json').read_text())
    for record in sources['files'].values():
        if lm.sha256(record['path']) != record['sha256']:
            raise ValueError('Downloaded source hash mismatch')
    dest = root/'data'
    lm.require_mount(MOUNT)
    dest.mkdir(exist_ok=True)
    if (dest/'metadata.json').exists():
        return json.loads((dest/'metadata.json').read_text())
    tokpath = dest/'tokenizer.json'
    if tokpath.exists():
        tok = Tokenizer.from_file(str(tokpath))
    else:
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.Metaspace()
        tok.decoder = decoders.Metaspace()
        trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=[EOT], show_progress=True)
        print('Training full-corpus legacy Metaspace BPE-4096', flush=True)
        tok.train([sources['files']['train']['path']], trainer)
        tok.save(str(tokpath))
    assert tok.get_vocab_size() == 4096
    records = {}
    for split in ('valid', 'train'):
        print('Encoding', split, flush=True)
        records[split] = (parallel_encode(tokpath, Path(sources['files'][split]['path']), dest/(split+'.bin')) if split=='train'
                          else encode(tok, Path(sources['files'][split]['path']), dest/(split+'.bin')))
    sample = 'Once upon a time, there was a little fox who loved to play.'
    assert tok.decode(tok.encode(sample).ids) == sample
    meta = dict(protocol='tinystories-legacy-metaspace-v1', sources=sources,
                vocab=4096, token_bytes=4, endian='little', eot_id=tok.token_to_id(EOT),
                tokenizers_version=tokenizers.__version__, splits=records,
                tokenizer_sha256=lm.sha256(tokpath),
                split_policy='full official train and validation, no independent test',
                encoding_policy='legacy story delimiter, strip each story, Metaspace BPE, append EOT',
                limitations='Original tokenizer/weights absent; rebuilt merges and fresh seeded weights.')
    import hashlib
    meta['data_id'] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    lm.write_json(dest/'metadata.json', meta)
    print(json.dumps(meta, indent=2), flush=True)
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    prepare(parser.parse_args().root)
