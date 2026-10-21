"""Pinned TinyStories subset, story-disjoint splits and train-only byte BPE.

Raw subsets, token files and provenance are written only under the given mounted
external output. No Hugging Face cache or complete multi-GB corpus is downloaded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import tokenizers
import bpe_lm as bpe
import text8_lm as lm

EOT = '<|endoftext|>'
REPO = 'roneneldan/TinyStories'


def story_hash(story):
    return hashlib.sha256(' '.join(story.split()).encode('utf-8')).hexdigest()


def stream_stories(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Juzhen-TinyStories-experiment/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        lines = []
        for line in response:
            text = line.decode('utf-8')
            if text.strip() == EOT:
                story = ''.join(lines).strip()
                lines.clear()
                if story:
                    yield story
            else:
                lines.append(text)
        story = ''.join(lines).strip()
        if story:
            yield story


def download_subset(out, train_count, heldout_count, mount):
    lm.require_mount(mount)
    raw = out/'raw'
    raw.mkdir(parents=True, exist_ok=True)
    metadata = raw/'subset.json'
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved['requested_train_stories'] != train_count or saved['heldout_stories_per_split'] != heldout_count:
            raise ValueError('Existing subset uses another requested size')
        for name, record in saved['splits'].items():
            if lm.sha256(raw/(name+'.txt')) != record['sha256']:
                raise ValueError('Raw story subset changed')
        return saved
    with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{REPO}', timeout=30) as response:
        revision = json.load(response)['sha']
    urls = {name: f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/TinyStoriesV2-GPT4-{name}.txt?download=true'
            for name in ('train', 'valid')}
    seen = set()
    counts = dict(train=0, valid=0, test=0)
    skipped = dict(train=0, heldout=0)
    paths = {name: raw/(name+'.txt') for name in counts}
    with paths['train'].open('w', encoding='utf-8') as output:
        for story in stream_stories(urls['train']):
            digest = story_hash(story)
            if digest in seen:
                skipped['train'] += 1
                continue
            seen.add(digest)
            output.write(story+'\n'+EOT+'\n')
            counts['train'] += 1
            if counts['train'] % 10000 == 0:
                print('downloaded unique training stories', counts['train'], flush=True)
            if counts['train'] == train_count:
                break
    if counts['train'] != train_count:
        raise RuntimeError('Official training source ended before requested subset')
    with paths['valid'].open('w', encoding='utf-8') as valid, paths['test'].open('w', encoding='utf-8') as test:
        outputs = dict(valid=valid, test=test)
        for story in stream_stories(urls['valid']):
            digest = story_hash(story)
            if digest in seen:
                skipped['heldout'] += 1
                continue
            seen.add(digest)
            name = 'valid' if sum(counts[k] for k in ('valid', 'test')) % 2 == 0 else 'test'
            outputs[name].write(story+'\n'+EOT+'\n')
            counts[name] += 1
            if counts['valid'] == heldout_count and counts['test'] == heldout_count:
                break
    if counts['valid'] != heldout_count or counts['test'] != heldout_count:
        raise RuntimeError('Official heldout source ended before requested subset')
    splits = {name: dict(stories=counts[name], sha256=lm.sha256(path), raw_bytes=path.stat().st_size)
              for name, path in paths.items()}
    result = dict(repository=REPO, revision=revision, source_urls=urls,
                  requested_train_stories=train_count, heldout_stories_per_split=heldout_count,
                  subset_policy='first unique training stories; alternating unique official validation stories for validation and test',
                  deduplication='SHA256 after whitespace normalization, across all three splits',
                  skipped_duplicates=skipped, splits=splits)
    lm.write_json(metadata, result)
    return result


def prepare(out, vocab_size, mount):
    lm.require_mount(mount)
    raw, data = out/'raw', out/'data'
    if (data/'metadata.json').exists():
        return bpe.load_metadata(data)
    data.mkdir(exist_ok=True)
    if any(data.iterdir()):
        raise ValueError('Incomplete data exists; inspect before rerunning')
    source = json.loads((raw/'subset.json').read_text())
    train = (raw/'train.txt').read_text(encoding='utf-8')
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=[EOT],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    print('training TinyStories tokenizer on training stories only', flush=True)
    tokenizer.train_from_iterator(bpe.chunks(train), trainer=trainer)
    tokenizer.save(str(data/'tokenizer.json'))
    vocab = tokenizer.get_vocab_size()
    lengths = np.array([len(tokenizer.id_to_token(i)) for i in range(vocab)], dtype='<u4')
    lengths.tofile(data/'token-byte-lengths.bin')
    splits = {}
    for name in ('train', 'valid', 'test'):
        lm.require_mount(mount)
        text = train if name == 'train' else (raw/(name+'.txt')).read_text(encoding='utf-8')
        count, byte_count = 0, 0
        with (data/(name+'.bin')).open('wb') as output:
            for chunk in bpe.chunks(text):
                ids = tokenizer.encode(chunk, add_special_tokens=False).ids
                assert tokenizer.decode(ids, skip_special_tokens=False) == chunk
                encoded = np.array(ids, dtype='<i4')
                assert int(lengths[encoded].sum()) == len(chunk.encode('utf-8'))
                encoded.tofile(output)
                count += len(ids)
                byte_count += len(chunk.encode('utf-8'))
        splits[name] = dict(tokens=count, raw_bytes=byte_count, stories=source['splits'][name]['stories'],
                            sha256=lm.sha256(data/(name+'.bin')),
                            source_sha256=source['splits'][name]['sha256'])
        print('encoded', name, splits[name], flush=True)
    files = {name: lm.sha256(data/name) for name in ('tokenizer.json', 'token-byte-lengths.bin', 'train.bin', 'valid.bin', 'test.bin')}
    meta = dict(protocol='byte-bpe-v1', source_sha256=bpe.fingerprint(source['splits']),
                source_bytes=sum(r['raw_bytes'] for r in splits.values()),
                train_source_sha256=source['splits']['train']['sha256'], requested_vocab=vocab_size,
                vocab=vocab, token_bytes=4, endian='little', tokenizers_version=tokenizers.__version__,
                normalizer=None, special_tokens=[EOT], eos_token_id=tokenizer.token_to_id(EOT),
                split_policy='story-disjoint official train/validation subset; see raw/subset.json',
                chunk_policy='before whitespace runs, ~1Mi Unicode characters',
                provenance=source, splits=splits, files=files)
    meta['data_id'] = bpe.fingerprint(meta)
    lm.write_json(data/'metadata.json', meta)
    bpe.load_metadata(data)
    print('TinyStories prepared', meta['data_id'], flush=True)
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--storage-mount', type=Path, required=True)
    parser.add_argument('--train-stories', type=int, default=100000)
    parser.add_argument('--heldout-stories', type=int, default=5000)
    parser.add_argument('--vocab-size', type=int, default=8192)
    args = parser.parse_args()
    if min(args.train_stories, args.heldout_stories) <= 0 or not 256 <= args.vocab_size <= 65536:
        parser.error('Positive story counts and vocabulary 256..65536 required')
    lm.require_mount(args.storage_mount)
    args.output.mkdir(parents=True, exist_ok=True)
    download_subset(args.output, args.train_stories, args.heldout_stories, args.storage_mount)
    prepare(args.output, args.vocab_size, args.storage_mount)
