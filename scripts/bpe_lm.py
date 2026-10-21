"""Byte-level BPE preprocessing and explicit Juzhen/PyTorch LM training.

Tokenization is Python-only; Juzhen reads checked little-endian int32 IDs.
All tokenizer, dataset, checkpoint and evaluation paths are caller-supplied.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('RAYON_NUM_THREADS', '4')
import numpy as np
import text8_lm as lm


def chunks(text, size=1 << 20):
    """Cut before whitespace runs, preserving byte-level word prefixes."""
    start = 0
    while start < len(text):
        end = min(len(text), start+size)
        while end < len(text) and not (text[end].isspace() and not text[end-1].isspace()):
            end += 1
        yield text[start:end]
        start = end


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_metadata(directory):
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text())
    payload = {k: v for k, v in meta.items() if k != 'data_id'}
    if meta['data_id'] != fingerprint(payload):
        raise ValueError('Prepared BPE metadata changed')
    expected = {'tokenizer.json', 'token-byte-lengths.bin', 'train.bin', 'valid.bin', 'test.bin'}
    if meta.get('protocol') != 'byte-bpe-v1' or set(meta['files']) != expected:
        raise ValueError('Invalid BPE metadata schema')
    for name, digest in meta['files'].items():
        if lm.sha256(directory/name) != digest:
            raise ValueError('Prepared BPE file changed: '+name)
    return meta


def prepare(source, directory, vocab_size=8192, mount=None):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    import tokenizers
    lm.require_mount(mount)
    source, directory = Path(source).resolve(), Path(directory).resolve()
    raw = source.read_bytes()
    text = raw.decode('utf-8')
    if len(text) < 1000 or not 256 <= vocab_size <= 65536:
        raise ValueError('Need at least 1000 Unicode characters and vocab_size 256..65536')
    source_sha = hashlib.sha256(raw).hexdigest()
    if (directory/'metadata.json').exists():
        meta = load_metadata(directory)
        if meta['source_sha256'] != source_sha or meta['requested_vocab'] != vocab_size:
            raise ValueError('Existing BPE data belongs to another source/tokenizer protocol')
        return meta
    lm.require_mount(mount)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('Choose an empty BPE data directory')
    bounds = [0, len(text)*90//100, len(text)*95//100, len(text)]
    train = text[:bounds[1]]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator(chunks(train), trainer=trainer)
    tokenizer.save(str(directory/'tokenizer.json'))
    vocab = tokenizer.get_vocab_size()
    lengths = np.asarray([len(tokenizer.id_to_token(i)) for i in range(vocab)], dtype='<u4')
    lengths.tofile(directory/'token-byte-lengths.bin')
    splits, byte_start = {}, 0
    for i, name in enumerate(('train', 'valid', 'test')):
        split = text[bounds[i]:bounds[i+1]]
        count, byte_count = 0, 0
        temporary = directory/(name+'.bin.tmp')
        with temporary.open('wb') as f:
            for part in chunks(split):
                ids = tokenizer.encode(part, add_special_tokens=False).ids
                if not ids or min(ids) < 0 or max(ids) >= vocab:
                    raise ValueError('Invalid tokenizer output')
                if tokenizer.decode(ids, skip_special_tokens=False) != part:
                    raise ValueError('Tokenizer round trip failed')
                encoded = np.asarray(ids, dtype='<i4')
                if int(lengths[encoded].sum()) != len(part.encode('utf-8')):
                    raise ValueError('Byte lengths do not reproduce source text')
                encoded.tofile(f)
                count += len(ids)
                byte_count += len(part.encode('utf-8'))
        temporary.replace(directory/(name+'.bin'))
        splits[name] = dict(char_start=bounds[i], char_end=bounds[i+1], byte_start=byte_start,
                            byte_end=byte_start+byte_count, raw_bytes=byte_count, tokens=count,
                            sha256=lm.sha256(directory/(name+'.bin')))
        byte_start += byte_count
    files = {name: lm.sha256(directory/name) for name in
             ('tokenizer.json', 'token-byte-lengths.bin', 'train.bin', 'valid.bin', 'test.bin')}
    meta = dict(protocol='byte-bpe-v1', source_sha256=source_sha, source_bytes=len(raw),
                train_source_sha256=hashlib.sha256(train.encode('utf-8')).hexdigest(),
                requested_vocab=vocab_size, vocab=vocab, token_bytes=4, endian='little',
                tokenizers_version=tokenizers.__version__, normalizer=None, special_tokens=[],
                split_policy='contiguous 90/5/5 Unicode-character ranges, encoded independently',
                chunk_policy='before whitespace runs, ~1Mi Unicode characters', splits=splits, files=files)
    meta['data_id'] = fingerprint(meta)
    lm.write_json(directory/'metadata.json', meta)
    return meta


def native_environment(c, output, device, stop=None, prompt_ids=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('T8_')}
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NVIDIA_TF32_OVERRIDE='0', NVIDIA_TF32='0', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(output))
    names = ('dim', 'layers', 'heads', 'seq', 'batch', 'steps', 'warmup', 'eval_every',
             'eval_batches', 'test_batches', 'seed', 'generate', 'vocab', 'token_bytes', 'data', 'data_id')
    env.update({'T8_'+name.upper(): str(c[name]) for name in names})
    env.update(T8_OUTPUT=str(output), T8_EXPECT_DEVICE=device,
               T8_PROMPT_IDS=str(prompt_ids or c['prompt_ids']))
    if stop is not None:
        env['T8_STOP_AFTER'] = str(stop)
    return env


def decode_output(c, directory):
    directory = Path(directory)
    path = directory/'generated.ids'
    if path.exists():
        ids = np.fromfile(path, dtype='<i4').tolist()
        if not ids or min(ids) < 0 or max(ids) >= c['vocab']:
            raise ValueError('Invalid generated token IDs')
        text = lm.decode_generated(c, ids)
        (directory/'sample.txt').write_text(text+'\n', encoding='utf-8')
        return text


def report(directory):
    directory = Path(directory)
    c = json.loads((directory/'config.json').read_text())
    meta = load_metadata(c['data'])
    parity = json.loads((directory/'parity.json').read_text())
    summary = json.loads((directory/'juzhen/summary.json').read_text())
    if summary['steps'] != c['steps']:
        raise ValueError('Training is incomplete')
    data = np.memmap(Path(c['data'])/'test.bin', mode='r', dtype='<i4')
    lengths = np.fromfile(Path(c['data'])/'token-byte-lengths.bin', dtype='<u4')
    target_bytes = 0
    for i in range(c['test_batches']):
        _, y = lm.batch(data, i, c, False, c['test_batches'])
        target_bytes += int(lengths[y].sum())
    native = dict(summary, sampled_target_bytes=target_bytes,
                  bits_per_byte=summary['test_loss']*summary['test_tokens']/target_bytes/math.log(2))
    hist = lm.rows(directory/'juzhen/metrics.csv')
    if not all(math.isfinite(float(row[k])) for row in hist for k in ('train_loss', 'val_loss', 'grad_norm') if row[k]):
        raise AssertionError('Nonfinite training history')
    paired = None
    if (directory/'torch/summary.json').exists():
        candidate = json.loads((directory/'torch/summary.json').read_text())
        if candidate['steps'] == c['steps']:
            gap = abs(candidate['test_loss']-summary['test_loss'])
            if gap > .12:
                raise AssertionError('Full-run Torch/Juzhen test loss gap exceeds 0.12')
            paired = dict(torch=candidate, test_loss_gap=gap)
    steady = [float(row['step_ms']) for row in hist if int(row['step']) > c['warmup']]
    result = dict(config=c, data=meta, parity=parity, native=native, torch_full_run=paired,
                  initial_validation_loss=float(hist[0]['val_loss']), final_validation_loss=float(hist[-1]['val_loss']),
                  median_step_ms=float(np.median(steady)) if steady else None,
                  metric_note='CE and perplexity are per BPE token. Bits/byte uses the exact byte lengths of sampled target tokens; windows differ from character experiments.')
    files = ['examples/demo_text8_lm.cu', 'scripts/text8_lm.py', 'scripts/bpe_lm.py', 'tests/testBPELM.py']
    source = Path(__file__).resolve().parents[1]
    result['source_sha256'] = {f: lm.sha256(source/f) for f in files if (source/f).exists()}
    lm.write_json(directory/'results.json', result)
    text = decode_output(c, directory/'juzhen')
    lines = [f'Byte-level BPE LM: vocab {c["vocab"]}, {summary["parameters"]:,} parameters, {c["steps"]} updates.',
             f'Tokenizer learned only from {meta["splits"]["train"]["raw_bytes"]:,} training bytes; no normalization, 256 byte alphabet, no special tokens.',
             f'Validation CE/token: {result["initial_validation_loss"]:.5f} -> {result["final_validation_loss"]:.5f}.',
             f'Test CE/token={summary["test_loss"]:.6f}; token perplexity={summary["test_ppl"]:.6f}; bits/byte={native["bits_per_byte"]:.6f}.',
             f'Test estimates use {summary["test_tokens"]:,} tokens / {target_bytes:,} target bytes in sampled windows, not the entire split.',
             'Token perplexities cannot be compared directly with character perplexities. Bits/byte is byte-normalized but the sampled windows also differ.',
             'Architecture remains pre-LN, ReLU, tied embedding/output weights; FP32, TF32 off. This is not a full GPT-2 replica.',
             f'First-step gradient/update parity: {parity}', f'Full Torch training comparison: {paired}',
             f'Steady median step ms: {result["median_step_ms"]}', '', text or 'Generation disabled.']
    (directory/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:8]), flush=True)
    return result


def run(args):
    from tokenizers import Tokenizer
    lm.require_mount(args.storage_mount)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.prepared_data:
        prepared = args.prepared_data.resolve()
        meta = load_metadata(prepared)
        if not (out/'data').exists():
            (out/'data').symlink_to(prepared, target_is_directory=True)
        if (out/'data').resolve() != prepared:
            raise ValueError('Run points to another prepared dataset')
    else:
        meta = prepare(args.input, out/'data', args.vocab_size, args.storage_mount)
    names = ('dim', 'layers', 'heads', 'seq', 'batch', 'steps', 'warmup', 'eval_every', 'eval_batches', 'test_batches', 'seed', 'generate')
    c = {name: getattr(args, name) for name in names}
    if c['dim'] % c['heads'] or not 1 <= c['warmup'] < c['steps']:
        raise ValueError('Invalid heads/warmup')
    if args.stop_after is not None and not 1 <= args.stop_after <= c['steps']:
        raise ValueError('Invalid stop-after')
    if min(x['tokens'] for x in meta['splits'].values()) <= c['seq']:
        raise ValueError('A BPE split is too short for the requested context')
    c.update(vocab=meta['vocab'], token_bytes=4, data=str(out/'data'), data_id=meta['data_id'],
             tokenizer=str(out/'data/tokenizer.json'), tokenizer_sha256=meta['files']['tokenizer.json'],
             prompt=args.prompt, prompt_ids=str(out/'prompt.ids'), device=args.device,
             revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent, text=True).strip())
    if not args.prompt:
        raise ValueError('Prompt cannot be empty')
    if (out/'config.json').exists():
        saved = json.loads((out/'config.json').read_text())
        if {k: v for k, v in saved.items() if k != 'revision'} != {k: v for k, v in c.items() if k != 'revision'}:
            raise ValueError('Existing BPE run uses another protocol')
        c = saved
    else:
        lm.write_json(out/'config.json', c)
    ids = Tokenizer.from_file(c['tokenizer']).encode(args.prompt, add_special_tokens=False).ids
    np.asarray(ids, dtype='<i4').tofile(c['prompt_ids'])
    for b in ('juzhen', 'torch'):
        (out/b).mkdir(exist_ok=True)
    if args.juzhen_only and (not (out/'parity.json').exists() or not json.loads((out/'parity.json').read_text()).get('passed')):
        raise ValueError('Run once with --stop-after 2 to establish parity before --juzhen-only')
    env = native_environment(c, out/'juzhen', args.device, args.stop_after)
    lm.monitored([str(args.binary.resolve())], out, 'juzhen', env, args.storage_mount)
    decode_output(c, out/'juzhen')
    if not args.juzhen_only:
        command = [sys.executable, str(Path(__file__).with_name('text8_lm.py')), '_torch', '--config', str(out/'config.json'),
                   '--initial', str(out/'juzhen/initial.params'), '--output', str(out/'torch'), '--device', args.device]
        if args.stop_after is not None:
            command += ['--stop-after', str(args.stop_after)]
        lm.monitored(command, out, 'torch', env, args.storage_mount)
        parity = {name: lm.compare_archives(lm.read_archive(out/'juzhen'/name), lm.read_archive(out/'torch'/name), 'BPE '+name)
                  for name in ('first.grad', 'first.params')}
        for key, index in (('val_loss', 0), ('train_loss', 1)):
            np.testing.assert_allclose(float(lm.rows(out/'juzhen/metrics.csv')[index][key]),
                                       float(lm.rows(out/'torch/metrics.csv')[index][key]), atol=3e-5, rtol=1e-4)
        lm.write_json(out/'parity.json', dict(passed=True, **parity))
    if args.stop_after is None or args.stop_after == c['steps']:
        report(out)


def generate(args):
    from tokenizers import Tokenizer
    lm.require_mount(args.storage_mount)
    out = args.output.resolve()
    c = json.loads((out/'config.json').read_text())
    meta = load_metadata(c['data'])
    if meta['data_id'] != c['data_id'] or not args.prompt or not 1 <= args.tokens <= 4096:
        raise ValueError('Changed tokenizer/data or invalid generation request')
    ids = Tokenizer.from_file(c['tokenizer']).encode(args.prompt, add_special_tokens=False).ids
    path = out/'generation-prompt.ids'
    np.asarray(ids, dtype='<i4').tofile(path)
    if not math.isfinite(args.temperature) or args.temperature < 0 or not 0 <= args.top_k <= c['vocab']:
        raise ValueError('Invalid temperature/top-k')
    checkpoint = out/'juzhen'/(args.checkpoint+'.ckpt')
    if not checkpoint.exists():
        raise ValueError('Requested checkpoint is unavailable: '+str(checkpoint))
    destination = args.destination.resolve() if args.destination else out/'juzhen'
    lm.require_mount(args.storage_mount)
    destination.mkdir(parents=True, exist_ok=True)
    c['generate'] = args.tokens
    sample_seed = c['seed']+1000 if args.sample_seed is None else args.sample_seed
    env = native_environment(c, out/'juzhen', c['device'], prompt_ids=path)
    env.update(T8_GENERATE_ONLY='1', T8_CHECKPOINT=str(checkpoint), T8_GENERATION_OUTPUT=str(destination),
               T8_TEMPERATURE=str(args.temperature), T8_TOP_K=str(args.top_k), T8_SAMPLE_SEED=str(sample_seed))
    lm.monitored([str(args.binary.resolve())], out, 'generation', env, args.storage_mount)
    lm.write_json(destination/'generation-config.json', dict(prompt=args.prompt, tokens=args.tokens,
                  temperature=args.temperature, top_k=args.top_k, sample_seed=sample_seed,
                  checkpoint=str(checkpoint), checkpoint_sha256=lm.sha256(checkpoint), data_id=c['data_id']))
    print(decode_output(c, destination), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--vocab-size', type=int, default=8192)
    p.add_argument('--storage-mount', type=Path)
    p = sub.add_parser('run')
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input', type=Path)
    inputs.add_argument('--prepared-data', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--storage-mount', type=Path)
    p.add_argument('--vocab-size', type=int, default=8192)
    p.add_argument('--prompt', default='the meaning of life is ')
    p.add_argument('--stop-after', type=int)
    p.add_argument('--juzhen-only', action='store_true')
    for name, default in dict(dim=384, layers=6, heads=6, seq=128, batch=16, steps=10000,
                              warmup=300, eval_every=500, eval_batches=32, test_batches=128, seed=42, generate=128).items():
        p.add_argument('--'+name.replace('_', '-'), type=int, default=default)
    p = sub.add_parser('generate')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--prompt', default='the meaning of life is ')
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--temperature', type=float, default=.8)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--sample-seed', type=int)
    p.add_argument('--checkpoint', choices=('last', 'best'), default='last')
    p.add_argument('--destination', type=Path)
    p.add_argument('--storage-mount', type=Path)
    p = sub.add_parser('report')
    p.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.input, args.output, args.vocab_size, args.storage_mount), indent=2))
    elif args.command == 'run':
        run(args)
    elif args.command == 'generate':
        generate(args)
    else:
        report(args.output)

if __name__ == '__main__':
    main()
