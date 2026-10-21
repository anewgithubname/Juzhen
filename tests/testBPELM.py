"""BPE integrity, int32 IDs, Torch parity, decoding and process restart."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
try:
    import numpy as np
    import torch
    import tokenizers
except ImportError as error:
    print('SKIP:', error)
    raise SystemExit(77)
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE/'scripts'))
import bpe_lm as bpe
import text8_lm as lm


def check_generation(root, config):
    torch.set_num_threads(4)
    prompt = np.fromfile(config['prompt_ids'], dtype='<i4')
    checks = {}
    for backend in ('juzhen', 'torch'):
        folder = root/'uninterrupted'/backend
        ids = np.fromfile(folder/'generated.ids', dtype='<i4')
        trace = lm.read_archive(folder/'generation.params')['logits']
        assert np.array_equal(ids[:len(prompt)], prompt)
        assert len(ids) == len(prompt)+config['generate']
        assert trace.shape == (config['vocab'], config['generate'])
        assert (folder/'sample.txt').read_bytes().decode('utf-8')[:-1] == lm.decode_generated(config, ids.tolist())
        dynamic = dict(config, batch=1)
        model = lm.torch_model(dynamic, lm.read_archive(folder/'final.params'), 'cpu')
        for block in model.blocks:
            block.attention = 'sdpa'
        worst = 0.
        with torch.no_grad():
            for i in range(config['generate']):
                window = ids[:len(prompt)+i][-config['seq']:]
                dynamic['seq'] = len(window)
                model.positions = torch.arange(len(window))
                expected = model(torch.tensor(window[None].astype(np.int64)))[0, -1].numpy()
                np.testing.assert_allclose(trace[:, i], expected, atol=3e-5, rtol=1e-3)
                worst = max(worst, float(np.abs(trace[:, i]-expected).max()))
        checks[backend] = dict(passed=True, max_abs=worst)
    return checks


def main(binary, output, device):
    output.mkdir(parents=True, exist_ok=True)
    root = output/('run-'+str(os.getpid()))
    root.mkdir()
    words = ['word'+chr(97+i//26)+chr(97+i%26) for i in range(600)]
    base = 'the quick brown fox jumps over the lazy dog '+ ' '.join(words)+' '
    train = (base*((90000//len(base))+1))[:90000]
    held = ('🦄 Unseen, 中文!\n\t '*1000)[:10000]
    held = (held*2)[:10000]
    text = train+held
    assert len(text) == 100000
    raw = root/'source.txt'
    raw.write_text(text, encoding='utf-8')
    prepared = root/'prepared'
    meta = bpe.prepare(raw, prepared, 512)
    assert meta['vocab'] == 512
    tokenizer = tokenizers.Tokenizer.from_file(str(prepared/'tokenizer.json'))
    assert len(tokenizer.encode('🦄', add_special_tokens=False).ids) == 4, 'held-out text leaked into BPE merges'
    lengths = np.fromfile(prepared/'token-byte-lengths.bin', dtype='<u4')
    for name, record in meta['splits'].items():
        ids = np.fromfile(prepared/(name+'.bin'), dtype='<i4')
        expected = text[record['char_start']:record['char_end']]
        assert tokenizer.decode(ids.tolist(), skip_special_tokens=False) == expected
        assert int(lengths[ids].sum()) == len(expected.encode('utf-8'))
        assert ids.min() >= 0 and ids.max() < meta['vocab']
        if name == 'train':
            assert ids.max() > 255
            x, y = lm.batch(ids, 0, dict(seq=8, batch=2, seed=42), True)
            assert np.array_equal(x[:, 1:], y[:, :-1])
    probe = "Hello, café!  中文🙂\r\n\t contractions aren't lost. "+base[:500]
    expected = tokenizer.encode(probe, add_special_tokens=False).ids
    for size in (5, 17, 128):
        joined = [i for part in bpe.chunks(probe, size) for i in tokenizer.encode(part, add_special_tokens=False).ids]
        assert joined == expected, 'chunk boundaries changed tokenization'
    missing = root/'must-not-create'
    try:
        bpe.prepare(raw, missing, 512, root/'not-a-mount')
        raise AssertionError('Unavailable mount accepted')
    except RuntimeError:
        assert not missing.exists()
    for filename in ('tokenizer.json', 'valid.bin'):
        path = prepared/filename
        original = path.read_bytes()
        path.write_bytes(original+b'\n')
        try:
            bpe.prepare(raw, prepared, 512)
            raise AssertionError('Changed BPE file accepted')
        except ValueError:
            pass
        path.write_bytes(original)
    assert bpe.prepare(raw, prepared, 512) == meta
    common = [sys.executable, str(SOURCE/'scripts/bpe_lm.py'), 'run', '--input', str(raw), '--binary', str(binary.resolve()),
              '--device', device, '--vocab-size', '512', '--dim', '32', '--layers', '2', '--heads', '4',
              '--seq', '32', '--batch', '2', '--steps', '12', '--warmup', '2', '--eval-every', '4',
              '--eval-batches', '2', '--test-batches', '2', '--generate', '48']
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NVIDIA_TF32_OVERRIDE='0', NVIDIA_TF32='0', PYTHONDONTWRITEBYTECODE='1',
               T8_TOKEN_BYTES='1', T8_VOCAB='27', T8_GENERATE_ONLY='1', T8_BENCHMARK='1')
    def invoke(name, extra=()):
        with (root/(name+'.log')).open('a') as log:
            subprocess.run(common+['--output', str(root/name), *extra], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Reuse exactly one frozen tokenizer; independent tokenizer training can
    # choose different equal-frequency merge orders and must not define parity.
    for name in ('uninterrupted', 'resumed'):
        (root/name).mkdir()
        (root/name/'data').symlink_to(prepared, target_is_directory=True)
    invoke('uninterrupted')
    invoke('resumed', ['--stop-after', '5'])
    invoke('resumed')
    restart = {}
    for backend in ('juzhen', 'torch'):
        restart[backend] = lm.compare_archives(lm.read_archive(root/'resumed'/backend/'final.params'),
                                               lm.read_archive(root/'uninterrupted'/backend/'final.params'),
                                               'BPE restart '+backend, atol=2e-6, rtol=1e-5)
        assert [int(x['step']) for x in lm.rows(root/'resumed'/backend/'metrics.csv')] == list(range(13))
    c = json.loads((root/'uninterrupted/config.json').read_text())
    generation = check_generation(root, c)
    for name in ('uninterrupted', 'resumed'):
        folder = root/name/'juzhen'
        step, val = (folder/'best-validation.txt').read_text().split()
        candidates = [r for r in lm.rows(folder/'metrics.csv') if r['val_loss'] and int(r['step']) > 0]
        best = min(candidates, key=lambda r: float(r['val_loss']))
        assert int(step) == int(best['step']) and abs(float(val)-float(best['val_loss'])) < 1e-8
        assert (folder/'best.ckpt').stat().st_size > 0
        assert set(lm.read_archive(folder/'best.params')) == set(lm.read_archive(folder/'final.params'))
    native = root/'uninterrupted/juzhen'
    protected = ('last.ckpt', 'metrics.csv', 'initial.params', 'first.grad', 'first.params', 'final.params')
    before = {f: lm.sha256(native/f) for f in protected}
    command = [sys.executable, str(SOURCE/'scripts/bpe_lm.py'), 'generate', '--output', str(root/'uninterrupted'),
               '--binary', str(binary.resolve()), '--tokens', '7', '--prompt', 'Unicode 中文 works!']
    with (root/'generate.log').open('w') as f:
        subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    assert before == {f: lm.sha256(native/f) for f in protected}
    prompt_ids = np.fromfile(root/'uninterrupted/generation-prompt.ids', dtype='<i4')
    generated = np.fromfile(native/'generated.ids', dtype='<i4')
    assert np.array_equal(generated[:len(prompt_ids)], prompt_ids) and len(generated)==len(prompt_ids)+7
    assert (native/'sample.txt').read_text().startswith('Unicode 中文 works!')
    sampling = dict(c, generate=9)
    seed_outputs = []
    for index, (temperature, seed) in enumerate(((0., 1), (0., 99), (.6, 7), (.6, 7), (1., 8))):
        folder = root/f'sampling-{index}'
        folder.mkdir()
        sample_env = bpe.native_environment(sampling, native, device)
        sample_env.update(T8_GENERATE_ONLY='1', T8_GENERATION_OUTPUT=str(folder),
                          T8_TEMPERATURE=str(temperature), T8_TOP_K='20', T8_SAMPLE_SEED=str(seed))
        subprocess.run([str(binary.resolve())], env=sample_env, check=True, capture_output=True, text=True)
        generated_ids = np.fromfile(folder/'generated.ids', dtype='<i4')
        if temperature == 0:
            logits = lm.read_archive(folder/'generation.params')['logits']
            assert np.array_equal(generated_ids[-9:], logits.argmax(axis=0)), 'Greedy is not argmax'
        seed_outputs.append(generated_ids)
    assert np.array_equal(seed_outputs[0], seed_outputs[1]), 'Greedy depends on seed'
    assert np.array_equal(seed_outputs[2], seed_outputs[3]), 'Sampling is not reproducible'
    assert before == {f: lm.sha256(native/f) for f in protected}, 'Sampling changed training files'
    # Direct native reader rejects negative, too-large and truncated int32 IDs.
    bad = root/'bad-data'
    bad.mkdir()
    out = root/'bad-output'
    out.mkdir()
    wrong = bpe.native_environment(c, out, device)
    wrong['T8_DATA'] = str(bad)
    for payload in (np.array([-1]*64, dtype='<i4').tobytes(),
                    np.array([c['vocab']]*64, dtype='<i4').tobytes(), b'\x00'*257):
        (bad/'train.bin').write_bytes(payload)
        p = subprocess.run([str(binary.resolve())], env=wrong, capture_output=True, text=True)
        assert p.returncode != 0 and ('invalid encoded token ID' in p.stdout+p.stderr or 'invalid token file size' in p.stdout+p.stderr)
        assert not (out/'last.ckpt').exists()
    wrong = bpe.native_environment(c, native, device)
    wrong['T8_VOCAB'] = str(c['vocab']+1)
    p = subprocess.run([str(binary.resolve())], env=wrong, capture_output=True, text=True)
    assert p.returncode != 0 and 'differs from saved protocol' in p.stdout+p.stderr
    assert before == {f: lm.sha256(native/f) for f in protected}
    result = dict(passed=True, device=device, vocab=meta['vocab'], ids_above255=True,
                  train_only_tokenizer=True, unicode_roundtrip=True, chunk_equivalence=True,
                  changed_data_or_tokenizer_rejected=True, invalid_int32_rejected=True,
                  unavailable_mount_rejected=True, restart=restart, generation=generation,
                  generation_preserves_training=True, best_checkpoint=True, greedy_argmax=True, sampling_reproducible=True, run=str(root))
    lm.write_json(output/'results.json', result)
    print('PASS BPE pipeline', device, restart, generation)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    a = p.parse_args()
    main(a.binary, a.output, a.device)
