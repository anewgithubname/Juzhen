"""Data-boundary, protocol, multi-layer Torch parity and process-restart checks."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
try:
    import numpy as np
    import torch
    import matplotlib
except ImportError as error:
    print('SKIP:', error)
    sys.exit(77)

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('text8_lm', SOURCE/'scripts/text8_lm.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def check_generation(run, config):
    # Compare saved native/Torch logits with an independent variable-length
    # causal forward. No padding is used in this reference. The sample spans
    # short prefixes, the full context, and the sliding-window boundary.
    torch.set_num_threads(4)
    prompt = 'the meaning of life is '
    checks = {}
    for backend in ('juzhen', 'torch'):
        root = run/'uninterrupted'/backend
        text = (root/'sample.txt').read_text().rstrip('\n')
        trace = experiment.read_archive(root/'generation.params')['logits']
        assert text.startswith(prompt) and len(text) == len(prompt)+config['generate']
        assert trace.shape == (27, config['generate'])
        dynamic = dict(config, batch=1)
        model = experiment.torch_model(dynamic, experiment.read_archive(root/'final.params'), 'cpu')
        for block in model.blocks:
            block.attention = 'sdpa'
        maximum, worst = config['seq'], 0.
        with torch.no_grad():
            for i in range(config['generate']):
                window = text[:len(prompt)+i][-maximum:]
                dynamic['seq'] = len(window)
                model.positions = torch.arange(len(window))
                ids = torch.tensor([[experiment.ALPHABET.index(ch) for ch in window]])
                expected = model(ids)[0, -1].numpy()
                np.testing.assert_allclose(trace[:, i], expected, atol=3e-5, rtol=1e-3,
                                           err_msg=backend+' unpadded causal generation step '+str(i))
                worst = max(worst, float(np.abs(trace[:, i]-expected).max()))
        checks[backend] = dict(passed=True, max_abs=worst, steps=config['generate'])
    return checks


def main(binary, output, device):
    output.mkdir(parents=True, exist_ok=True)
    # Repeated test runs must not accidentally just reload finished checkpoints.
    run = output/('run-'+str(os.getpid()))
    run.mkdir()
    raw = run/'source.txt'
    raw.write_text('the quick brown fox jumps over the lazy dog this is a language model test '*200)
    prepared = run/'prepared'
    meta = experiment.prepare(raw, prepared)
    source = raw.read_bytes()
    encoded = bytes(0 if ch == 32 else ch-96 for ch in source)
    joined = b''.join((prepared/(name+'.bin')).read_bytes() for name in ('train', 'valid', 'test'))
    assert joined == encoded
    bounds = [(v['start'], v['end']) for v in meta['splits'].values()]
    assert bounds[0][1] == bounds[1][0] and bounds[1][1] == bounds[2][0]
    assert experiment.splitmix(0) == 0xe220a8397b1dcdaf
    c = dict(seq=8, batch=2, seed=42)
    for name in meta['splits']:
        data = np.fromfile(prepared/(name+'.bin'), dtype=np.uint8)
        for step in (0, 1, 100000):
            x, y = experiment.batch(data, step, c, True)
            assert np.array_equal(x[:, 1:], y[:, :-1])
            x2, y2 = experiment.batch(data, step, c, True)
            assert np.array_equal(x, x2) and np.array_equal(y, y2)
        experiment.batch(data, 0, dict(c, batch=1), False, 1)
    no_output = run/'must-not-create'
    try:
        experiment.prepare(raw, no_output, run/'not-a-mount')
        raise AssertionError('Unavailable storage accepted')
    except RuntimeError:
        assert not no_output.exists()
    path = prepared/'valid.bin'
    original = path.read_bytes()
    path.write_bytes(b'\x00'+original[1:])
    # Make sure the fixture actually changed, independent of its first token.
    if path.read_bytes() == original:
        path.write_bytes(b'\x01'+original[1:])
    try:
        experiment.prepare(raw, prepared)
        raise AssertionError('Changed prepared split accepted')
    except ValueError:
        pass
    path.write_bytes(original)
    # Width 32 exercises CUDA's cooperative LayerNorm as well as fused ReLU
    # backward during multi-layer Torch parity and separate-process restart.
    common = [sys.executable, str(SOURCE/'scripts/text8_lm.py'), 'run', '--binary', str(binary.resolve()),
              '--input', str(raw), '--device', device, '--dim', '32', '--layers', '2', '--heads', '4',
              '--seq', '32', '--batch', '2', '--steps', '12', '--warmup', '2', '--eval-every', '4',
              '--eval-batches', '2', '--test-batches', '2', '--generate', '48']
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1', NVIDIA_TF32_OVERRIDE='0', NVIDIA_TF32='0')
    env.update(T8_BENCHMARK='1', T8_STOP_AFTER='1', T8_GENERATE_ONLY='1')  # The CLI must own its protocol despite stale native flags.
    def invoke(name, extra=()):
        with (run/(name+'-outer.log')).open('a') as log:
            subprocess.run(common+['--output', str(run/name), *extra], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    invoke('uninterrupted')
    invoke('resumed', ['--stop-after', '5'])
    invoke('resumed')
    checks = {}
    for backend in ('juzhen', 'torch'):
        expected = experiment.read_archive(run/'uninterrupted'/backend/'final.params')
        actual = experiment.read_archive(run/'resumed'/backend/'final.params')
        checks[backend] = experiment.compare_archives(actual, expected, backend+' process restart', atol=2e-6, rtol=1e-5)
        history = experiment.rows(run/'resumed'/backend/'metrics.csv')
        assert [int(r['step']) for r in history] == list(range(13)), 'Missing/duplicated resumed steps'
    config = json.loads((run/'uninterrupted/config.json').read_text())
    generation_checks = check_generation(run, config)
    native_root = run/'uninterrupted/juzhen'
    protected = ('last.ckpt', 'metrics.csv', 'initial.params', 'first.grad', 'first.params', 'final.params', 'run-spec.txt')
    hashes = {name: experiment.sha256(native_root/name) for name in protected}
    generate_env = {k: v for k, v in env.items() if not k.startswith('T8_')}
    generate_env.update(T8_GENERATE_ONLY='1', T8_EXPECT_DEVICE=device, T8_DATA=config['data'],
                        T8_DATA_ID=config['data_id'], T8_OUTPUT=str(native_root))
    for key in ('dim', 'layers', 'heads', 'seq', 'batch', 'steps', 'warmup', 'eval_every', 'eval_batches', 'test_batches', 'seed', 'generate'):
        generate_env['T8_'+key.upper()] = str(config[key])
    generated = subprocess.run([str(binary.resolve())], env=generate_env, capture_output=True, text=True)
    assert generated.returncode == 0, generated.stdout+generated.stderr
    assert hashes == {name: experiment.sha256(native_root/name) for name in protected}, 'generation modified training artifacts'
    assert generation_checks == check_generation(run, config)

    wrong = dict(env, T8_DATA=config['data'], T8_OUTPUT=str(run/'uninterrupted/juzhen'), T8_DATA_ID=config['data_id'])
    for key in ('dim', 'layers', 'heads', 'seq', 'batch', 'steps', 'warmup', 'eval_every', 'eval_batches', 'test_batches', 'seed', 'generate'):
        wrong['T8_'+key.upper()] = str(config[key])
    wrong.pop('T8_BENCHMARK', None)
    wrong.pop('T8_GENERATE_ONLY', None)
    wrong.pop('T8_STOP_AFTER', None)
    wrong['T8_EXPECT_DEVICE'] = device
    wrong['T8_SEED'] = '43'
    rejected = subprocess.run([str(binary.resolve())], env=wrong, capture_output=True, text=True)
    assert rejected.returncode != 0 and 'differs from saved protocol' in rejected.stdout+rejected.stderr
    wrong['T8_EXPECT_DEVICE'] = 'cpu' if device == 'cuda' else 'cuda'
    wrong_device = subprocess.run([str(binary.resolve())], env=wrong, capture_output=True, text=True)
    assert wrong_device.returncode != 0 and 'binary backend differs' in wrong_device.stdout+wrong_device.stderr
    try:
        experiment.compare_archives({'x': np.array([[np.nan]])}, {'x': np.array([[np.nan]])}, 'NaN fixture')
        raise RuntimeError('Nonfinite parity fixture accepted')
    except AssertionError:
        pass
    experiment.write_json(output/'results.json', dict(passed=True, device=device, restart=checks, run=str(run),
                          boundary_checks=True, corrupted_split_rejected=True, wrong_resume_protocol_rejected=True,
                          unavailable_mount_rejected=True, multilayer_torch_parity=True, generation=generation_checks,
                          generation_only_preserves_training_files=True))
    print('PASS text8 pipeline:', device, checks)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    a = p.parse_args()
    main(a.binary, a.output, a.device)
