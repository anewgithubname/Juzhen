"""Matched Juzhen/PyTorch character LM experiment; explicit data/output paths.

Run --help. Nothing is downloaded. Optional --storage-mount prevents writing
into an unavailable external mount. See docs/text8-lm.md for the protocol.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '4')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/juzhen-matplotlib')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import numpy as np

ALPHABET = ' abcdefghijklmnopqrstuvwxyz'
MASK64 = (1 << 64) - 1


def require_mount(mount):
    if mount and not os.path.ismount(mount):
        raise RuntimeError(f'Storage is not mounted: {mount}')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(source, directory, mount=None):
    require_mount(mount)
    source, directory = Path(source).resolve(), Path(directory).resolve()
    raw = source.read_bytes()
    if len(raw) < 1000 or any(ch != 32 and not 97 <= ch <= 122 for ch in set(raw)):
        raise ValueError('Expected lower-case ASCII text8 with spaces, at least 1000 bytes')
    fingerprint = hashlib.sha256(raw).hexdigest()
    if (directory/'metadata.json').exists():
        meta = json.loads((directory/'metadata.json').read_text())
        if meta['source_sha256'] != fingerprint or meta['alphabet'] != ALPHABET:
            raise ValueError('Existing prepared data belongs to a different source/protocol')
        for name, record in meta['splits'].items():
            if sha256(directory/(name+'.bin')) != record['sha256']:
                raise ValueError(f'Prepared split changed: {name}')
        return meta
    require_mount(mount)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('Choose an empty directory for prepared data')
    lut = np.zeros(256, dtype=np.uint8)
    lut[97:123] = np.arange(1, 27, dtype=np.uint8)
    encoded = lut[np.frombuffer(raw, dtype=np.uint8)]
    bounds = [0, len(raw)*90//100, len(raw)*95//100, len(raw)]
    meta = dict(protocol=1, source_sha256=fingerprint, source_bytes=len(raw),
                alphabet=ALPHABET, splits={})
    for i, name in enumerate(('train', 'valid', 'test')):
        tokens = encoded[bounds[i]:bounds[i+1]]
        path = directory/(name+'.bin')
        tokens.tofile(path)
        meta['splits'][name] = dict(start=bounds[i], end=bounds[i+1], tokens=len(tokens), sha256=sha256(path))
    write_json(directory/'metadata.json', meta)
    return meta


def splitmix(x):
    x = (x + 0x9e3779b97f4a7c15) & MASK64
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK64
    return x ^ (x >> 31)


def batch(data, index, c, training, batches=0):
    choices = len(data) - c['seq']
    if choices <= 0:
        raise ValueError('Split is shorter than the context')
    if training:
        offsets = [splitmix(c['seed'] + index*c['batch'] + j) % choices for j in range(c['batch'])]
    else:
        denominator = max(1, batches*c['batch']-1)
        offsets = [(choices-1)*(index*c['batch']+j)//denominator for j in range(c['batch'])]
    positions = np.asarray(offsets, dtype=np.int64)[:, None] + np.arange(c['seq']+1)
    if positions.min() < 0 or positions.max() >= len(data):
        raise AssertionError('A batch crossed its split boundary')
    ids = data[positions].astype(np.int64)
    return np.ascontiguousarray(ids[:, :-1]), np.ascontiguousarray(ids[:, 1:])


def learning_rate(step, c):
    if step < c['warmup']:
        return np.float32(.0006*np.float32(step+1)/np.float32(c['warmup'])).item()
    ratio = (step-c['warmup']) / max(1, c['steps']-c['warmup']-1)
    return np.float32(.00006 + (.0006-.00006)*.5*(1+math.cos(math.pi*ratio))).item()


def read_archive(path):
    records = {}
    with Path(path).open('rb') as f:
        def read(n):
            value = f.read(n)
            if len(value) != n:
                raise ValueError('Truncated parameter archive')
            return value
        def integer():
            return struct.unpack('<Q', read(8))[0]
        if read(8) != b'T8PAR001':
            raise ValueError('Invalid parameter archive')
        count, = struct.unpack('<I', read(4))
        if count > 10000:
            raise ValueError('Invalid parameter count')
        for _ in range(count):
            size = integer()
            if size > 4096:
                raise ValueError('Invalid parameter name')
            name = read(size).decode('ascii')
            rows, cols = integer(), integer()
            if not rows or not cols or rows*cols > 100000000:
                raise ValueError('Invalid parameter shape')
            if name in records:
                raise ValueError('Duplicate parameter')
            records[name] = np.frombuffer(read(rows*cols*4), dtype='<f4').reshape(cols, rows).T.copy()
        if f.read(1):
            raise ValueError('Trailing archive data')
    return records


def write_archive(path, records):
    with Path(path).open('wb') as f:
        f.write(b'T8PAR001' + struct.pack('<I', len(records)))
        for name, value in records.items():
            value = np.asarray(value, dtype='<f4')
            name = name.encode('ascii')
            f.write(struct.pack('<Q', len(name)) + name + struct.pack('<QQ', *value.shape))
            f.write(value.T.copy().tobytes())


def token_dtype(c):
    width = c.get('token_bytes', 1)
    if width not in (1, 4):
        raise ValueError('token_bytes must be 1 or 4')
    return np.dtype('<i4') if width == 4 else np.dtype('u1')


def generation_prompt(c):
    if c.get('token_bytes', 1) == 4:
        ids = np.fromfile(c['prompt_ids'], dtype='<i4').tolist()
        if not ids or min(ids) < 0 or max(ids) >= c['vocab']:
            raise ValueError('Invalid token generation prompt')
        return ids
    return [ALPHABET.index(ch) for ch in 'the meaning of life is ']


def decode_generated(c, ids):
    if c.get('token_bytes', 1) == 4:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(c['tokenizer']).decode(ids, skip_special_tokens=False)
    return ''.join(ALPHABET[i] for i in ids)


def torch_model(c, initial, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            d = c['dim']
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.Wq, self.Wk, self.Wv = [nn.Linear(d, d, bias=False) for _ in range(3)]
            self.Wo = nn.Linear(d, d)
            self.ffn1, self.ffn2 = nn.Linear(d, 4*d), nn.Linear(4*d, d)
            self.register_buffer('mask', torch.ones(c['seq'], c['seq'], dtype=torch.bool).triu(1))
            self.attention = 'eager'
        def forward(self, h):
            x = self.ln1(h)
            def heads(p):
                return p.view(-1, c['seq'], c['heads'], c['dim']//c['heads']).transpose(1, 2)
            q, k, v = [heads(layer(x)) for layer in (self.Wq, self.Wk, self.Wv)]
            if self.attention == 'sdpa':
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.)
            else:
                scores = (q @ k.transpose(-2, -1)) * (1/math.sqrt(c['dim']//c['heads']))
                a = scores.masked_fill(self.mask, -1e9).softmax(-1) @ v
            r = h + self.Wo(a.transpose(1, 2).reshape(-1, c['seq'], c['dim']))
            return r + self.ffn2(F.relu(self.ffn1(self.ln2(r))))
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Parameter(torch.empty(c['dim'], c.get('vocab', 27)))
            self.position = nn.Parameter(torch.empty(c['dim'], c['seq']))
            self.blocks = nn.ModuleList([Block() for _ in range(c['layers'])])
            self.legacy = c.get('legacy_tinystories', False)
            if self.legacy:
                self.input_bias = nn.Parameter(torch.empty(c['dim']))
                self.projection = nn.Linear(c['dim'], c['vocab'])
            else:
                self.final = nn.LayerNorm(c['dim'])
            self.register_buffer('positions', torch.arange(c['seq']))
        def forward(self, ids):
            h = F.embedding(ids, self.token.T) + F.embedding(self.positions, self.position.T)
            if self.legacy:
                h = h + self.input_bias
            for block in self.blocks:
                h = block(h)
            return self.projection(h) if self.legacy else self.final(h) @ self.token
        def named_refs(self):
            result = {'token.embedding': self.token, 'position.embedding': self.position}
            if self.legacy:
                result['input.bias'] = self.input_bias
            for i, b in enumerate(self.blocks):
                entries = [('Wq', b.Wq.weight), ('Wk', b.Wk.weight), ('Wv', b.Wv.weight),
                           ('Wo', b.Wo.weight), ('bo', b.Wo.bias), ('W1', b.ffn1.weight),
                           ('b1', b.ffn1.bias), ('W2', b.ffn2.weight), ('b2', b.ffn2.bias),
                           ('ln1.gamma', b.ln1.weight), ('ln1.beta', b.ln1.bias),
                           ('ln2.gamma', b.ln2.weight), ('ln2.beta', b.ln2.bias)]
                result.update({f'blocks.{i}.{n}': p for n, p in entries})
            if self.legacy:
                result.update({'projection.weight': self.projection.weight, 'projection.bias': self.projection.bias})
            else:
                result.update({'final.gamma': self.final.weight, 'final.beta': self.final.bias})
            return result
    model = Model().to(device)
    refs = model.named_refs()
    if refs.keys() != initial.keys():
        raise AssertionError('Juzhen and PyTorch parameter registrations differ')
    with torch.no_grad():
        for name, p in refs.items():
            p.copy_(torch.from_numpy(initial[name]).reshape(p.shape))
    return model


def torch_optimizer(model, fused=False):
    import torch
    decay, no_decay = [], []
    for name, p in model.named_refs().items():
        (no_decay if p.ndim == 1 else decay).append(p)
    return torch.optim.AdamW([dict(params=decay, weight_decay=.1), dict(params=no_decay, weight_decay=0.)],
                            lr=.0006, betas=(.9, .95), eps=1e-8, foreach=False, fused=fused)


def torch_records(model, gradients=False):
    return {name: (p.grad if gradients else p).detach().cpu().numpy().reshape(
        (p.shape[0], p.shape[1] if p.ndim == 2 else 1)).copy() for name, p in model.named_refs().items()}


def compare_archives(actual, expected, label, atol=3e-5, rtol=1e-3):
    if actual.keys() != expected.keys():
        raise AssertionError(f'{label}: parameters differ')
    worst = 0.
    for name, x in actual.items():
        if not np.isfinite(x).all() or not np.isfinite(expected[name]).all():
            raise AssertionError(label+': nonfinite values in '+name)
        np.testing.assert_allclose(x, expected[name], atol=atol, rtol=rtol, equal_nan=False, err_msg=label+': '+name)
        worst = max(worst, float(np.abs(x-expected[name]).max()))
    return dict(passed=True, max_abs=worst, parameters=len(actual), atol=atol, rtol=rtol)


def torch_setup(device):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this PyTorch installation')
    return torch


def torch_train(config_path, initial_path, output, device, stop_after=None):
    torch = torch_setup(device)
    import torch.nn.functional as F
    c = json.loads(Path(config_path).read_text())
    root, output = Path(c['data']), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {k: np.memmap(root/(k+'.bin'), mode='r', dtype=token_dtype(c)) for k in ('train', 'valid', 'test')}
    initial = read_archive(initial_path)
    model = torch_model(c, initial, device)
    opt = torch_optimizer(model)
    checkpoint = output/'last.pt'
    step = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['config'] != c:
            raise ValueError('Torch resume protocol mismatch')
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        step = saved['step']
    rows = []
    if (output/'metrics.csv').exists():
        with (output/'metrics.csv').open() as f:
            rows = [r for r in csv.DictReader(f) if int(r['step']) <= step]
    fields = ['step', 'train_loss', 'lr', 'grad_norm', 'step_ms', 'val_loss']
    def save_rows():
        with (output/'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)
    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()
    @torch.no_grad()
    def evaluate(name, count):
        total = 0.
        for i in range(count):
            x, y = batch(datasets[name], i, c, False, count)
            z = model(torch.from_numpy(x).to(device))
            total += F.cross_entropy(z.flatten(0, 1), torch.from_numpy(y).flatten().to(device)).item()
        return total/count
    if not step:
        val = evaluate('valid', c['eval_batches'])
        rows = [dict(zip(fields, [0, '', '', '', '', val]))]
        save_rows()
        print(f'Torch step=0 val_loss={val:.6f}', flush=True)
    stop = c['steps'] if stop_after is None else stop_after
    while step < stop:
        sync()
        started = time.perf_counter()
        x, y = batch(datasets['train'], step, c, True)
        opt.zero_grad(set_to_none=True)
        z = model(torch.from_numpy(x).to(device))
        loss = F.cross_entropy(z.flatten(0, 1), torch.from_numpy(y).flatten().to(device))
        loss.backward()
        if step == 0:
            raw = torch_records(model, True)
        norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.)
        lr = learning_rate(step, c)
        for group in opt.param_groups:
            group['lr'] = lr
        opt.step()
        value = loss.item()
        sync()
        ms = (time.perf_counter()-started)*1000
        norm_value = norm.item()
        if not math.isfinite(value) or not math.isfinite(norm_value):
            raise RuntimeError('Nonfinite Torch loss/gradient norm')
        step += 1
        if step == 1:
            write_archive(output/'first.grad', raw)
            write_archive(output/'first.params', torch_records(model))
        row = dict(zip(fields, [step, value, lr, norm_value, ms, '']))
        if step % c['eval_every'] == 0 or step == stop:
            row['val_loss'] = evaluate('valid', c['eval_batches'])
        rows.append(row)
        if row['val_loss'] != '':
            save_rows()
            temporary = checkpoint.with_suffix('.pt.tmp')
            torch.save(dict(config=c, step=step, model=model.state_dict(), optimizer=opt.state_dict()), temporary)
            temporary.replace(checkpoint)
            print(f'Torch step={step} train_loss={value:.6f} val_loss={row["val_loss"]:.6f} step_ms={ms:.3f}', flush=True)
    save_rows()
    write_archive(output/'final.params', torch_records(model))
    if step == c['steps']:
        test_loss = evaluate('test', c['test_batches'])
        write_json(output/'summary.json', dict(steps=step, parameters=sum(p.numel() for p in model.parameters()),
                   test_loss=test_loss, test_ppl=math.exp(test_loss), **({'test_bits_per_token': test_loss/math.log(2)} if c.get('token_bytes', 1) == 4 else {'test_bpc': test_loss/math.log(2)}),
                   test_tokens=c['test_batches']*c['seq']*c['batch']))
        with torch.no_grad():
            ids = generation_prompt(c)
            rng = np.random.default_rng(c['seed']+1000)
            trace = []
            for _ in range(c['generate']):
                window = ids[-c['seq']:]
                last = len(window)-1
                window = window+[0]*(c['seq']-len(window))
                x = torch.tensor([window], device=device)
                scores = model(x)[0, last].cpu().numpy()
                trace.append(scores.copy())
                order = np.argsort(-scores)[:10]
                probabilities = np.exp((scores[order]-scores[order[0]])/.8)
                ids.append(int(rng.choice(order, p=probabilities/probabilities.sum())))
            if trace:
                write_archive(output/'generation.params', {'logits': np.stack(trace, axis=1)})
            if c.get('token_bytes', 1) == 4:
                np.asarray(ids, dtype='<i4').tofile(output/'generated.ids')
            text = decode_generated(c, ids)
            (output/'sample.txt').write_text(text+'\n')
            print('Torch sample:', text, flush=True)



def torch_benchmark(config_path, initial_path, output, device):
    torch = torch_setup(device)
    import torch.nn.functional as F
    c = json.loads(Path(config_path).read_text())
    initial = read_archive(initial_path)
    data = np.memmap(Path(c['data'])/'train.bin', mode='r', dtype=token_dtype(c))
    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()
    # Establish that the optimized attention computes the same model first.
    model = torch_model(c, initial, device)
    x, y = batch(data, 0, c, True)
    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).flatten().to(device)
    logits = model(x)
    eager_loss = F.cross_entropy(logits.flatten(0, 1), y)
    eager_loss.backward()
    eager_grad = torch_records(model, True)
    model.zero_grad(set_to_none=True)
    for block in model.blocks:
        block.attention = 'sdpa'
    fast_logits = model(x)
    fast_loss = F.cross_entropy(fast_logits.flatten(0, 1), y)
    fast_loss.backward()
    np.testing.assert_allclose(fast_logits.detach().cpu().numpy(), logits.detach().cpu().numpy(), atol=3e-5, rtol=1e-3)
    parity = compare_archives(torch_records(model, True), eager_grad, 'SDPA raw gradients')
    del model, logits, fast_logits, eager_loss, fast_loss, eager_grad
    results = {}
    for name, attention, fused in [('eager', 'eager', False), ('sdpa', 'sdpa', False), ('sdpa_fused_adamw', 'sdpa', device == 'cuda')]:
        model = torch_model(c, initial, device)
        for block in model.blocks:
            block.attention = attention
        opt = torch_optimizer(model, fused=fused)
        samples = []
        for step in range(450):
            sync()
            start = time.perf_counter()
            x, y = batch(data, step, c, True)
            opt.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(x).to(device))
            loss = F.cross_entropy(logits.flatten(0, 1), torch.from_numpy(y).flatten().to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.)
            opt.step()
            value = loss.item()
            sync()
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite benchmark loss')
            if step % 150 >= 50:
                samples.append((time.perf_counter()-start)*1000)
        round_medians = [float(np.median(samples[i*100:(i+1)*100])) for i in range(3)]
        median = float(np.median(round_medians))
        results[name] = dict(median_step_ms=median, median_tokens_per_second=c['batch']*c['seq']*1000/median,
                             rounds=3, warmup_steps_per_round=50, measured_steps=300, attention=attention, fused_adamw=fused,
                             round_medians_ms=round_medians,
                             samples_ms=samples)
        print('Torch benchmark', name, median, 'ms/step', flush=True)
        del opt, model, logits, loss
    write_json(output, dict(device=device, dtype='float32', tf32=False, sdpa_gradient_parity=parity, results=results))


def monitored(command, directory, name, env, mount=None):
    require_mount(mount)
    started = time.monotonic()
    peak = 0
    previous_path = directory/(name+'-runtime.json')
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    with (directory/(name+'.log')).open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=directory)
        while process.poll() is None:
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                try:
                    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                    for line in query.stdout.splitlines():
                        fields = [v.strip() for v in line.split(',')]
                        if len(fields) == 2 and fields[0] == str(process.pid):
                            peak = max(peak, int(fields[1]))
                except (OSError, subprocess.SubprocessError, ValueError):
                    pass
    result = dict(returncode=process.returncode, wall_seconds=time.monotonic()-started,
                  sampled_process_gpu_peak_mib=peak or None, sample_interval_seconds=.5)
    invocation = result.copy()
    result['wall_seconds'] += previous.get('wall_seconds', 0.)
    result['sampled_process_gpu_peak_mib'] = max(peak, previous.get('sampled_process_gpu_peak_mib') or 0) or None
    result['invocations'] = previous.get('invocations', []) + [invocation]
    write_json(directory/(name+'-runtime.json'), result)
    if process.returncode:
        raise RuntimeError(f'{name} failed: {(directory/(name+".log")).read_text()[-4000:]}')
    print(name, 'completed', result, flush=True)
    return result


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def report(directory):
    c = json.loads((directory/'config.json').read_text())
    histories = {b: rows(directory/b/'metrics.csv') for b in ('juzhen', 'torch')}
    parity = {field: compare_archives(read_archive(directory/'juzhen'/field), read_archive(directory/'torch'/field), field)
              for field in ('first.grad', 'first.params')}
    for i in (0, 1):
        key = 'val_loss' if i == 0 else 'train_loss'
        np.testing.assert_allclose(float(histories['juzhen'][i][key]), float(histories['torch'][i][key]), atol=3e-5, rtol=1e-4)
    summaries = {b: json.loads((directory/b/'summary.json').read_text()) for b in histories}
    for b, history in histories.items():
        for row in history:
            for key in ('train_loss', 'lr', 'grad_norm', 'step_ms', 'val_loss'):
                if row[key] and not math.isfinite(float(row[key])):
                    raise AssertionError(f'{b} has nonfinite {key}')
        if not math.isfinite(summaries[b]['test_loss']):
            raise AssertionError(f'{b} has nonfinite test loss')
    paired = []
    torch_val = {int(r['step']): float(r['val_loss']) for r in histories['torch'] if r['val_loss']}
    for r in histories['juzhen']:
        if r['val_loss'] and int(r['step']) in torch_val:
            paired.append(dict(step=int(r['step']), juzhen=float(r['val_loss']), torch=torch_val[int(r['step'])]))
    final_gap = abs(paired[-1]['juzhen']-paired[-1]['torch'])
    test_gap = abs(summaries['juzhen']['test_loss']-summaries['torch']['test_loss'])
    warmup = min(50, max(1, c['steps']//5))
    timing = {}
    for b, history in histories.items():
        samples = [float(r['step_ms']) for r in history if int(r['step']) > warmup]
        timing[b] = dict(excluded_initial_steps=warmup, median_step_ms=float(np.median(samples)),
                         mean_step_ms=float(np.mean(samples)), median_tokens_per_second=c['seq']*c['batch']*1000/float(np.median(samples)))
        timing[b].update(json.loads((directory/(b+'-runtime.json')).read_text()))
    frequencies = np.bincount(np.memmap(Path(c['data'])/'train.bin', mode='r', dtype=np.uint8), minlength=27)+.5
    probabilities = frequencies/frequencies.sum()
    data = np.memmap(Path(c['data'])/'test.bin', mode='r', dtype=np.uint8)
    baseline = 0.
    for i in range(c['test_batches']):
        _, y = batch(data, i, c, False, c['test_batches'])
        baseline += float(-np.log(probabilities[y]).mean())
    baseline /= c['test_batches']
    result = dict(config=c, parity=parity, initial_validation_loss=paired[0]['juzhen'], validation=paired,
                  final_validation_gap=final_gap, test_gap=test_gap, summaries=summaries,
                  unigram_test_loss=baseline, performance=timing,
                  train_loss_curve_note='Long-run optimizer trajectories can differ in float32; compare held-out losses, not final weight equality.')
    if c['steps'] >= 100 and c['dim'] >= 64:
        for b in histories:
            if float(histories[b][-1]['val_loss']) > paired[0]['juzhen']-.2:
                raise AssertionError(f'{b} failed to learn on held-out text')
        if final_gap > .08 or test_gap > .08:
            raise AssertionError('Long-run held-out loss gap exceeds 0.08 nats/character')
    if (directory/'juzhen-benchmark/benchmark.json').exists():
        result['juzhen_benchmark'] = json.loads((directory/'juzhen-benchmark/benchmark.json').read_text())
    if (directory/'torch-benchmark.json').exists():
        result['torch_optimized_benchmark'] = json.loads((directory/'torch-benchmark.json').read_text())
    import torch
    repo = Path(__file__).resolve().parents[1]
    source_files = ['CMakeLists.txt', 'examples/demo_text8_lm.cu', 'scripts/text8_lm.py',
                    'tests/testText8LM.py', 'docs/text8-lm.md', 'ml/parameter.hpp',
                    'ml/optimizer.hpp', 'ml/optimizer_cuda.cuh', 'ml/llm.hpp',
                    'ml/layer.hpp', 'ml/util.cuh', 'cpp/cukernels.cu']
    result['provenance'] = dict(base_revision=c['revision'], python=sys.version.split()[0],
        numpy=np.__version__, torch=torch.__version__, torch_cuda=torch.version.cuda,
        cpu_threads=4, tf32=False,
        source_sha256={name: sha256(repo/name) for name in source_files if (repo/name).exists()})
    write_json(directory/'results.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for b, history in histories.items():
        train = [(int(r['step']), float(r['train_loss'])) for r in history if r['train_loss']]
        window = min(50, max(1, len(train)//10))
        smoothed = np.convolve([r[1] for r in train], np.ones(window)/window, mode='valid')
        axes[0].plot([r[0] for r in train][window-1:], smoothed, label=b)
        val = [(int(r['step']), float(r['val_loss'])) for r in history if r['val_loss']]
        axes[1].plot(*zip(*val), marker='o', label=b)
    for ax, title in zip(axes, ('Training loss (moving average)', 'Held-out validation loss')):
        ax.set(xlabel='Optimizer steps', ylabel='Nats / character', title=title)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle(f'text8: {c["layers"]} layers, width {c["dim"]}, context {c["seq"]}, FP32')
    fig.savefig(directory/'learning-curves.png', dpi=160)
    plt.close(fig)
    lines = [f'text8 character LM — {c["layers"]} layers, width {c["dim"]}, {c["heads"]} heads, context {c["seq"]}, batch {c["batch"]}',
             f'{c["steps"]} updates; {c["steps"]*c["seq"]*c["batch"]:,} training-token presentations; matched initial weights and stateless sample offsets.',
             'Architecture: pre-LN, causal attention, ReLU FFN, final LayerNorm, tied token/output weights. No dropout. CPU/CUDA FP32.',
             'AdamW lr 6e-4 -> 6e-5, linear warmup + cosine; betas .9/.95, epsilon 1e-8, decay .1 on matrices/embeddings, clip 1.',
             'Dataset is split by contiguous byte ranges (90/5/5); windows and next-token targets stay within each split.',
             f'Validation evaluates {c["eval_batches"]*c["batch"]*c["seq"]:,} tokens per checkpoint; final test evaluates {c["test_batches"]*c["batch"]*c["seq"]:,} tokens in evenly spaced windows, not the entire split.',
             f'First-step raw-gradient parity: {parity["first.grad"]}', f'First-step parameter parity: {parity["first.params"]}',
             f'Final validation gap: {final_gap:.6g}; test gap: {test_gap:.6g} nats/character.', f'Training-unigram baseline test CE: {baseline:.6f}.']
    for b, summary in summaries.items():
        p = timing[b]
        lines += [f'{b}: parameters={summary["parameters"]:,}; test CE={summary["test_loss"]:.6f}, BPC={summary["test_bpc"]:.6f}, character perplexity={summary["test_ppl"]:.6f}; median={p["median_step_ms"]:.3f} ms/step, {p["median_tokens_per_second"]:,.0f} tokens/s; sampled process GPU peak={p["sampled_process_gpu_peak_mib"]} MiB.']
    lines += ['Throughput includes batch construction/upload, forward, backward, clipping, AdamW and a loss read; excludes validation, checkpoints and initial warmup. Both runs use explicit matmul/softmax attention, not SDPA/FlashAttention. Peak memory is nvidia-smi process sampling at 0.5 s, not an exact allocation peak.',
              'This is a bounded learning/correctness experiment, not a foundation-model quality claim. Generation is a qualitative sample with temperature .8 and top-k 10.',
              'Results, logs, checkpoints, parameter archives and plot are in this directory. Source revision: '+c['revision']]
    if 'juzhen_benchmark' in result:
        p = result['juzhen_benchmark']
        lines += [f'Matched Juzhen benchmark: {p["median_step_ms"]:.3f} ms/step, {p["median_tokens_per_second"]:,.0f} tokens/s; 3 rounds of 50 warmup + 100 measured updates.']
    if 'torch_optimized_benchmark' in result:
        lines += ['Additional PyTorch benchmark (same FP32 architecture, gradients checked):']
        for name, p in result['torch_optimized_benchmark']['results'].items():
            lines += [f'{name}: {p["median_step_ms"]:.3f} ms/step, {p["median_tokens_per_second"]:,.0f} tokens/s; 3 rounds of 50 warmup + 100 measured updates.']
    (directory/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[-7:]), flush=True)
    return result


def run(args):
    require_mount(args.storage_mount)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = prepare(args.input, out/'data', args.storage_mount)
    names = ('dim', 'layers', 'heads', 'seq', 'batch', 'steps', 'warmup', 'eval_every', 'eval_batches', 'test_batches', 'seed', 'generate')
    c = {name: getattr(args, name) for name in names}
    if args.stop_after is not None and not 1 <= args.stop_after <= c['steps']:
        raise ValueError('stop-after must be within the total training steps')
    if c['dim'] % c['heads'] or not 1 <= c['warmup'] < c['steps']:
        raise ValueError('Invalid head count or warmup')
    if min(x['tokens'] for x in meta['splits'].values()) <= c['seq']:
        raise ValueError('Data split is too short for this context')
    c.update(data=str(out/'data'), data_id=meta['source_sha256'], device=args.device,
             revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent, text=True).strip())
    if (out/'config.json').exists():
        saved = json.loads((out/'config.json').read_text())
        protocol = lambda value: {k: v for k, v in value.items() if k != 'revision'}
        if protocol(saved) != protocol(c):
            raise ValueError('Existing experiment has a different configuration; choose a new output')
        c = saved  # A commit of identical code does not change the saved training protocol.
    else:
        write_json(out/'config.json', c)
    for b in ('juzhen', 'torch'):
        (out/b).mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('T8_')}
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NVIDIA_TF32_OVERRIDE='0', NVIDIA_TF32='0', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(out))
    env.update({f'T8_{k.upper()}': str(v) for k, v in c.items() if k in names or k in ('data', 'data_id')})
    env['T8_OUTPUT'] = str(out/'juzhen')
    env['T8_EXPECT_DEVICE'] = args.device
    if args.stop_after is not None:
        env['T8_STOP_AFTER'] = str(args.stop_after)
    monitored([str(args.binary.resolve())], out, 'juzhen', env, args.storage_mount)
    command = [sys.executable, str(Path(__file__).resolve()), '_torch', '--config', str(out/'config.json'),
               '--initial', str(out/'juzhen/initial.params'), '--output', str(out/'torch'), '--device', args.device]
    if args.stop_after is not None:
        command += ['--stop-after', str(args.stop_after)]
    monitored(command, out, 'torch', env, args.storage_mount)
    if args.stop_after is None or args.stop_after == c['steps']:
        return report(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--storage-mount', type=Path)
    p = sub.add_parser('run')
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--storage-mount', type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    for name, default in dict(dim=256, layers=4, heads=4, seq=128, batch=32, steps=1000, warmup=100,
                              eval_every=100, eval_batches=32, test_batches=128, seed=42, generate=128).items():
        p.add_argument('--'+name.replace('_', '-'), type=int, default=default)
    p.add_argument('--stop-after', type=int)
    p = sub.add_parser('_torch')
    for name in ('config', 'initial', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--stop-after', type=int)
    p = sub.add_parser('benchmark')
    for name in ('config', 'initial', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p = sub.add_parser('report')
    p.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.input, args.output, args.storage_mount), indent=2))
    elif args.command == 'run':
        run(args)
    elif args.command == '_torch':
        torch_train(args.config, args.initial, args.output, args.device, args.stop_after)
    elif args.command == 'benchmark':
        torch_benchmark(args.config, args.initial, args.output, args.device)
    else:
        report(args.output.resolve())

if __name__ == '__main__':
    main()
