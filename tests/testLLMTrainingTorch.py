"""Independent PyTorch oracle for explicit CPU/CUDA LLM training.

The caller must provide an output directory. The binary emits reference
fixtures and exercises checkpoint recovery there; this script checks all
parameter gradients, clipping, AdamW moments and three updates against Torch.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError as error:
    print(f"SKIP: {error}")
    sys.exit(77)
from demo_transformer import TransformerBlock

def matrix(record):
    rows, cols, values = record
    return np.array(values, dtype=np.float64).reshape((cols, rows)).T.copy()

def compare(actual, expected, name, atol=3e-5, rtol=5e-4):
    a = np.asarray(actual, dtype=np.float64)
    if isinstance(expected, torch.Tensor):
        b = expected.detach().numpy()
    else:
        b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"{name}: {a.shape} != {b.shape}")
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=name)
    return float(np.max(np.abs(a-b))) if a.size else 0.0

def check(path):
    records = json.loads(path.read_text())
    get = lambda name: matrix(records[name])
    worst = 0.0
    table = torch.tensor(get('embedding.table'), requires_grad=True)
    ids = torch.tensor([2, 2, 6, 0])
    out = F.embedding(ids, table.T).T
    worst = max(worst, compare(get('embedding.output'), out, 'embedding lookup', 0, 0))
    upstream = torch.tensor(get('embedding.upstream'))
    (out*upstream).sum().mul(2).backward()
    worst = max(worst, compare(get('embedding.grad_twice'), table.grad, 'embedding repeated IDs'))
    logits = torch.tensor(get('ce.logits').T, requires_grad=True)
    targets = torch.tensor([1, -100, 6, 0])
    loss = F.cross_entropy(logits, targets, ignore_index=-100)
    loss.backward()
    worst = max(worst, compare(get('ce.loss'), loss.reshape(1, 1), 'large-logit CE', 2e-5))
    worst = max(worst, compare(get('ce.grad'), logits.grad.T, 'large-logit CE gradient', 2e-5))

    torch.manual_seed(919)
    block = TransformerBlock(8, 8, 16, 4, 2, causal=True).double()
    mapping = {
        'layer0.Wq': block.Wq.weight, 'layer0.Wk': block.Wk.weight,
        'layer0.Wv': block.Wv.weight, 'layer0.Wo': block.Wo.weight,
        'layer0.bo': block.Wo.bias, 'layer0.W1': block.ffn1.weight,
        'layer0.b1': block.ffn1.bias, 'layer0.W2': block.ffn2.weight,
        'layer0.b2': block.ffn2.bias, 'layer0.ln1.gamma': block.ln1.weight,
        'layer0.ln1.beta': block.ln1.bias, 'layer0.ln2.gamma': block.ln2.weight,
        'layer0.ln2.beta': block.ln2.bias,
    }
    token = torch.nn.Parameter(torch.tensor(get('initial.layer1.embedding')))
    position = torch.nn.Parameter(torch.tensor(get('initial.position')))
    mapping['layer1.embedding'] = token
    mapping['position'] = position
    with torch.no_grad():
        for name, parameter in mapping.items():
            parameter.copy_(torch.tensor(get('initial.'+name)).reshape(parameter.shape))
    lr, beta1, beta2, eps, decay = get('options').ravel()
    groups = [dict(params=[p], weight_decay=float(decay) if get('decay.'+n)[0, 0] else 0.)
              for n, p in mapping.items()]
    optimizer = torch.optim.AdamW(groups, lr=float(lr), betas=(float(beta1), float(beta2)), eps=float(eps))
    inputs = torch.tensor([1, 1, 3, 7, 4, 5, 1, 2])
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    targets = torch.tensor([2, -100, 4, 1, 8, 7, -100, 6])
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        x = F.embedding(inputs, token.T) + F.embedding(positions, position.T)
        x.retain_grad()
        logits = block(x) @ token
        logits.retain_grad()
        loss = F.cross_entropy(logits, targets, ignore_index=-100)
        loss.backward()
        prefix = f'step{step}.'
        worst = max(worst, compare(get(prefix+'logits'), logits.T, prefix+'logits'))
        worst = max(worst, compare(get(prefix+'loss'), loss.reshape(1, 1), prefix+'loss'))
        worst = max(worst, compare(get(prefix+'dlogits'), logits.grad.T, prefix+'dlogits'))
        worst = max(worst, compare(get(prefix+'input_dx'), x.grad.T, prefix+'input dx'))
        for name, parameter in mapping.items():
            got = get(prefix+'raw_grad.'+name)
            worst = max(worst, compare(got, parameter.grad.reshape(got.shape), prefix+name+' raw gradient'))
        norm = torch.nn.utils.clip_grad_norm_(list(mapping.values()), .35)
        worst = max(worst, compare(get(prefix+'grad_norm'), norm.reshape(1, 1), prefix+'global norm'))
        for name, parameter in mapping.items():
            got = get(prefix+'grad.'+name)
            worst = max(worst, compare(got, parameter.grad.reshape(got.shape), prefix+name+' clipped gradient'))
        optimizer.step()
        for name, parameter in mapping.items():
            state = optimizer.state[parameter]
            for field, reference in [('parameter', parameter), ('m', state['exp_avg']), ('v', state['exp_avg_sq'])]:
                got = get(prefix+field+'.'+name)
                worst = max(worst, compare(got, reference.reshape(got.shape), prefix+name+' '+field))
    print(f"PASS {path.stem}: Torch loss/all 15 gradients/clip/3 AdamW steps; max_abs={worst:.3g}")
    return dict(backend=path.stem,passed=True,max_abs=worst)

def main(binary, output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # This list is invocation-specific; an old CUDA fixture can never make a
    # CPU-only run appear to have tested CUDA.
    for backend in ['CPU', 'CUDA']:
        (root/(backend+'.json')).unlink(missing_ok=True)
    env = dict(os.environ, JUZHEN_LLM_DUMP=str(root), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NVIDIA_TF32_OVERRIDE='0', NVIDIA_TF32='0')
    run = subprocess.run([str(Path(binary).resolve())], env=env, capture_output=True, text=True, timeout=120)
    (root/'binary.log').write_text(run.stdout+run.stderr)
    if run.returncode:
        print(run.stdout+run.stderr)
        return run.returncode
    paths = [root/'CPU.json']
    if (root/'CUDA.json').exists():
        paths.append(root/'CUDA.json')
    results = [check(path) for path in paths]
    (root/'results.json').write_text(json.dumps(results, indent=2))
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
