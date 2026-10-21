#!/usr/bin/env python3
"""Independent Torch verification and plots for the native grokking example."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from text8_lm import compare_archives, read_archive


def storage(path):
    mount = Path('/mnt/experiments')
    if not os.path.ismount(mount):
        raise RuntimeError('External drive is not mounted')
    path = Path(path).resolve()
    path.relative_to((mount / 'Backup/Juzhen-data').resolve())
    return path


def forward(refs, ids, heads=4):
    """Independent functional MLP or PreLN block, using Torch autograd."""
    import torch
    import torch.nn.functional as F
    if 'hidden.weight' in refs:
        p = refs['head.weight'].shape[0]
        x = torch.cat([F.one_hot(ids[:, i], p) for i in (0, 1)], dim=-1)
        x = x.to(refs['hidden.weight'].dtype)
        h = F.relu(F.linear(x, refs['hidden.weight'], refs['hidden.bias'].flatten()))
        return F.linear(h, refs['head.weight'], refs['head.bias'].flatten())
    dim = refs['token.embedding'].shape[0]
    h = F.embedding(ids, refs['token.embedding'].T)
    h = h + refs['position.embedding'].T[None, :, :]

    def norm(x, which):
        return F.layer_norm(x, (dim,), refs[f'block.{which}.gamma'].flatten(),
                            refs[f'block.{which}.beta'].flatten(), 1e-5)

    def linear(x, w, b=None):
        return F.linear(x, refs['block.'+w], None if b is None else refs['block.'+b].flatten())

    x = norm(h, 'ln1')
    q, k, v = [linear(x, name).reshape(-1, 3, heads, dim//heads).transpose(1, 2)
               for name in ('Wq', 'Wk', 'Wv')]
    mask = torch.ones(3, 3, device=ids.device, dtype=torch.bool).triu(1)
    scores = (q @ k.transpose(-1, -2)) / (dim//heads)**.5
    attention = scores.masked_fill(mask, -1e9).softmax(-1) @ v
    r = h + linear(attention.transpose(1, 2).reshape(-1, 3, dim), 'Wo', 'bo')
    r = r + linear(F.relu(linear(norm(r, 'ln2'), 'W1', 'b1')), 'W2', 'b2')
    return F.linear(r[:, -1], refs['head.weight'], refs['head.bias'].flatten())


def verify(run, device):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c = json.loads((run/'config.json').read_text())
    pairs = list(csv.DictReader((run/'pairs.csv').open()))
    assert len(pairs) == c['p']**2
    unique = {(int(row['a']), int(row['b'])) for row in pairs}
    assert len(unique) == len(pairs), 'Split overlaps or duplicates'
    for row in pairs:
        assert int(row['target']) == (int(row['a'])+int(row['b'])) % c['p']
    training = [row for row in pairs if row['split'] == 'train']
    heldout = [row for row in pairs if row['split'] == 'heldout']
    assert len(training) == c['train_examples'] and len(heldout) == c['heldout_examples']

    def batch(rows):
        ids = torch.tensor([[int(r['a']), int(r['b']), c['p']] for r in rows], device=device)
        target = torch.tensor([int(r['target']) for r in rows], device=device)
        return ids, target

    refs = {name: torch.tensor(value, device=device, requires_grad=True)
            for name, value in read_archive(run/'initial.params').items()}
    optimizer = torch.optim.AdamW(list(refs.values()), lr=c['lr'], betas=(c['beta1'], c['beta2']),
                                 eps=c['epsilon'], weight_decay=c['weight_decay'], foreach=False)
    ids, target = batch(training)
    loss = F.cross_entropy(forward(refs, ids), target)
    loss.backward()
    expected = {name: p.grad.detach().cpu().numpy() for name, p in refs.items()}
    result = {'split_passed': True, 'initial_loss': loss.item(), 'device': device}
    result['gradient'] = compare_archives(read_archive(run/'first.grad'), expected, 'gradient')
    optimizer.step()
    expected = {name: p.detach().cpu().numpy() for name, p in refs.items()}
    result['update'] = compare_archives(read_archive(run/'first.params'), expected, 'AdamW update')
    metrics = list(csv.DictReader((run/'metrics.csv').open()))
    np.testing.assert_allclose(loss.item(), float(metrics[0]['train_loss']), atol=3e-5, rtol=1e-5)

    refs = {name: torch.tensor(value, device=device)
            for name, value in read_archive(run/'final.params').items()}
    with torch.no_grad():
        for label, rows in [('train', training), ('heldout', heldout)]:
            total, correct = 0., 0
            for start in range(0, len(rows), 512):
                ids, target = batch(rows[start:start+512])
                logits = forward(refs, ids)
                total += F.cross_entropy(logits, target, reduction='sum').item()
                correct += (logits.argmax(-1) == target).sum().item()
            value, accuracy = total/len(rows), correct/len(rows)
            np.testing.assert_allclose(value, float(metrics[-1][label+'_loss']), atol=5e-5, rtol=1e-4)
            np.testing.assert_allclose(accuracy, float(metrics[-1][label+'_accuracy']), atol=1e-10, rtol=0)
            result[label] = dict(loss=value, accuracy=accuracy)
    (run/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def plot(runs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    summaries = []
    for run in runs:
        rows = list(csv.DictReader((run/'metrics.csv').open()))
        c = json.loads((run/'config.json').read_text())
        values = {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}
        label = f"{c.get('model', 'transformer')} / decay {c['weight_decay']:g}"
        line, = axes[0].plot(values['step'], values['train_accuracy']*100, '--', label=label+' / train')
        color = line.get_color()
        axes[0].plot(values['step'], values['heldout_accuracy']*100, color=color, label=label+' / held out')
        axes[1].semilogy(values['step'], values['train_loss'], '--', color=color, label=label+' / train')
        axes[1].semilogy(values['step'], values['heldout_loss'], color=color, label=label+' / held out')
        memorized = next((r for r in rows if float(r['train_accuracy']) >= .99), None)
        generalized = next((r for r in rows if float(r['heldout_accuracy']) >= .95), None)
        summary = dict(run=str(run), config=c, final=rows[-1],
                       first_train_99_step=None if memorized is None else int(memorized['step']),
                       first_heldout_95_step=None if generalized is None else int(generalized['step']),
                       heldout_accuracy_at_memorization=None if memorized is None else float(memorized['heldout_accuracy']))
        # Operational criterion for this experiment, not a universal definition.
        summary['delayed_generalization'] = bool(memorized and generalized and
            float(memorized['heldout_accuracy']) < .2 and int(generalized['step']) > int(memorized['step']))
        summaries.append(summary)
    axes[0].set(ylabel='Accuracy (%)', ylim=(-2, 102))
    axes[1].set(ylabel='Cross entropy')
    for ax in axes:
        ax.set_xlabel('Optimizer steps')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Juzhen: modular addition')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    output.with_suffix('.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    check = sub.add_parser('verify')
    check.add_argument('run', type=storage)
    check.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    chart = sub.add_parser('plot')
    chart.add_argument('runs', type=storage, nargs='+')
    chart.add_argument('--output', type=storage, required=True)
    args = parser.parse_args()
    if args.action == 'verify':
        verify(args.run, args.device)
    else:
        plot(args.runs, args.output)
