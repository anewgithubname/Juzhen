#!/usr/bin/env python3
"""Run the explicit LM with the historical TinyStories model/training settings."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import text8_lm as lm
from tinystories_legacy_prepare import checked_root


def environment(c, out, *, stop=None, tf32=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith('T8_')}
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NVIDIA_TF32_OVERRIDE='1' if tf32 else '0', NVIDIA_TF32='1' if tf32 else '0',
               PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(out))
    for key in ('dim','layers','heads','seq','batch','steps','warmup','eval_every',
                'eval_batches','test_batches','seed','generate','vocab','token_bytes','data','data_id'):
        env['T8_'+key.upper()] = str(c[key])
    env.update(T8_LEGACY_TINYSTORIES='1', T8_OUTPUT=str(out), T8_EXPECT_DEVICE='cuda')
    if stop is not None:
        env['T8_STOP_AFTER'] = str(stop)
    return env


def invoke(binary, c, out, stop=None, tf32=False):
    checked_root(out)
    out.mkdir(exist_ok=True, parents=True)
    with (out/'native.log').open('a') as log:
        subprocess.run([str(binary)], env=environment(c,out,stop=stop,tf32=tf32),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def validate(binary, root):
    import torch
    import torch.nn.functional as F
    lm.torch_setup('cuda')
    folder = root/'verification'/('run-'+str(time.time_ns()))
    folder.mkdir(parents=True)
    data = folder/'data'
    data.mkdir(exist_ok=True)
    rng = np.random.default_rng(19)
    for split in ('train', 'valid'):
        rng.integers(0,512,4096,dtype=np.int32).astype('<i4').tofile(data/(split+'.bin'))
    c = dict(dim=32,layers=2,heads=4,seq=16,batch=3,steps=13,warmup=2,
             eval_every=4,eval_batches=1,test_batches=1,seed=42,generate=0,
             vocab=512,token_bytes=4,data=str(data),data_id='legacy-numerical-fixture-v1',legacy_tinystories=True)
    full, resumed = folder/'uninterrupted',folder/'resumed'
    invoke(binary,c,full)
    invoke(binary,c,resumed,stop=5)
    invoke(binary,c,resumed)
    restart = lm.compare_archives(lm.read_archive(resumed/'final.params'),lm.read_archive(full/'final.params'),
                                  'legacy separate-process restart',atol=2e-6,rtol=1e-5)
    initial = lm.read_archive(full/'initial.params')
    model = lm.torch_model(c,initial,'cuda')
    # Check the old two-block one-hot input projection against the lookup graph.
    x = np.fromfile(full/'first-input.ids',dtype='<i4').reshape(c['batch'],c['seq'])
    y = np.fromfile(full/'first-target.ids',dtype='<i4').reshape(c['batch'],c['seq'])
    ids = torch.tensor(x.astype(np.int64),device='cuda')
    onehot = torch.cat((F.one_hot(ids,c['vocab']).float(),
                       F.one_hot(torch.arange(c['seq'],device='cuda'),c['seq']).float().expand(c['batch'],-1,-1)),dim=-1)
    dense = onehot @ torch.cat((model.token,model.position),dim=1).T + model.input_bias
    lookup = F.embedding(ids,model.token.T)+F.embedding(model.positions,model.position.T)+model.input_bias
    torch.testing.assert_close(dense,lookup,atol=1e-9,rtol=1e-6)
    optimizer = torch.optim.Adam(model.parameters(),lr=.0005/2,betas=(.9,.999),eps=1e-8,foreach=False)
    loss = F.cross_entropy(model(ids).reshape(-1,c['vocab']),torch.tensor(y.astype(np.int64),device='cuda').flatten())
    loss.backward()
    gradients = lm.compare_archives(lm.read_archive(full/'first.grad'),lm.torch_records(model,True),
                                    'legacy FP32 first raw gradients',atol=3e-5,rtol=1e-3)
    optimizer.step()
    update = lm.compare_archives(lm.read_archive(full/'first.params'),lm.torch_records(model),
                                'legacy FP32 first Adam update',atol=3e-6,rtol=1e-3)
    rows = lm.rows(full/'metrics.csv')
    assert abs(float(rows[1]['train_loss'])-loss.item())<3e-6
    valid = np.fromfile(data/'valid.bin',dtype='<i4')
    hi = len(valid)-c['seq']-1
    offsets = [int(float(b)*hi/c['batch']) for b in range(c['batch'])]
    vx = torch.tensor(np.stack([valid[o:o+c['seq']] for o in offsets]).astype(np.int64),device='cuda')
    vy = torch.tensor(np.stack([valid[o+1:o+c['seq']+1] for o in offsets]).astype(np.int64),device='cuda')
    initial_model = lm.torch_model(c,initial,'cuda')
    with torch.no_grad():
        vloss = F.cross_entropy(initial_model(vx).reshape(-1,c['vocab']),vy.flatten()).item()
    assert abs(vloss-float(rows[0]['val_loss']))<3e-6
    result = dict(passed=True,onehot_lookup_equivalence=True,validation_window_equivalence=True,
                  first_gradient=gradients,first_adam_update=update,restart=restart)
    lm.write_json(folder/'results.json',result)
    lm.write_json(root/'verification/results.json',dict(result,artifacts=str(folder)))
    print(json.dumps(result,indent=2),flush=True)
    return result


def run(binary, root):
    root=checked_root(root)
    if (root/'experiment/completion.json').exists() and (root/'evaluation/results.json').exists():
        print('Training and evaluation already complete',flush=True)
        return
    if not (root/'verification/results.json').exists() or not json.loads((root/'verification/results.json').read_text())['passed']:
        raise ValueError('Run verification before full training')
    meta=json.loads((root/'data/metadata.json').read_text())
    for name,record in meta['splits'].items():
        assert lm.sha256(root/'data'/(name+'.bin'))==record['sha256']
    assert lm.sha256(root/'data/tokenizer.json')==meta['tokenizer_sha256']
    c=dict(dim=512,layers=8,heads=8,seq=256,batch=64,steps=60000,warmup=1000,
           eval_every=500,eval_batches=1,test_batches=1,seed=42,generate=0,
           vocab=4096,token_bytes=4,data=str(root/'data'),data_id=meta['data_id'],
           legacy_tinystories=True,tf32=True,early_stopping_patience=12,
           optimizer=dict(name='Adam',beta1=.9,beta2=.999,eps=1e-8,weight_decay=0,clip=None),
           lr_peak=.0005,lr_min=.00005,tokenizer=str(root/'data/tokenizer.json'),eot_id=meta['eot_id'],
           historical_limitations='Original weights/tokenizer unavailable. Same tokenizer recipe, fresh seed42 weights. Historical final step count unknown; use documented 60k budget.',
           random_windows='C++ mt19937 seeded42 after initialization; exact stream saved for resume',
           evaluation='one fixed legacy validation batch of 64x256, no independent test',
           source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    dest=root/'experiment'
    dest.mkdir(exist_ok=True)
    if (dest/'config.json').exists():
        saved = json.loads((dest/'config.json').read_text())
        # The original revision records provenance, not the training protocol.
        protocol = lambda value: {k: v for k, v in value.items() if k != 'source_commit'}
        if protocol(saved) != protocol(c):
            raise ValueError('Resume configuration/data differs from saved protocol')
        c = saved
    else:
        lm.write_json(dest/'config.json',c)
    pinned=root/'bin/demo_tinystories_explicit_cuda'
    pinned.parent.mkdir(exist_ok=True)
    if not pinned.exists():
        shutil.copy2(binary,pinned)
    lm.write_json(root/'binary.json',dict(path=str(pinned),sha256=lm.sha256(pinned)))
    print('Starting full legacy-settings run',json.dumps(c,indent=2),flush=True)
    started=time.time()
    invoke(pinned,c,dest/'juzhen',tf32=True)
    lm.write_json(dest/'completion.json',dict(completed=True,elapsed_seconds=time.time()-started,
                                           summary=json.loads((dest/'juzhen/summary.json').read_text())))
    print((dest/'completion.json').read_text(),flush=True)
    from tinystories_legacy_evaluate import evaluate
    evaluate(pinned,root)
    lm.write_json(root/'status.json',dict(state='complete',summary=json.loads((dest/'juzhen/summary.json').read_text()),report=str(root/'evaluation/report.md')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('verify','run'))
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=checked_root(a.root)
    (validate if a.mode=='verify' else run)(a.binary.resolve(),root)
