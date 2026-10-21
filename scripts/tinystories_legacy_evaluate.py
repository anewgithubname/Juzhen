#!/usr/bin/env python3
"""Evaluate the legacy-settings run and save the original 40 story prompts.

No paid judge or remote generation is used. All outputs stay on the external
storage root. The historical 1.32 loss is a reference, not an exact replay.
"""
from pathlib import Path
import argparse
import json
import math
import subprocess
import time
import numpy as np
from tokenizers import Tokenizer
import text8_lm as lm
from tinystories_legacy_run import environment
from tinystories_legacy_prepare import checked_root
from tinystories_gpteval import PROMPTS
from bpe_generation_eval import repetition


def native_eval(binary,c,training,out,batches,tf32):
    env=environment(dict(c,eval_batches=batches),training,tf32=tf32)
    destination=out/('validation-'+str(batches)+('-tf32' if tf32 else '-fp32')+'.json')
    env.update(T8_EVALUATE_ONLY='1',T8_CHECKPOINT=str(training/'best.ckpt'),T8_EVALUATION_OUTPUT=str(destination))
    with destination.with_suffix('.log').open('w') as log:
        subprocess.run([str(binary)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads(destination.read_text())


def reference(c,archive):
    torch=lm.torch_setup('cuda')
    import torch.nn.functional as F
    model=lm.torch_model(c,lm.read_archive(archive),'cuda')
    model.eval()
    for block in model.blocks:
        block.attention='sdpa'
    data=np.memmap(Path(c['data'])/'valid.bin',dtype='<i4',mode='r')
    hi=len(data)-c['seq']-1
    offsets=[int(float(b)*hi/c['batch']) for b in range(c['batch'])]
    x=np.stack([data[o:o+c['seq']] for o in offsets]).astype(np.int64)
    y=np.stack([data[o+1:o+c['seq']+1] for o in offsets]).astype(np.int64)
    with torch.no_grad():
        logits=model(torch.from_numpy(x).cuda())
        targets=torch.from_numpy(y).cuda()
        loss=F.cross_entropy(logits.flatten(0,1),targets.flatten()).item()
        accuracy=(logits.argmax(-1)==targets).float().mean().item()
    del model,logits
    torch.cuda.empty_cache()
    return dict(validation_loss=loss,validation_ppl=math.exp(loss),validation_top1_accuracy=accuracy,
                precision='FP32, TF32 off; independent PyTorch frozen weights')


def evaluate(binary,root):
    root=checked_root(root)
    run=root/'experiment'
    c=json.loads((run/'config.json').read_text())
    training=run/'juzhen'
    summary=json.loads((training/'summary.json').read_text())
    out=root/'evaluation';out.mkdir(exist_ok=True)
    protected={name:lm.sha256(training/name) for name in ('last.ckpt','best.ckpt','best.params','final.params','metrics.csv','run-spec.txt')}
    tf32=native_eval(binary,c,training,out,1,True)
    fp32=native_eval(binary,c,training,out,1,False)
    expanded=native_eval(binary,c,training,out,32,False)
    ref=reference(c,training/'best.params')
    assert abs(tf32['validation_loss']-summary['best_val_loss'])<1e-6
    gap=abs(fp32['validation_loss']-ref['validation_loss'])
    assert gap<1e-4,('Frozen native/PyTorch validation mismatch',gap)
    tok=Tokenizer.from_file(c['tokenizer'])
    eot=tok.token_to_id('<|endoftext|>')
    protocol=dict(prompts=PROMPTS,temperature=.8,top_k=40,new_tokens=200,sample_seed=42,
                  checkpoint=str(training/'best.ckpt'),checkpoint_sha256=protected['best.ckpt'],
                  prompt_source='original scripts/tinystories_gpteval.py, all 40 openings in original order',
                  padding='legacy left-pad with EOT; final-position logits',precision='FP32 storage, TF32 GEMMs',
                  eos_policy='save full 200-token traces; render completion only up to first EOT, equivalent to stopping at EOT',
                  judge='none; no claims of GPT-eval score improvement',no_independent_test=True)
    lm.write_json(out/'generation-protocol.json',protocol)
    samples=[]
    for i,prompt in enumerate(PROMPTS):
        checked_root(root)
        sample=out/('story-%02d'%i);sample.mkdir(exist_ok=True)
        prompt_ids=tok.encode(prompt).ids
        np.asarray(prompt_ids,dtype='<i4').tofile(sample/'prompt.ids')
        env=environment(dict(c,generate=200),training,tf32=True)
        env.update(T8_GENERATE_ONLY='1',T8_CHECKPOINT=str(training/'best.ckpt'),T8_GENERATION_OUTPUT=str(sample),
                   T8_TEMPERATURE='.8',T8_TOP_K='40',T8_SAMPLE_SEED='42',T8_EOT_ID=str(eot),T8_PROMPT_IDS=str(sample/'prompt.ids'))
        with (sample/'native.log').open('w') as log:
            subprocess.run([str(binary)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        ids=np.fromfile(sample/'generated.ids',dtype='<i4').tolist()
        assert ids[:len(prompt_ids)]==prompt_ids and len(ids)==len(prompt_ids)+200
        new=ids[len(prompt_ids):]
        eos_position=new.index(eot) if eot in new else None
        completion_ids=new[:eos_position] if eos_position is not None else new
        text=tok.decode(prompt_ids+completion_ids,skip_special_tokens=False)
        if not text.startswith(prompt):
            raise ValueError('Legacy Metaspace prompt decode differs: '+repr(prompt))
        completion=text[len(prompt):]
        record=dict(prompt=prompt,completion=completion,eos_generated=eos_position is not None,
                    tokens_before_eos=len(completion_ids),metrics=repetition(completion))
        lm.write_json(sample/'sample.json',record)
        (sample/'sample.txt').write_text(text+'\n',encoding='utf-8')
        samples.append(record)
        print('generated legacy prompt',i+1,'/40',flush=True)
    lm.write_json(out/'completions.json',[dict(prompt=r['prompt'],completion=r['completion']) for r in samples])
    rows=lm.rows(training/'metrics.csv')
    times=[float(r['step_ms']) for r in rows if r['step_ms'] and int(r['step'])>c['warmup']]
    result=dict(summary=summary,native_tf32=tf32,native_fp32=fp32,expanded_validation_fp32=expanded,
                torch_reference=ref,reference_loss_gap=gap,median_training_step_ms=float(np.median(times)),
                generation_protocol=protocol,stories=samples,
                historical=dict(validation_loss=1.32,validation_ppl=math.exp(1.32),step_ms=141,
                                caveat='Original tokenizer, weights and exact final step unavailable. Recipe-level comparison; validation loss is not an exact identical-tokenizer comparison.'),
                metric_note='One fixed 64x256-token legacy validation batch selects the model. Expanded validation uses 32 fixed batches; no independent test set. Perplexities across tokenizers are not directly comparable.')
    assert protected=={name:lm.sha256(training/name) for name in protected}
    result['training_files_preserved']=True
    lm.write_json(out/'results.json',result)
    lines=['# TinyStories: new explicit training, historical settings','',
           f"Trained {summary['steps']:,}/{c['steps']:,} updates; best update {summary['best_step']:,}.",
           f"Parameters: {summary['parameters']:,}. Best legacy validation CE {tf32['validation_loss']:.6f}, perplexity {tf32['validation_ppl']:.6f}.",
           f"Historical reported CE: 1.32. Original tokenizer/weights absent; this is a rebuilt tokenizer and fresh initialization, not an exact replay.",
           f"Expanded validation (FP32, 32 batches): CE {expanded['validation_loss']:.6f}, perplexity {expanded['validation_ppl']:.6f}.",
           f"Independent frozen-weight FP32 loss gap: {gap:.3g}.",
           f"Training median: {result['median_training_step_ms']:.3f} ms/update; historical TF32 reference: 141 ms/update.",
           'Same eight-layer 512-dimensional ReLU PreLN graph as the old demo, independent projection/input bias, no final LayerNorm. Adam (.9,.999), no weight decay or clipping. Full GPT4 TinyStories, Metaspace BPE-4096. 60k cosine budget, warmup1k, batch64, context256, TF32.',
           'Selection uses the original fixed validation batch. No independent test split or paid judge score was created.',
           '', '## Original 40 prompts', '']
    for i,r in enumerate(samples):
        lines += [f'### {i+1}', '', r['prompt']+r['completion'], '']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    # Scientific figure is a standalone artifact.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    observed=[r for r in rows if r['val_loss']]
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot([int(r['step']) for r in observed],[float(r['val_loss']) for r in observed],label='New implementation / rebuilt legacy tokenizer')
    ax.axhline(1.32,linestyle='--',color='gray',label='Historical reported 1.32 (tokenizer unavailable)')
    ax.set(xlabel='Optimizer updates',ylabel='Validation cross entropy / token',title='TinyStories, historical model and training settings')
    ax.set_ylim(bottom=0,top=min(4,max(2,tf32['validation_loss']*2)))
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'validation.png',dpi=180);fig.savefig(out/'validation.pdf');plt.close(fig)
    print('evaluation complete',summary,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    a=p.parse_args();evaluate(a.binary.resolve(),a.root)
