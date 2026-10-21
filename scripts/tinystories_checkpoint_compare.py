#!/usr/bin/env python3
"""Compare recovered TinyStories weights and the completed explicit run.

Evaluates each original fixed validation batch, then scores the same raw story
continuations with each model's own tokenizer. Paired bits/byte accounts for
vocabulary differences. No training, remote judge or source-file changes.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import os
import re
import struct
import numpy as np
from tokenizers import Tokenizer
import text8_lm as lm
from tinystories_legacy_prepare import checked_root, stories
from bpe_generation_eval import repetition


def read_legacy(path):
    with path.open('rb') as f:
        header=struct.unpack('<8i2f',f.read(40))
        magic,vocab,seq,dim,dk,ff,heads,layers,loss,accuracy=header
        if magic!=0x4a5a5453 or dk!=dim or ff!=4*dim:
            raise ValueError('Unsupported old autoregressive model')
        def matrix(shape):
            raw=f.read(12)
            if len(raw)!=12:
                raise ValueError('Truncated old matrix header')
            rows,cols,transpose=struct.unpack('<3i',raw)
            if rows<=0 or cols<=0 or rows*cols>100000000 or transpose not in (0,1):
                raise ValueError('Invalid old matrix dimensions')
            raw=f.read(rows*cols*4)
            if len(raw)!=rows*cols*4:
                raise ValueError('Truncated old matrix values')
            value=np.frombuffer(raw,dtype='<f4').reshape(cols,rows).T.copy()
            if transpose:
                value=value.T.copy()
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('Old matrix shape/nonfinite error')
            return value
        embed=matrix((dim,vocab+seq))
        refs={'token.embedding':embed[:,:vocab].copy(),
              'position.embedding':embed[:,vocab:].copy(),'input.bias':matrix((dim,1))}
        for i in range(layers):
            names_shapes=[('Wq',(dk,dim)),('Wk',(dk,dim)),('Wv',(dk,dim)),
                          ('Wo',(dim,dk)),('bo',(dim,1)),('W1',(ff,dim)),('b1',(ff,1)),
                          ('W2',(dim,ff)),('b2',(dim,1)),('ln1.gamma',(dim,1)),
                          ('ln1.beta',(dim,1)),('ln2.gamma',(dim,1)),('ln2.beta',(dim,1))]
            for name,shape in names_shapes:
                refs[f'blocks.{i}.{name}']=matrix(shape)
        refs['projection.weight']=matrix((vocab,dim))
        refs['projection.bias']=matrix((vocab,1))
        if f.read(1):
            raise ValueError('Unexpected trailing old weight data')
    return refs,dict(vocab=vocab,seq=seq,dim=dim,heads=heads,layers=layers,
                     legacy_tinystories=True,stored_validation_loss=loss,
                     stored_validation_accuracy_percent=accuracy)


def fixed_batch(model,c,path,dtype):
    import torch
    import torch.nn.functional as F
    data=np.memmap(path,dtype=dtype,mode='r')
    hi=len(data)-c['seq']-1
    offsets=[int(float(b)*hi/c['batch']) for b in range(c['batch'])]
    x=np.stack([data[o:o+c['seq']] for o in offsets]).astype(np.int64)
    y=np.stack([data[o+1:o+c['seq']+1] for o in offsets]).astype(np.int64)
    with torch.no_grad():
        logits=model(torch.from_numpy(x).cuda())
        target=torch.from_numpy(y).cuda()
        loss=F.cross_entropy(logits.flatten(0,1),target.flatten()).item()
        accuracy=(logits.argmax(-1)==target).float().mean().item()
    return dict(cross_entropy=loss,perplexity=math.exp(loss),top1_accuracy=accuracy,tokens=c['batch']*c['seq'])


def choose_shared(raw,tokenizers,context,count):
    selected=[]
    rejected=dict(too_long=0,too_short=0,roundtrip=0,prefix=0)
    total_stories=0
    for index,story in enumerate(stories(raw)):
        total_stories+=1
        words=list(re.finditer(r'\S+',story))
        if len(words)<12:
            rejected['too_short']+=1;continue
        boundary=words[7].end()
        prefix=story[:boundary]
        all_ids=[t.encode(story).ids for t in tokenizers]
        prefix_ids=[t.encode(prefix).ids for t in tokenizers]
        if any(len(ids)>context for ids in all_ids):
            rejected['too_long']+=1;continue
        if any(t.decode(ids,skip_special_tokens=False)!=story for t,ids in zip(tokenizers,all_ids)):
            rejected['roundtrip']+=1;continue
        if any(not p or ids[:len(p)]!=p or len(p)>=len(ids) for p,ids in zip(prefix_ids,all_ids)):
            rejected['prefix']+=1;continue
        selected.append(dict(source_story_index=index,text=story,prefix_characters=boundary,
                             continuation_bytes=len(story[boundary:].encode('utf-8')),
                             ids=all_ids,prefix_tokens=[len(p) for p in prefix_ids]))
    if len(selected)<count:
        raise ValueError('Not enough shared stories')
    indices=np.linspace(0,len(selected)-1,count,dtype=np.int64)
    return [selected[i] for i in indices],dict(total_source_stories=total_stories,eligible=len(selected),rejected=rejected,
                  selected=count,selection='evenly spaced within all eligible source stories, source order preserved',
                  eligibility='at least 12 whitespace words, full story <=256 tokens for both tokenizers, exact UTF8 roundtrip for both, stable eight-word prefix tokenization')


def score_shared(model,c,selected,which,eot):
    import torch
    import torch.nn.functional as F
    result=[]
    with torch.no_grad():
        for start in range(0,len(selected),c['batch']):
            group=selected[start:start+c['batch']]
            x=np.full((len(group),c['seq']),eot,dtype=np.int64)
            y=np.full_like(x,-100)
            for b,row in enumerate(group):
                ids=row['ids'][which]
                prefix=row['prefix_tokens'][which]
                x[b,:len(ids)]=ids
                targets=ids[1:]+[eot]
                y[b,prefix-1:len(ids)]=targets[prefix-1:]
            logits=model(torch.from_numpy(x).cuda())
            target=torch.from_numpy(y).cuda()
            losses=F.cross_entropy(logits.transpose(1,2),target,reduction='none',ignore_index=-100)
            totals=losses.double().sum(dim=1).cpu().numpy()
            counts=(y!=-100).sum(axis=1)
            for row,nats,tokens in zip(group,totals,counts):
                result.append(dict(source_story_index=row['source_story_index'],nats=float(nats),
                                   target_tokens=int(tokens),continuation_bytes=row['continuation_bytes']))
            print('scored',which,start+len(group),'/',len(selected),flush=True)
    total_nats=sum(r['nats'] for r in result)
    total_tokens=sum(r['target_tokens'] for r in result)
    total_bytes=sum(r['continuation_bytes'] for r in result)
    return dict(bits_per_utf8_byte=total_nats/math.log(2)/total_bytes,nats=total_nats,
                continuation_bytes=total_bytes,target_tokens=total_tokens,
                cross_entropy_per_own_token=total_nats/total_tokens,stories=result)


def main(root,old_directory):
    root=checked_root(root)
    out=root/'recovered-checkpoint-comparison'
    out.mkdir(exist_ok=True)
    torch=lm.torch_setup('cuda')
    c=json.loads((root/'experiment/config.json').read_text())
    old_path=old_directory/'tinystories.weights'
    old,old_config=read_legacy(old_path)
    assert all(old_config[k]==c[k] for k in ('vocab','seq','dim','heads','layers'))
    newer=lm.read_archive(root/'experiment/juzhen/best.params')
    assert old.keys()==newer.keys()
    assert all(old[k].shape==newer[k].shape for k in old)
    lm.write_archive(out/'historical-converted.params',old)
    old_tokenizer=Tokenizer.from_file(str(old_directory/'tokenizer.json'))
    new_tokenizer=Tokenizer.from_file(c['tokenizer'])
    old_raw=old_directory/'TinyStoriesV2-GPT4-valid.txt'
    new_raw=root/'raw/TinyStoriesV2-GPT4-valid.txt'
    assert lm.sha256(old_raw)==lm.sha256(new_raw),'Validation text differs'
    selected,selection=choose_shared(old_raw,[old_tokenizer,new_tokenizer],c['seq'],1024)
    lm.write_json(out/'protocol.json',dict(source=str(old_raw),source_sha256=lm.sha256(old_raw),selection=selection,
        context=256,prefix='condition on the same first eight whitespace-delimited words, exclude their likelihood',
        scoring='teacher-forced complete continuation plus terminating EOT, causal positions start at zero, right-pad ignored',
        normalization='negative log probability / log(2) / original continuation UTF8 bytes; EOT adds probability cost but no text bytes',
        precision='same independent PyTorch graph, FP32, TF32 off, SDPA',
        limitation='validation data, not independent test; one checkpoint per training run; different training corpora and initial weights; this does not isolate framework effect'))
    lm.write_json(out/'shared-stories.json',selected)
    results={}
    for which,(name,params,tok,valid,dtype) in enumerate([
        ('historical',old,old_tokenizer,old_directory/'val.bin','<u2'),
        ('new',newer,new_tokenizer,root/'data/valid.bin','<i4')]):
        model=lm.torch_model(c,params,'cuda').eval()
        for block in model.blocks:block.attention='sdpa'
        fixed=fixed_batch(model,c,valid,dtype)
        shared=score_shared(model,c,selected,which,tok.token_to_id('<|endoftext|>'))
        results[name]=dict(original_fixed_validation=fixed,shared_text=shared)
        print(name,'fixed',fixed,'shared_bpb',shared['bits_per_utf8_byte'],flush=True)
        del model
        torch.cuda.empty_cache()
    assert abs(results['historical']['original_fixed_validation']['cross_entropy']-old_config['stored_validation_loss'])<2e-4
    prior=json.loads((root/'evaluation/results.json').read_text())
    assert abs(results['new']['original_fixed_validation']['cross_entropy']-prior['native_fp32']['validation_loss'])<1e-4
    # Paired bootstrap over the same stories. It reflects sample variation only.
    nats_old=np.array([s['nats'] for s in results['historical']['shared_text']['stories']])
    nats_new=np.array([s['nats'] for s in results['new']['shared_text']['stories']])
    byte_counts=np.array([s['continuation_bytes'] for s in selected])
    rng=np.random.default_rng(42)
    estimates=[]
    for _ in range(10000):
        index=rng.integers(0,len(selected),len(selected))
        estimates.append(float((nats_new[index]-nats_old[index]).sum()/byte_counts[index].sum()/math.log(2)))
    delta=results['new']['shared_text']['bits_per_utf8_byte']-results['historical']['shared_text']['bits_per_utf8_byte']
    relative=100*delta/results['historical']['shared_text']['bits_per_utf8_byte']
    comparison=dict(new_minus_old_bits_per_byte=delta,new_relative_percent=relative,
                    paired_bootstrap_95_percentile_interval=np.quantile(estimates,[.025,.975]).tolist(),bootstrap_samples=10000,bootstrap_seed=42,
                    uncertainty_note='interval reflects only selected validation stories; not training seed/run variance')
    historical=json.loads((old_directory/'gpteval_completions.json').read_text())
    fresh=json.loads((root/'evaluation/completions.json').read_text())
    assert len(historical)==len(fresh)==40
    assert all(a['prompt']==b['prompt'] for a,b in zip(historical,fresh))
    repetition_summary={}
    for label,items in [('historical',historical),('new',fresh)]:
        repetition_summary[label]=dict(samples=40,mean_repeated_word_4gram_fraction=float(np.mean([repetition(x['completion'])['repeated_4gram_fraction'] for x in items])),
            note='Existing samples on the same prompts; historical RNG unknown. Repetition does not measure story quality.')
    result=dict(models=results,common_text_comparison=comparison,selection=selection,stored_historical=old_config,
                hashes=dict(historical_weights=lm.sha256(old_path),new_weights=lm.sha256(root/'experiment/juzhen/best.params'),historical_tokenizer=lm.sha256(old_directory/'tokenizer.json'),new_tokenizer=lm.sha256(c['tokenizer'])),
                existing_40_prompt_repetition=repetition_summary,
                historical_exact_loss_reproduced=True,new_exact_loss_reproduced=True,
                no_paid_judge_invoked=True)
    lm.write_json(out/'results.json',result)
    lines=['Recovered TinyStories checkpoint comparison','',
           f"Historical original fixed validation CE: {results['historical']['original_fixed_validation']['cross_entropy']:.8f}; new: {results['new']['original_fixed_validation']['cross_entropy']:.8f}. Different tokenizers, so these are not directly comparable.",
           f"Common validation continuation: {len(selected)} stories, {int(byte_counts.sum()):,} original UTF8 bytes, same eight-word prefix.",
           f"Historical: {results['historical']['shared_text']['bits_per_utf8_byte']:.8f} bits/byte; new: {results['new']['shared_text']['bits_per_utf8_byte']:.8f} bits/byte.",
           f"New relative difference: {relative:+.4f}%. Paired validation-story bootstrap 95% interval for new-minus-old bits/byte: {comparison['paired_bootstrap_95_percentile_interval']}.",
           'Lower bits/byte is better. Validation subset selected only by length/roundtrip eligibility before scoring. This is not an independent test set or an estimate of training-run variance.',
           'Scoring uses each original tokenizer and the same FP32 frozen-weight graph. No paid story judge was invoked.',
           '', 'Original and new 40-prompt samples (same openings; different sampling runs):','']
    for i,(a,b) in enumerate(zip(historical,fresh)):
        lines.extend([f'Prompt {i+1}: '+a['prompt'],'Historical: '+a['completion'],'New: '+b['completion'],''])
    (out/'report.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--old-directory',type=Path,required=True)
    a=p.parse_args();main(a.root,a.old_directory)
