"""Fixed native generation suite and independent frozen-weight LM evaluation.

All generation uses the Juzhen binary. Torch only checks frozen-weight losses
and selected native logits. Prompt text, seeds and decoding rules are fixed
before examining samples. Repetition metrics are descriptive, not quality scores.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from tokenizers import Tokenizer
import bpe_lm as bpe
import text8_lm as lm

PROMPTS = [
    'the meaning of life is ',
    'the history of the world ',
    'the scientist discovered that ',
    'once upon a time there was a ',
    'one day a little girl found a ',
    'the dog was sad because ',
]
SETTINGS = [('greedy', 0., 0, [17]), ('t06', .6, 40, [17, 42]),
            ('t08', .8, 40, [17, 42]), ('t10', 1., 40, [17, 42])]


def repetition(text):
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    result = dict(words=len(words), replacement_characters=text.count('\ufffd'))
    for n in (2, 4):
        grams = [tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1))]
        result[f'repeated_{n}gram_fraction'] = 1-len(set(grams))/len(grams) if grams else 0.
        result[f'distinct_{n}'] = len(set(grams))/len(grams) if grams else 0.
        result[f'max_{n}gram_count'] = max(Counter(grams).values(), default=0)
    return result


def validate_frozen(run, config, output):
    import torch
    import torch.nn.functional as F
    lm.require_mount(output['mount'])
    torch = lm.torch_setup('cuda')
    c = dict(config)
    datasets = {name: np.memmap(Path(c['data'])/(name+'.bin'), mode='r', dtype='<i4') for name in ('valid', 'test')}
    native_summary = json.loads((run/'juzhen/summary.json').read_text())
    best = run/'juzhen/best.params'
    result = {}
    for name, archive in [('last', run/'juzhen/final.params'), ('selected', best if best.exists() else run/'juzhen/final.params')]:
        model = lm.torch_model(c, lm.read_archive(archive), 'cuda')
        model.eval()
        for block in model.blocks:
            block.attention = 'sdpa'
        values = {}
        with torch.no_grad():
            for split, batches in [('valid', c['eval_batches']), ('test', c['test_batches'])]:
                total = 0.
                for i in range(batches):
                    x, y = lm.batch(datasets[split], i, c, False, batches)
                    logits = model(torch.from_numpy(x).cuda())
                    total += F.cross_entropy(logits.flatten(0, 1), torch.from_numpy(y).cuda().flatten()).item()
                values[split+'_loss'] = total/batches
        values['test_ppl'] = math.exp(values['test_loss'])
        values['archive_sha256'] = lm.sha256(archive)
        result[name] = values
        del model
    gap = abs(result['last']['test_loss']-native_summary['test_loss'])
    if gap > 3e-5:
        raise AssertionError(f'Final native/Torch evaluation gap {gap}')
    result['native_last_test_loss_gap'] = gap
    meta = run/'juzhen/best-validation.txt'
    if meta.exists():
        step, val = meta.read_text().split()
        rows = [r for r in lm.rows(run/'juzhen/metrics.csv') if r['val_loss'] and int(r['step']) > 0]
        expected = min(rows, key=lambda r: float(r['val_loss']))
        assert int(step) == int(expected['step']) and abs(float(val)-float(expected['val_loss'])) < 1e-8
        assert abs(result['selected']['valid_loss']-float(val)) < 3e-5
        result['selection'] = dict(step=int(step), native_validation_loss=float(val), criterion='minimum evaluated validation CE after an update; test unused for selection')
    else:
        result['selection'] = dict(step=c['steps'], criterion='previous completed baseline final checkpoint')
    return result


def check_logits(config, archive, sample):
    import torch
    c = dict(config, batch=1)
    model = lm.torch_model(c, lm.read_archive(archive), 'cuda')
    model.eval()
    for block in model.blocks:
        block.attention = 'sdpa'
    ids = np.fromfile(sample/'generated.ids', dtype='<i4')
    trace = lm.read_archive(sample/'generation.params')['logits']
    prompt_ids = np.fromfile(sample/'prompt.ids', dtype='<i4')
    steps = sorted(set([0, 1, 31, 127, trace.shape[1]-1]))
    worst = 0.
    with torch.no_grad():
        for step in steps:
            window = ids[:len(prompt_ids)+step][-config['seq']:]
            c['seq'] = len(window)
            model.positions = torch.arange(len(window), device='cuda')
            logits = model(torch.from_numpy(window[None].astype(np.int64)).cuda())[0, -1].cpu().numpy()
            np.testing.assert_allclose(trace[:, step], logits, atol=3e-5, rtol=1e-3)
            worst = max(worst, float(np.max(np.abs(trace[:, step]-logits))))
    return dict(passed=True, checked_generation_steps=steps, max_abs=worst)


def evaluate(run, binary, out, mount, tokens=192, trim_prompt_end=False):
    lm.require_mount(mount)
    out.mkdir(parents=True, exist_ok=True)
    c = json.loads((run/'config.json').read_text())
    meta = bpe.load_metadata(c['data'])
    assert meta['data_id'] == c['data_id']
    native = run/'juzhen'
    checkpoint = native/('best.ckpt' if (native/'best.ckpt').exists() else 'last.ckpt')
    archive = native/('best.params' if (native/'best.params').exists() else 'final.params')
    protected = {p.name: lm.sha256(p) for p in native.iterdir()
                 if p.name in ('last.ckpt', 'best.ckpt', 'best.params', 'final.params', 'metrics.csv', 'run-spec.txt')}
    tokenizer = Tokenizer.from_file(c['tokenizer'])
    prompts = [p.rstrip() for p in PROMPTS] if trim_prompt_end else list(PROMPTS)
    protocol = dict(prompts=prompts, trim_prompt_end=trim_prompt_end, posthoc_boundary_check=trim_prompt_end, prompt_note='Six shared lowercase prefixes: three general/encyclopedic and three story prefixes. No prompts selected from results.',
                    settings=SETTINGS, new_tokens=tokens, checkpoint=str(checkpoint), checkpoint_sha256=lm.sha256(checkpoint),
                    note='Fixed token budget without EOS stopping. Metrics exclude prompt. EOS count and first-story metrics are also reported; cross-tokenizer token budgets do not imply equal words.',
                    metric_definition='repeated ngram fraction = 1 - unique word ngrams / total word ngrams; case folded ASCII English words. Token 4grams are additionally reported to expose loops without whitespace; token metrics across vocabularies are not equivalent.',
                    run=str(run), binary_sha256=lm.sha256(binary))
    lm.write_json(out/'protocol.json', protocol)
    rows = []
    selected_sample = None
    for prompt_index, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        for label, temperature, top_k, seeds in SETTINGS:
            for seed in seeds:
                lm.require_mount(mount)
                sample = out/f'p{prompt_index:02d}-{label}-seed{seed}'
                sample.mkdir(exist_ok=True)
                np.asarray(prompt_ids, dtype='<i4').tofile(sample/'prompt.ids')
                dynamic = dict(c, generate=tokens)
                env = bpe.native_environment(dynamic, native, c['device'], prompt_ids=sample/'prompt.ids')
                env.update(T8_GENERATE_ONLY='1', T8_CHECKPOINT=str(checkpoint), T8_GENERATION_OUTPUT=str(sample),
                           T8_TEMPERATURE=str(temperature), T8_TOP_K=str(top_k), T8_SAMPLE_SEED=str(seed))
                with (sample/'native.log').open('w') as log:
                    subprocess.run([str(binary)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                ids = np.fromfile(sample/'generated.ids', dtype='<i4').tolist()
                assert ids[:len(prompt_ids)] == prompt_ids and len(ids) == len(prompt_ids)+tokens
                text = tokenizer.decode(ids, skip_special_tokens=False)
                assert text.startswith(prompt)
                completion = text[len(prompt):]
                body = completion.replace('<|endoftext|>', '\n')
                token_grams = [tuple(ids[i:i+4]) for i in range(len(prompt_ids), len(ids)-3)]
                token_repeat = 1-len(set(token_grams))/len(token_grams) if token_grams else 0.
                record = dict(prompt_index=prompt_index, prompt=prompt, setting=label, temperature=temperature,
                              top_k=top_k, seed=seed, generated_tokens=tokens, eos_count=completion.count('<|endoftext|>'),
                              completion=completion, metrics=repetition(body),
                              first_story_metrics=repetition(completion.split('<|endoftext|>')[0]))
                record['metrics']['repeated_token_4gram_fraction'] = token_repeat
                lm.write_json(sample/'sample.json', record)
                (sample/'sample.txt').write_text(text, encoding='utf-8')
                rows.append(record)
                if selected_sample is None:
                    selected_sample = sample
                print('sample', run.name, prompt_index, label, seed, flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row['setting']].append(row)
    aggregate = {}
    for name, group in groups.items():
        aggregate[name] = dict(samples=len(group),
                              mean={key: float(np.mean([r['metrics'][key] for r in group])) for key in group[0]['metrics']},
                              mean_eos_count=float(np.mean([r['eos_count'] for r in group])),
                              first_story_mean={key: float(np.mean([r['first_story_metrics'][key] for r in group])) for key in group[0]['first_story_metrics']})
    frozen = validate_frozen(run, c, dict(mount=mount))
    logits = check_logits(c, archive, selected_sample)
    assert protected == {name: lm.sha256(native/name) for name in protected}, 'Evaluation changed training files'
    result = dict(protocol=protocol, aggregate=aggregate, frozen_weight_evaluation=frozen,
                  native_logits_reference=logits, preserves_training_files=True, samples=rows)
    lm.write_json(out/'results.json', result)
    lines = ['# Fixed generation suite', '', f'Run: {run}', '',
             'All samples below were specified before generation. Repetition is descriptive and does not measure factuality or story consistency.', '']
    for row in rows:
        lines += [f"## Prompt {row['prompt_index']}, {row['setting']}, seed {row['seed']}", '',
                  f"Prompt: {row['prompt']!r}", '', row['prompt']+row['completion'], '',
                  f"Repeated word 4-gram fraction: {row['metrics']['repeated_4gram_fraction']:.4f}", '']
    (out/'samples.md').write_text('\n'.join(lines), encoding='utf-8')
    print('evaluation complete', run.name, aggregate, frozen, logits, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--storage-mount', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=192)
    parser.add_argument('--trim-prompt-end', action='store_true', help='Boundary ablation: remove trailing whitespace from every fixed prompt')
    args = parser.parse_args()
    if not 128 <= args.tokens <= 4096:
        parser.error('Use at least 128 and at most 4096 new tokens')
    evaluate(args.run.resolve(), args.binary.resolve(), args.output.resolve(), args.storage_mount, args.tokens, args.trim_prompt_end)
