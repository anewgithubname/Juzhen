"""Assemble the three completed BPE experiments; never starts training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
import text8_lm as lm


def build(root, mount):
    lm.require_mount(mount)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    baseline = Path('/mnt/experiments/Backup/Juzhen-data/llm/bpe-runs/20261020-101224/experiment')
    runs = {'text8-10000': baseline, 'text8-50000': root/'text8-50000', 'tinystories-50000': root/'tinystories-50000'}
    labels = {'text8-10000': 'text8 10k', 'text8-50000': 'text8 50k', 'tinystories-50000': 'TinyStories 50k'}
    colors = {'text8-10000': '#8597a5', 'text8-50000': '#2266cc', 'tinystories-50000': '#14957b'}
    results = {}
    for name, run in runs.items():
        config = json.loads((run/'config.json').read_text())
        evaluation = json.loads((root/'evaluation'/name/'results.json').read_text())
        native = json.loads((run/'juzhen/summary.json').read_text())
        assert native['steps'] == config['steps']
        history = lm.rows(run/'juzhen/metrics.csv')
        val = [r for r in history if r['val_loss']]
        selected = evaluation['frozen_weight_evaluation']['selected']
        selection = evaluation['frozen_weight_evaluation']['selection']
        training = json.loads((run/'results.json').read_text())
        boundary = json.loads((root/'evaluation-no-trailing-space'/name/'results.json').read_text())
        results[name] = dict(run=str(run), selected_step=selection['step'],
                             validation_loss=selected['valid_loss'], test_loss=selected['test_loss'],
                             test_perplexity=selected['test_ppl'], final_native_test_perplexity=native['test_ppl'],
                             final_native_validation_loss=float(val[-1]['val_loss']),
                             parameters=native['parameters'], updates=config['steps'],
                             tokens_presented=config['steps']*config['batch']*config['seq'],
                             median_step_ms=training['median_step_ms'],
                             repetition=evaluation['aggregate'], boundary_repetition=boundary['aggregate'],
                             boundary_logits_reference=boundary['native_logits_reference'],
                             boundary_preserves_training=boundary['preserves_training_files'],
                             native_torch_test_gap=evaluation['frozen_weight_evaluation']['native_last_test_loss_gap'],
                             logits_reference=evaluation['native_logits_reference'],
                             unchanged_training_files=evaluation['preserves_training_files'])
    lm.write_json(root/'comparison.json', results)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    axes = axes_grid.ravel()
    for name in ('text8-10000', 'text8-50000'):
        rows = [r for r in lm.rows(runs[name]/'juzhen/metrics.csv') if r['val_loss'] and int(r['step']) >= 500]
        axes[0].plot([int(r['step'])/1000 for r in rows], [float(r['val_loss']) for r in rows],
                     label=labels[name], color=colors[name], linewidth=2)
    axes[0].set(title='text8: longer training', xlabel='Updates (thousands)', ylabel='Validation CE / BPE token')
    axes[0].legend(frameon=False)
    name = 'tinystories-50000'
    rows = [r for r in lm.rows(runs[name]/'juzhen/metrics.csv') if r['val_loss'] and int(r['step']) >= 500]
    axes[1].plot([int(r['step'])/1000 for r in rows], [float(r['val_loss']) for r in rows], color=colors[name], linewidth=2)
    axes[1].set(title='TinyStories: separate dataset/tokenizer', xlabel='Updates (thousands)', ylabel='Validation CE / BPE token')
    modes = ['greedy', 't06', 't08', 't10']
    for offset, name in enumerate(runs):
        means = [results[name]['repetition'][m]['mean']['repeated_4gram_fraction']*100 for m in modes]
        axes[2].bar(np.arange(4)+(offset-1)*.24, means, width=.24, label=labels[name], color=colors[name])
    axes[2].set(title='Original prompts: trailing whitespace', ylabel='Mean repeated fraction (%)', xticks=np.arange(4),
                xticklabels=['Greedy', 'T=0.6', 'T=0.8', 'T=1.0'])
    axes[2].legend(frameon=False, fontsize=8)
    for offset, name in enumerate(runs):
        means = [results[name]['boundary_repetition'][m]['mean']['repeated_4gram_fraction']*100 for m in modes]
        axes[3].bar(np.arange(4)+(offset-1)*.24, means, width=.24, label=labels[name], color=colors[name])
    axes[3].set(title='Boundary ablation: trailing whitespace removed', ylabel='Mean repeated word 4-gram fraction (%)', xticks=np.arange(4), xticklabels=['Greedy', 'T=0.6', 'T=0.8', 'T=1.0'])
    axes[3].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('Same 13.84M-parameter Juzhen model; training seed 42; generation 192 tokens', fontsize=13)
    fig.savefig(root/'comparison.png', dpi=180)
    fig.savefig(root/'comparison.pdf')
    plt.close(fig)
    old, new = results['text8-10000'], results['text8-50000']
    delta = (1-new['test_perplexity']/old['test_perplexity'])*100
    lines = ['# BPE 三项改进实验', '',
             '三项实验均已完成：text8 延长至 50,000 步、固定生成评估、TinyStories 独立训练。', '',
             '所有完整训练使用 Juzhen 原生 CUDA/FP32，关闭 TF32；PyTorch 仅做首步对照和冻结权重评估。', '',
             '| 实验 | 更新步数 | 选中检查点 | 验证 CE/token | 测试困惑度 |',
             '|---|---:|---:|---:|---:|']
    for name, r in results.items():
        lines.append(f"| {labels[name]} | {r['updates']:,} | {r['selected_step']:,} | {r['validation_loss']:.4f} | {r['test_perplexity']:.2f} |")
    lines += ['', f'同一 text8 数据和词表下，50k 选中模型的测试困惑度相对旧 10k 模型降低 {delta:.1f}%。',
              '新实验使用相同初始权重和冻结数据，但总步数与余弦学习率计划一起改变，不能把收益完全分解为单独的训练时长效应。',
              'TinyStories 的数据和词表不同，困惑度不能与 text8 直接排名。上述选中模型损失由独立 PyTorch 冻结权重评估；每个最终原生模型的测试损失另作一致性检查。', '',
              '更长的 text8 训练降低了测试损失，但没有稳定减少循环重复；在固定 T=0.8 下，50k 模型的重复率反而高于旧 10k 模型，两个提示词版本均如此。',
              'TinyStories 在这组样例中明显较少循环，并能生成故事结构，但仍有角色切换、语义矛盾和因果脱节。提高温度后重复率下降，也不代表逻辑或事实质量提高。',
              '[固定样例的人工阅读记录](qualitative-review.md)保留了具体样例编号和问题，没有把低重复率当作连贯性得分。', '',
              '![实验曲线](comparison.png)', '', '## 数据与预算', '',
              '模型固定为 6 层、宽度 384、6 头、FFN 1536、上下文 128 token、batch 16、词表 8,192，13,835,520 参数。',
              '两轮新训练各呈现 102,400,000 个 token。text8 使用原 90/5/5 连续划分及原词表；词表只从训练数据学习。',
              '每次验证评估 65,536 个目标 token，测试评估 262,144 个目标 token，使用各自划分中的固定均匀分布窗口；这些是抽样估计，不是全量留出集评估。',
              'TinyStories 使用官方 GPT-4 版本的 100,000 篇唯一训练故事，验证和测试各为官方验证来源中的 5,000 篇独立故事。三份数据按空白归一化的故事哈希去重。其训练集约 19.95M token，保留大小写、标点并加入一个故事结束符。',
              'TinyStories 使用专门从其训练故事学习的词表；这是一组完整训练示范，不是只改变单个数据因素的因果实验。', '',
              '## 固定生成评估', '',
              '6 个事先固定的共同小写提示词；每个模型 42 个样例，共 126 个。每次生成 192 个新 token。贪心采样 6 例，其余温度 0.6/0.8/1.0 各为 top-k 40、种子 17 和 42 的 12 例。',
              '重复率定义：1 − 唯一单词四元组数 / 全部单词四元组数，排除提示词并折叠大小写。另保留词数、二元组重复、token 循环、故事结束符数量与第一篇故事指标。',
              '以下为样例重复率的算术平均。它不是语义质量或事实正确率；不同分词器在相同 token 预算下的文本长度也不同。', '',
              '| 实验 | 贪心重复率 | T=0.6 | T=0.8 | T=1.0 | T=0.8 平均词数 |',
              '|---|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        values = [r['repetition'][mode]['mean']['repeated_4gram_fraction']*100 for mode in modes]
        lines.append('| '+labels[name]+' | '+' | '.join(f'{v:.1f}%' for v in values)+f" | {r['repetition']['t08']['mean']['words']:.1f} |")
    lines += ['', 'T=0.8 的第一篇故事与结束符统计（正文遇到首个故事结束符截断；没有结束符的模型保持完整正文）：', '',
              '| 实验 | 平均结束符数 | 首篇平均词数 | 首篇四元组重复率 |', '|---|---:|---:|---:|']
    for name, r in results.items():
        agg = r['repetition']['t08']
        lines.append(f"| {labels[name]} | {agg['mean_eos_count']:.2f} | {agg['first_story_mean']['words']:.1f} | {agg['first_story_mean']['repeated_4gram_fraction']*100:.1f}% |")
    lines += ['', '## 提示词分词边界补充对照', '',
              '原实验结束后发现：六个提示词末尾的空格被单独编码。它在 text8 训练流中只出现 5,162 / 19,455,826 次，TinyStories 中出现 26,364 / 19,953,020 次。这个稀有前缀可能干扰接词，因此额外增加相同六个提示词去掉末尾空格的对照。',
              '这组补充检查是在查看原结果后添加，未替换或删除原来的 126 个样例。采样温度、种子、模型、192-token 预算均不变；又生成 126 个样例，总计 252 个。', '',
              '| 去掉末尾空格的实验 | 贪心重复率 | T=0.6 | T=0.8 | T=1.0 | T=0.8 平均词数 |', '|---|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        values = [r['boundary_repetition'][mode]['mean']['repeated_4gram_fraction']*100 for mode in modes]
        lines.append('| '+labels[name]+' | '+' | '.join(f'{v:.1f}%' for v in values)+f" | {r['boundary_repetition']['t08']['mean']['words']:.1f} |")
    lines += ['', '去掉末尾空格的完整样例：', '',
              '- [text8 10k 对照](evaluation-no-trailing-space/text8-10000/samples.md)',
              '- [text8 50k 对照](evaluation-no-trailing-space/text8-50000/samples.md)',
              '- [TinyStories 50k 对照](evaluation-no-trailing-space/tinystories-50000/samples.md)', '',
              '原始完整样例（未经挑选）：', '',
              '- [text8 10k](evaluation/text8-10000/samples.md)',
              '- [text8 50k](evaluation/text8-50000/samples.md)',
              '- [TinyStories 50k](evaluation/tinystories-50000/samples.md)', '',
              '固定展示：三个模型均取提示词编号 4、T=0.8、seed=17，以下只截取前 900 个字符。']
    for name in runs:
        sample = json.loads((root/'evaluation'/name/'p04-t08-seed17/sample.json').read_text())
        text = sample['prompt']+sample['completion']
        lines += ['', f'### {labels[name]}', '', '```text', text[:900]+('…' if len(text)>900 else ''), '```']
    lines += ['', '## 正确性与复现', '',
              '- CPU/CUDA BPE 梯度、独立进程恢复、最佳检查点、贪心 argmax、采样可重复性与训练文件保护检查通过；旧字符模型 CPU 流水线检查通过。',
              '- TinyStories 的保留集 Unicode 不参与词表合并，故事结束符及三个编码集合 round trip 通过。实际三个集合的重复故事交集为零。',
              '- 每个最终模型的原生测试 CE 与独立 PyTorch SDPA 评估比较；选中的最佳模型验证 CE 与保存记录比较；生成第 0、1、31、127、191 步 logits 独立检查。',
              '- 每个新模型的 `juzhen/best.ckpt` 用于选中模型生成，`last.ckpt` 用于精确恢复。报告使用验证集挑选模型，没有用测试集选检查点。',
              '- 这是一次训练种子（42）的实验；生成种子变化不等同于重复训练，没有统计显著性结论。',
              '- `text8-command.json`、`tinystories-command.json`、`evaluation-commands.json`、`boundary-evaluation-commands.json` 保存完整命令；`data-integrity.json` 保存初始权重与划分检查。',
              '- `code/` 保存本次关键源文件，`working-tree.patch` 保存相对基础提交的全部已跟踪改动；原有未提交 BPE 工作也保留在内。本次没有提交或推送代码。', '',
              '## 来源', '',
              '- [TinyStories 官方数据](https://huggingface.co/datasets/roneneldan/TinyStories)',
              '- [TinyStories 论文](https://arxiv.org/abs/2305.07759)',
              '- 固定数据版本、子集选择规则、各文件哈希见 `tinystories-data/raw/subset.json` 和 `tinystories-data/data/metadata.json`。', '']
    (root/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    source = Path(__file__).resolve().parents[1]
    files = ['examples/demo_text8_lm.cu', 'scripts/bpe_lm.py', 'scripts/text8_lm.py', 'scripts/tinystories_bpe_prepare.py',
             'scripts/bpe_generation_eval.py', 'scripts/bpe_improvements_report.py', 'tests/testBPELM.py',
             'tests/testText8LM.py', 'docs/bpe-lm.md', 'ml/llm.hpp', 'ml/checkpoint.hpp']
    hashes = {}
    for name in files:
        destination = root/'code'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, destination)
        hashes[name] = lm.sha256(destination)
    lm.write_json(root/'source-hashes.json', hashes)
    print(json.dumps(results, indent=2), flush=True)
    print('Report:', root/'README.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--storage-mount', type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve(), args.storage_mount)
