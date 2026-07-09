# demo_word2vec 设计稿：SGNS 词向量 + 终端词语算术

在 text8（`datasets/text8`，已在仓库中）上用 skip-gram with negative sampling
训练词向量，全部计算表达为 Juzhen 现有的矩阵原语（GEMM + 逐元素操作），
不依赖 `ml/layer.hpp`，梯度手推。训完在终端做最近邻查询和 `king − man +
woman ≈ queen` 式的类比测试。

---

## 1. 训练目标函数

### 1.1 原始 skip-gram

给定中心词 $c$ 及其窗口内的上下文词 $o$，skip-gram 最大化

$$P(o \mid c) = \frac{\exp(u_o^\top v_c)}{\sum_{w=1}^{V} \exp(u_w^\top v_c)}$$

- $v_c \in \mathbb{R}^d$：中心词向量，参数矩阵 $W_{\text{in}} \in \mathbb{R}^{d \times V}$ 的第 $c$ 列；
- $u_o \in \mathbb{R}^d$：上下文词向量，$W_{\text{out}} \in \mathbb{R}^{d \times V}$ 的第 $o$ 列。

分母对全词表求和，每步 $O(V)$，太贵——负采样就是绕开它。

### 1.2 负采样目标（SGNS）

把多分类换成二分类：真实共现对 $(c,o)$ 判"真"，随机拼凑对 $(c,n_k)$ 判"假"。
单个正样本对配 $K$ 个负例的损失：

$$\ell(c,o) = -\log \sigma(u_o^\top v_c) - \sum_{k=1}^{K} \log \sigma(-u_{n_k}^\top v_c),
\qquad \sigma(x) = \frac{1}{1+e^{-x}}$$

负例从 $P_n(w) \propto \mathrm{freq}(w)^{3/4}$ 采样。总目标是对语料中所有
(中心词, 窗口内上下文词) 对求和：$\mathcal{L} = \sum_{(c,o)} \ell(c,o)$。

最优解满足 $u_o^\top v_c \approx \mathrm{PMI}(c,o) - \log K$
（Levy & Goldberg 2014），即隐式做 PMI 矩阵分解——这是词向量线性算术
成立的理论依据。从密度比估计的角度看，logistic loss 在估计
$\log\frac{P(o|c)}{P_n(o)}$，与 InfoNCE/SimCLR 同源（见 §6）。

### 1.3 批量化形式（实际实现的目标）

一批 $B$ 个正样本对共享同一组 $K$ 个负例（SimCLR 式 in-batch 共享，
把负例打分变成单个 GEMM）：

$$\mathcal{L}_{\text{batch}} = -\sum_{b=1}^{B} \log \sigma(s^{+}_b)
- \sum_{b=1}^{B}\sum_{k=1}^{K} \log \sigma(-S^{-}_{kb})$$

其中（列向量按批排列）：

| 记号 | 定义 | 形状 | Juzhen 表达 |
|---|---|---|---|
| $V_c$ | $W_{\text{in}} X_c$ | $d \times B$ | GEMM（one-hot 代替 lookup） |
| $U_o$ | $W_{\text{out}} X_o$ | $d \times B$ | GEMM |
| $U_n$ | $W_{\text{out}} X_n$ | $d \times K$ | GEMM |
| $s^{+}$ | $\mathbf{1}_d^\top (V_c \odot U_o)$ | $1 \times B$ | `sum(hadmd(Vc,Uo), 0)` |
| $S^{-}$ | $U_n^\top V_c$ | $K \times B$ | `Un.T() * Vc` |

$X_c, X_o \in \{0,1\}^{V\times B}$、$X_n \in \{0,1\}^{V\times K}$ 是 host 端
填充的 one-hot 矩阵（模式同 `demo_tinystories` 的 one-hot 输入构造）。

### 1.4 梯度（手推，全部是现成矩阵操作）

利用 $\frac{d}{dx}\log\sigma(x) = 1-\sigma(x)$：

$$g^{+} = \sigma(s^{+}) - 1 \in \mathbb{R}^{1\times B}, \qquad
G^{-} = \sigma(S^{-}) \in \mathbb{R}^{K\times B}$$

对中间量：

$$\frac{\partial \mathcal{L}}{\partial V_c} = U_o \,\mathrm{diag}(g^{+}) + U_n G^{-}
\qquad
\frac{\partial \mathcal{L}}{\partial U_o} = V_c \,\mathrm{diag}(g^{+})
\qquad
\frac{\partial \mathcal{L}}{\partial U_n} = V_c (G^{-})^\top$$

（$\mathrm{diag}(g^{+})$ 按列缩放，用行广播实现：`hadmd(Uo, ones(d,1)*gpos)`，
`ones(d,1)*gpos` 是一个 rank-1 GEMM，与 LayerNorm 中的既有惯用法一致。）

散射回参数矩阵是 one-hot 的转置 GEMM（即 scatter-add），SGD 更新：

$$W_{\text{in}} \mathrel{-}= \eta \cdot \frac{\partial \mathcal{L}}{\partial V_c} X_c^\top,
\qquad
W_{\text{out}} \mathrel{-}= \eta \left( \frac{\partial \mathcal{L}}{\partial U_o} X_o^\top
+ \frac{\partial \mathcal{L}}{\partial U_n} X_n^\top \right)$$

梯度按批平均（除以 $B$），学习率 $\eta$ 从 10 线性衰减到 0.1（等效单对步长约 0.01,与原版 word2vec 的 0.025 同量级）。

σ 的实现：`1.0 / (exp(-S) + 1.0)`（现成的 `exp` + 标量除法算子）。

---

## 2. 实现步骤

### Phase 1 — 数据管线（host，~80 行）

1. 读入 `datasets/text8`（单行、空格分词、纯小写）。
2. 词频统计，取 top `V−1`（默认 V=30000），其余映射 `<unk>`；建 `word→id`
   哈希表和 `id→word` 数组。
3. 语料编码为 `vector<int32_t>`（~17M token）。
4. **高频词下采样**：token $w$ 以概率 $(\sqrt{f/t}+1)\,t/f$ 保留（$t=10^{-4}$，
   $f$ 为词频占比,原版 word2vec 公式）。编码时一次性完成；text8 的 1700 万
   token 下采样后剩约 900 万。
5. 负采样表：unigram$^{3/4}$ 归一化后建累积分布数组，采样时二分查找。

### Phase 2 — 训练循环

每步（host 采样 + device 计算）：

1. 随机采 $B$ 个位置作中心词；每个位置采动态窗口宽 $r \sim U\{1,w\}$，
   在窗口内随机取一个上下文词 → $B$ 个 (c, o) 对。
2. 采 $K$ 个共享负例。
3. 上传 (1×B) 的 id 小矩阵,device 端 kernel 构造 one-hot（CUDA；其余
   后端在 host 端填充,模式同 `demo_tinystories`）。
4. 按 §1.3–1.4 执行：3 个 embedding GEMM → 打分 → σ → 梯度 →
   2 个 scatter GEMM → SGD 更新。
5. 每 `log_every` 步打印运行平均 loss 与当前学习率。

工程约定（沿用 `demo_tinystories` 模式）：

- `STEPS` / `BATCH` / `DIM` / `VOCAB` 环境变量覆盖默认值；
- 训练结束把 `W_in` + 词表写入权重缓存文件（`W2V_WEIGHTS` 指定路径，
  默认在数据目录），存在则跳过训练直接进入交互；
- 数据目录 `W2V_DATA` 覆盖，默认 `PROJECT_DIR/datasets`。

### Phase 3 — 评估与交互

1. 训完把 $W_{\text{in}}$ 每列 L2 归一化（`square`/`sum`/`eleminv` 拼装）。
2. **最近邻**：查询向量 $q$，`scores = W_in.T() * q`（GEMV），host 端
   top-k（排除查询词自身）。
3. **类比**：`q = v(a) − v(b) + v(c)` 归一化后同上，排除 a/b/c 三词。
4. 自动验收：内置一组经典类比（见 §4），打印 top-3 命中率。
5. 交互模式（stdin 循环）：
   - `word` → 10 个最近邻；
   - `a - b + c` → 类比 top-5；
   - 空行退出。

### Phase 4 — 集成

- `examples/demo_word2vec.cu` 单文件；`CMakeLists.txt` 加一行
  `GENERATE_PROJ(demo_word2vec examples/demo_word2vec.cu)`。
- `add_test(test29)`：`STEPS=1200 VOCAB=5000 BATCH=256 DIM=64 NEG=16`
  冒烟跑（<1 分钟）,断言 loss 降到初值 70% 以下;text8 缺失时退出码 77
  (跳过)。缓存写到 build 目录,不覆盖真实训练缓存。

---

## 3. 默认超参数

| 参数 | 默认 | 说明 |
|---|---|---|
| d | 200 | 向量维数 |
| V | 30000 | 词表（含 `<unk>`） |
| B | 1024 | 批大小（正样本对数） |
| K | 64 | 共享负例数 |
| w | 8 | 最大窗口半径（动态窗口） |
| t | 1e-4 | 下采样阈值 |
| STEPS | 300000 | ~34 遍下采样后语料（3.07 亿样本对） |
| lr | 10 → 0.1 | 批平均梯度,线性衰减 |

## 4. 规模估算与验收标准

- 实测（RTX 4090,默认配置 300k 步）：**约 9 分钟**;初始 loss = 
  $(1+K)\log 2 = 45.05$（与理论值逐位吻合,可当正确性检验）,收敛到 ~4.5。
- 显存：参数矩阵 2×(200×30000×4B) ≈ 48MB,one-hot 批矩阵 ~120MB。
- 实测验收（内置 5 道类比题,top-3 命中 4/5）：
  - `king − man + woman` → **queen**（第 1 名）;
  - `bigger − big + small` → `smaller`、`brother − he + she` → `sister`、
    `london − england + germany` → `berlin` 均命中;
  - `paris − france + italy` → 给出 venice/turin/bologna（方向正确但
    `rome` 未进 top-5——text8 已知难例,"rome" 被古罗马语境主导）;
  - `cat` 最近邻：dog, rat, dogs, hamster, bird…
- 换 fil9（~1.24 亿 token）可解锁 rome 这类难例,`W2V_DATA` 指过去即可。

## 5. 已知取舍

- **one-hot GEMM 代替 gather**：浪费算力但零新 kernel，且 V=3 万时
  单步 <2ms，不值得为此写 embedding-lookup kernel。
- **共享负例**：每个中心词面对同一组负例，梯度间有相关性；文献与实践
  上对词向量质量无可感知影响，换来单 GEMM 打分。
- **SGD 而非 Adam**：word2vec 原版即 SGD + 线性衰减,足够；省两份
  optimizer 状态。

## 6. 可选扩展：InfoNCE 对照实验（+~10 行）

把 §1.3 的损失换成 softmax 交叉熵——对每个 $b$，在 $[s^{+}_b; S^{-}_{:,b}]$
这 $K{+}1$ 个得分上做 log-softmax，取正例项：

$$\mathcal{L}_{\text{InfoNCE}} = -\sum_b \log
\frac{\exp(s^{+}_b/\tau)}{\exp(s^{+}_b/\tau) + \sum_k \exp(S^{-}_{kb}/\tau)}$$

环境变量 `LOSS=nce|infonce` 切换（`TAU` 调温度），跑同样的类比验收集对比
两种对比损失——即 word2vec (2013) vs SimCLR (2020) 损失形式的最小对照。

**已实现并实测**（text8,默认配置,InfoNCE 用 LR=40）：两种损失都是
**4/5 命中,king−man+woman=queen 均排第 1**——在这个规模上损失形式
不影响词向量质量。两点值得记录的差异：
- InfoNCE 的梯度质量(≤2/对)比 SGNS(~(K+1)/2·σ 量级)小得多,需要 ~4 倍
  学习率才有相近的训练动态;
- InfoNCE 的 loss 有理论下界 $\log(K{+}1)-I(c;o)$,词-上下文互信息只有
  ~1 nat,所以 loss 只从 log 65 = 4.17 降到 3.59——loss 的"下降幅度"
  在两种损失之间不可比,评向量质量只能看下游任务。
