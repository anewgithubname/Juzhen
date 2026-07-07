# 演 (Yan) — Juzhen 的实验描述语言 · 规格草案 v0.1

> 定位:**不是通用编程语言**,而是 Juzhen 后端的演示前端——一门描述机器学习
> 小实验的语言。设计北极星四个词:**可沙箱、可审核、可复现、秒级反馈**。
> 凡是伤害这四点的特性,一律不加。

文件扩展名 `.yan`。一个程序就是一次实验:自上而下执行,无用户定义函数、
无条件分支、无用户循环(迭代只存在于 `train` 等动词内部)。因此程序天然是
一个有向无环的数据流,**所有矩阵维度都可以在运行前静态检查**——错误在提交
时就报出来,而不是训练到一半崩掉。

---

## 1. 语法总则

- 每行一条语句;`#` 到行尾是注释;语句可用缩进续行。
- 两类语句:
  - **绑定**:`name = 表达式`
  - **动词**:`seed` / `train` / `sample` / `generate` / `eval` / `plot`
- 值的类型(用户不写类型,由构造原语决定):

  | 类型 | 含义 | 例 |
  |---|---|---|
  | `Number` / `Vector` | 标量、字面量向量 | `0.5`, `[-2, 0]` |
  | `Distribution` | 惰性分布,可采样、可混合、可变换 | `gaussian([2,0])` |
  | `Dataset` | 有限样本集(可带标签) | `dist ~ 500`, `mnist` |
  | `Net` | 无训练目标的网络骨架 | `mlp(2 -> 64 -> 2)` |
  | `Trainable` | 骨架 + 训练范式 | `rectified_flow(net)` |
  | `Figure` / `Report` | 输出产物,前端渲染 | `plot`/`eval` 的结果 |

- 核心操作符:

  | 操作符 | 含义 |
  |---|---|
  | `~ n` | 从分布采 n 个样本 → `Dataset`;或指定生成/采样数量 |
  | `\|` | 分布混合(可加权:`0.3*A \| 0.7*B`,省略权重则均分) |
  | `->` | 网络层尺寸链 |
  | `\|>` | 把左边的产物送进输出动词(`plot`、`table`) |

---

## 2. 原语(名词)

原语分四组。**v0.1 词汇表刻意只有 20 个左右**;不够用时走 §6 的扩展机制
(在宿主 C++ 层注册新单词),语言本身不变。

### 2.1 玩具分布(Distribution)

| 原语 | 参数(默认) | 说明 |
|---|---|---|
| `gaussian(mean, cov=I)` | mean 向量;cov 标量/向量/矩阵 | 各向同性写 `gaussian([2,0], 0.5)` |
| `uniform(lo, hi)` | 两个向量 | 盒内均匀 |
| `ring(r=1, noise=0.1)` | | 圆环 |
| `spiral(turns=2, noise=0.1)` | | 螺旋(对应 demo_rectified 的目标分布) |
| `moons(noise=0.1)` | | 双月 |
| `checkerboard(k=4)` | | 棋盘格 |

分布变换(仍返回 Distribution,惰性):`shift(d, v)`、`scale(d, s)`、`rotate(d, deg)`。

采样:`X = gaussian([-2,0]) ~ 500` → 500 个二维点的 Dataset。

### 2.2 数据集(Dataset)

| 原语 | 说明 |
|---|---|
| `mnist` | 内置,28×28 灰度 + 0–9 标签,自带 train/test 划分(`mnist.train` / `mnist.test`) |
| `text8` / `enwik8` | 内置字符语料(对应 demo_transformer / demo_discretediffusion) |
| `labeled(名0: 数据0, 名1: 数据1, ...)` | 把若干无标签 Dataset 拼成分类数据集,类名即标签 |
| `split(data, 0.9)` | 划成 `.train` / `.val` 两份 |

数据只有两个来源:**内置数据集**和**分布采样**。没有 `read`/`load`/URL——
这不是缺失,是沙箱边界(§5)。

### 2.3 网络骨架(Net)

| 原语 | 说明 | 对应 Juzhen |
|---|---|---|
| `mlp(d0 -> d1 -> ... -> dk, act=relu)` | 全连接骨架 | `LinearLayer` + `ReluLayer` |
| `cnn(28x28 -> 10)` | 预设小卷积网(结构由后端定,用户只给输入/输出) | `ConvLayer`(需 CUDA/Metal) |
| `transformer(dim=64, blocks=2, heads=4, context=64)` | 字符级 transformer 骨架,词表尺寸由绑定的语料推断 | `TransformerLayer` |

### 2.4 训练范式(Trainable)

骨架不知道自己要学什么;范式把骨架和目标绑在一起,并携带各自的默认超参:

| 原语 | 学什么 | 对应 demo |
|---|---|---|
| `classifier(net)` | softmax 交叉熵分类 | demo_mnist / demo_classification |
| `regressor(net)` | 最小二乘回归 | demo_classification 变体 |
| `rectified_flow(net)` | 直线流匹配,噪声→数据 | demo_rectified |
| `lm(net)` | 自回归下一字符预测 | demo_transformer |
| `masked_diffusion(net)` | 吸收态离散扩散(遮罩去噪) | demo_discretediffusion |

---

## 3. 动词

### `seed n`
全局随机种子。**每个程序第一句必须是 seed**(省略时解释器自动补 `seed 0`
并回显在结果里)——这是"可复现"的一半;另一半见 §5。

### `train X on data [for N steps] [with 参数表]`
训练一个 Trainable。运行期间以 `(step, train_loss, val_loss)` 流式回报,
前端画实时损失曲线。

`with` 可覆盖的参数及默认值(默认即 Juzhen demo 中调好的值):

| 参数 | 默认 | 说明 |
|---|---|---|
| `lr` | `1e-4` | Adam 学习率(后端即 `adam_state(.0001)`) |
| `batch` | `128` | |
| `steps` | 范式各异:classifier `2000`,rectified_flow `5000`,lm/diffusion `10000` | `for N steps` 是它的糖 |

优化器 v0.1 固定为 Adam,不暴露选择——可覆盖的东西每多一个,
"三行就能跑"的体验就弱一分。

### `sample X ~ n`
从训练好的生成模型采样,返回 Dataset。对 `rectified_flow` 即从噪声积分到
数据端(默认 50 步 Euler)。

### `generate X from "prompt" ~ n chars [at temperature t]`
语言模型专用:自回归续写(`lm`)或遮罩填空(`masked_diffusion`,prompt 中
`_` 为洞)。默认 `temperature 0.8`。逐字符流式输出,前端打字机效果。

### `eval X on data`
返回 Report:分类给 loss + accuracy,流给平均传输代价,LM/扩散给 bpc。
`|> table` 渲染成表(不加管道时默认也显示)。

### `plot 目标`
| 形式 | 效果 |
|---|---|
| `plot data` | 二维散点(仅二维 Dataset;MNIST 用 `plot data ~ 16` 画 16 张样本图) |
| `plot data1, data2` | 叠加对比,自动配色 |
| `plot trajectory of X ~ n` | 流模型专用:n 个粒子从噪声到数据的运动轨迹/动画 |
| `plot loss of X` | 训练曲线(train 时已实时显示,此为显式导出) |

---

## 4. 示例程序

### 4.1 双高斯 + rectified flow(动机场景)

```yan
seed 42

data = gaussian([-2, 0], 0.3) | gaussian([2, 0], 0.3)   # 双高斯混合
X    = data ~ 1000

flow = rectified_flow(mlp(2 -> 64 -> 64 -> 2))
train flow on X for 5000 steps

plot X, sample flow ~ 1000          # 真实样本 vs 生成样本
plot trajectory of flow ~ 200       # 粒子从噪声流向双高斯的动画
```

### 4.2 MNIST 分类

```yan
seed 0

net = classifier(mlp(784 -> 128 -> 32 -> 10))
train net on mnist.train for 2000 steps with batch=64

eval net on mnist.test |> table
```

### 4.3 字符 transformer 语言模型

```yan
seed 7

m = lm(transformer(dim=64, blocks=2, context=64))
train m on text8 for 10000 steps

generate m from "the meaning of life is " ~ 200 chars at temperature 0.8
```

### 4.4 离散扩散填空(bonus)

```yan
seed 7

m = masked_diffusion(transformer(dim=64, blocks=2, context=64))
train m on text8 for 10000 steps

generate m from "alice was ____ to see the ____" ~ 1 chars   # 填洞
```

### 4.5 玩具二分类(展示 labeled 和分布变换)

```yan
seed 1

pos  = ring(1.0, 0.05) ~ 400
neg  = gaussian([0, 0], 0.1) ~ 400
data = split(labeled(inside: neg, outside: pos), 0.9)

net = classifier(mlp(2 -> 32 -> 32 -> 2))
train net on data.train

eval net on data.val |> table
plot data.val                        # 按预测类别着色
```

---

## 5. 沙箱与可复现性(语言级保证)

这门语言存在的首要技术理由:**解释器即安全边界**。

- **无 IO 原语**。没有文件、网络、环境变量;数据只能来自内置数据集与分布
  采样。这些不是"待实现的功能",是永久排除项。
- **不图灵完备**。无用户循环/递归,程序长度有限 ⇒ 执行步数有上界,
  停机是显然的。
- **资源限额**(解释器在 parse 后、执行前静态核算,超限直接拒绝):
  - 单个 Dataset ≤ 100k 样本;单矩阵 ≤ 10^7 元素
  - 总训练 steps ≤ 50k;单程序墙钟 ≤ 60 s(超时软停,已完成部分照常回报)
- **可复现**:程序文本 + seed 完全决定结果。CPU 单线程后端逐比特可复现;
  GPU 后端保证统计意义上可复现(浮点归约顺序差异)。回显给用户的结果页
  永远附带程序文本和 seed,可一键重跑。
- **静态形状检查**:因为程序是 DAG,所有维度在执行前全部检查
  (如 `mlp(2 -> ...)` 接 `mnist` 会在提交时报 `expected input dim 784, got 2`)。

## 6. 扩展机制(逃生舱)

用户想要词汇表之外的东西(自定义 loss、新采样器)时,**语言不长大,宿主长大**:
在 C++ 层用注册宏把新原语挂进词汇表,DSL 只是多了一个名词/动词参数:

```cpp
YAN_REGISTER_DISTRIBUTION("two_rings", [](Args a){ ... });   // 立刻可写 two_rings(...)
YAN_REGISTER_OBJECTIVE("vae", ...);                          // 立刻可写 vae(net)
```

语法、解释器、沙箱边界都不动。这是防止"六个月后重新发明一个更差的 Python"
的结构性约束。

## 7. 执行模型(实现备忘)

- 递归下降 parser → 静态检查(名字解析、形状核算、资源限额)→ tree-walking
  解释器,每个动词映射到一段现成的 Juzhen demo 代码。预计 1–2k 行 C++。
- `train` 内部按 step 回调 `(step, loss, val_loss)`;`generate` 按字符回调。
  服务端经 WebSocket 推给前端;本地 CLI 模式打印到终端。
- 后端类型(`float` / `CUDAfloat` / `MPSfloat`)对语言不可见,由服务器构建
  决定——与 Juzhen 现有的 `#define FLOAT` 模式一致。
- v0.1 明确不做:save/load 权重(IO)、优化器选择、学习率调度、
  多 GPU、用户自定义层。

---

### 附:EBNF 速览

```
program   = { statement } ;
statement = seed | binding | verb ;
seed      = "seed" NUMBER ;
binding   = IDENT "=" expr ;
verb      = train | sample | generate | eval | plot ;

train     = "train" IDENT "on" expr [ "for" NUMBER "steps" ] [ "with" kwargs ] ;
sample    = "sample" IDENT "~" NUMBER ;
generate  = "generate" IDENT "from" STRING "~" NUMBER "chars"
            [ "at" "temperature" NUMBER ] ;
eval      = "eval" IDENT "on" expr [ "|>" "table" ] ;
plot      = "plot" plottarget { "," plottarget } ;

expr      = mixture ;
mixture   = weighted { "|" weighted } ;
weighted  = [ NUMBER "*" ] postfix ;
postfix   = primary { "~" NUMBER | "." IDENT } ;
primary   = call | IDENT | NUMBER | vector | "(" expr ")" ;
call      = IDENT "(" [ arglist ] ")" ;
arglist   = arg { "," arg } ;  arg = [ IDENT ":" ] expr | dims ;
dims      = NUMBER "->" NUMBER { "->" NUMBER } ;
```
