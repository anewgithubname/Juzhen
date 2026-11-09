# Python 接口计划（草案，待审查）

- 依据：2026-11-01 至 11-02 关于 Python 接口的讨论。本文按讨论的最终结论整理。讨论中途改过的地方，以后来的结论为准，并在文中注明。
- 基准：`main` @ `580310a`。
- 分支：`python-interface`。
- 状态：第 0 步已完成（PR #31 加 `juzhen-init` 分支上的提交）；第 1、2 步已在 `python-interface` 分支上实现，见第 13 节。

## 1. 目标

用 Juzhen 做底层，打造一个**简单、即学即用、没有高门槛**的机器学习“语言”。用 AI 写代码的是开发者，这个工具是给**使用者**用的：初学者不会写 C++，也不该被 C++ 挡在门外。

由这个目标得出：

1. **用 Python，不发明新语言。** 新语法反而抬高门槛：要学新语法，没有现成的教程和工具，出了错也搜不到答案。Python 是机器学习入门门槛最低的语言，在它上面做一个极简的库最合适。
2. **安装必须一条命令搞定。** `pip install juzhen` 就能用，不需要编译器、CUDA 或 ROCm。所以打包发布从“锦上添花”变成了**必须做**。先只发 CPU 版和 Metal 版的预编译 wheel，初学者的练习（MNIST、小网络）在 CPU 上就够了。
3. **数据布局是“每行一个样本”**（N×d），见 5.1。
4. **接口分层，从简单到底层：**
   - 第一层：`fit()` / `evaluate()`，五行代码完成训练；
   - 第二层：显式训练循环，`zero_grad()` → 前向 → `backward()` → `step()`；
   - 第三层：矩阵运算，`@`、`exp`、`sum` 这些；
   - 第四层：Juzhen 的 C++ 源码，写得清楚，能直接打开看。
5. **白盒是 Juzhen 的卖点。** 每一层都能打开，看到下一层是怎么实现的。PyTorch 的底层是庞大的 C++ 和 CUDA 代码，初学者基本看不懂；Juzhen 从 `fit()` 一路往下到矩阵运算都能读懂。没有这样一个明确的卖点，学习者很难不去选教程最多的 PyTorch。

## 2. 总体结构

**两层结构：**
- 底层用 nanobind 写一层薄绑定 `juzhen._core`，C++ 接口基本一对一暴露；
- 上层用纯 Python 写 `juzhen` 包，负责好用的 API、和 numpy 互转，以及 `nn` / `optim` 的封装。

C++ 本身尽量少改。

**数据一直留在 C++ 侧**，GPU 版本就留在显存里。Python 里的矩阵只是一个句柄：它包着一个 `Matrix<D>`，`Matrix<D>` 里是指向 Juzhen 内存池的 `shared_ptr`，Python 手上从来没有真正的数据。以 `C = jz.log(jz.exp(A @ B) + 1)` 为例：nanobind 只传句柄，C++ 在原来的设备上计算，结果从内存池分配，再包成新句柄返回。

**不用 DLPack，坚持用 Juzhen 自己的内存管理。** 每个矩阵的存储都从 `Memory<D>` 分配，也由它回收。Juzhen 不借用外部内存，也不把自己的内存借给别人。数据进出一律复制：

| 方向 | 做法 |
|---|---|
| numpy → Juzhen | `jz.array(a)`：从内存池分配，再 memcpy 进来。GPU 版本直接复制到显存，只复制一次 |
| Juzhen → numpy | `.numpy()`：复制出一份新的 numpy 数组，GPU 版本会先 `to_host()` |
| 和 PyTorch 交换 | 经过 numpy 中转：`jz.array(t.cpu().numpy())`、`torch.from_numpy(A.numpy())` |
| 读一个标量或单个元素 | `.item()`、`print(C)`、`C[i, j]`：只读出少量数据 |

不用 DLPack 的好处：
- 内存只有一个主人，生命周期最简单；
- 不用排查“代码里哪里默认数据来自内存池”，因为这个假设一直成立；
- 内存池复用照常生效。

代价：和 PyTorch 交换 GPU 数据要绕一圈，从显存到主机再回显存。训练循环本身完全在 Juzhen 里，不受影响。

## 3. 代码里决定设计的几处限制，以及第 0 步

| 现状 | 位置 | 对 Python 的影响 | 第 0 步怎么改 |
|---|---|---|---|
| 出错时 `ERROR_OUT` 调用 `exit(1)` | `cpp/helper.hpp` | 一个错误就会杀掉整个 Python 进程或 Jupyter kernel | **已完成（PR #31）**：抛 `Juzhen::Error`，nanobind 会把它转成 Python 异常 |
| `Memory::free` 在 `shared_ptr` 删除器里调用，不能抛异常 | `cpp/memory.hpp` | 析构时抛异常会直接 `std::terminate` | **已完成（PR #31）**：只记日志，不抛异常 |
| 初始化写在 `main()` 里：cuBLAS 句柄、`mpsInit()`、各个 `Memory<>` 守卫对象 | `cpp/launcher.cu` | Python 没有 `main()` | 抽成 `juzhen::init()` / `juzhen::shutdown()`，命令行启动器、iOS 的 `Run`（`ios/JuzhenDemo/JuzhenRunner.mm`）和 Python 共用这一份 |
| `Memory<D>` 的析构函数会释放**所有**块，包括还在用的 | `cpp/memory.hpp` | Python 对象什么时候销毁不确定。如果矩阵比守卫对象活得久，就会出问题 | Python 里**不创建这种守卫对象**，内存池一直活到进程结束。另加 `Memory<D>::release_cached()`，只释放空闲块，对应 `jz.empty_cache()` |
| 内存池的表、`global_rand_gen`、cuBLAS 句柄都是全局静态的，不是线程安全的 | memory / helper | 多线程同时调用会出错 | 先全程持有 GIL；以后要释放 GIL，就照 iOS 版加一把全局锁（`juzhen_mutex`） |
| 编译时只能选一个后端：CUDA、ROCm 或 Metal | CMake | — | 每个后端单独出一个 wheel |
| 按列存储，带懒转置标记；数据约定是特征 × batch（d×N） | `cpp/core.hpp` | 和 numpy、PyTorch 的 N×d 相反 | 见 5.1 |
| `Sequential` 存裸 `Layer*` 和指向输入的指针；`eval` 返回层内部 `val` 的引用；层列表是输出层在前 | `ml/layer.hpp` | Python 这边要管好这些对象的生命周期 | 见第 6 节 |
| 拷贝构造会做深拷贝 | `cpp/core.hpp` | Python 里可以直接按值持有 C++ 对象 | — |

## 4. 用起来的样子

以下都是设计示例，接口还没实现。

### 4.1 第一层：fit

```python
import juzhen as jz

X, y, Xt, yt = jz.datasets.mnist()          # 每行一个样本

model = jz.Sequential(
    jz.Dense(256, activation="relu"),
    jz.Linear(10),                           # 输出层不加激活
)
model.fit(X, y, epochs=3)                    # 一行训练
print(model.evaluate(Xt, yt))
```

### 4.2 第三层：hello world

内容和 README 里 C++ 版的 hello world 相同，数值也应该完全一样，因为底层是同一套 C++ 计算：

```python
import numpy as np
import juzhen as jz

A = jz.array([[1, 2, 3],
              [4, 5, 6]])                # 2×3，在 CPU 上
B = jz.array([[.1, .2],
              [.3, .4],
              [.5, .6]])                 # 3×2

C = jz.log(jz.exp(A @ B) + 1) / 5        # @ 是矩阵乘法
print(C)

# 换到 GPU：只加一个 device 参数，计算代码不用改
Ag = jz.array([[1, 2, 3], [4, 5, 6]], device="gpu")
Bg = jz.array([[.1, .2], [.3, .4], [.5, .6]], device="gpu")
Cg = jz.log(jz.exp(Ag @ Bg) + 1) / 5

print(np.allclose(C.numpy(), Cg.numpy(), atol=1e-5))
print(jz.backend())                      # "rocm"、"cuda"、"metal" 或 "cpu"
```

和 C++ 版的对应关系：
- **矩阵乘法写 `@`，`*` 是逐元素乘**，跟 numpy 一致。C++ 里 `A*B` 是矩阵乘法、逐元素乘是 `hadmd`。不这么改，Python 用户几乎一定会踩坑，文档里要写清楚。
- **不用写 `compute()` 和 `main()`**：`import juzhen` 时自动调用 `juzhen::init()`。
- **换 GPU 只改 `device`**：C++ 里是把 `float` 换成 `CUDAfloat` 或 `ROCMfloat`。Python 里一个 wheel 只编一个 GPU 后端，所以写 `device="gpu"` 就够了。

### 4.3 第二层：demo_mnist

照着 `examples/demo_mnist.cu` 改写，网络结构、超参数和训练流程都和 C++ 版一致。讨论时的写法是特征 × 样本，这里已经按后来定的“每行一个样本”改过：

```python
import time
import juzhen as jz
from juzhen import nn, optim

jz.manual_seed(1)
t0 = time.perf_counter()

k, batch = 10, 32
X, Y, XT, YT = jz.datasets.mnist()       # 每行一个样本，Y 是 one-hot
num_batches = X.shape[0] // batch
XT, YT = XT.to("gpu"), YT.to("gpu")      # 测试集一次性复制到 GPU

net = nn.Sequential(                     # 输入层在前；内部反转成 C++ 要的输出层在前
    nn.Dense(784, 1024),                 # tanh 激活，对应 C++ 的 Layer<D>
    nn.Dense(1024, 128),
    nn.Linear(128, k),                   # 无激活，对应 LinearLayer<D>
    batch_size=batch, device="gpu")
opt = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0)  # 和 C++ 版一样关掉权重衰减

for it in range(10000):
    b = it % num_batches
    X_i = X[b * batch:(b + 1) * batch].to("gpu")
    Y_i = Y[b * batch:(b + 1) * batch].to("gpu")

    opt.zero_grad()
    pred = net(X_i)
    loss = nn.softmax_cross_entropy(pred, Y_i)   # 返回 (value, grad)
    net.backward(loss.grad)
    opt.step()

    if it % 1000 == 0:
        err = nn.error_rate(net(XT), YT).item()  # .item() 会等 GPU 算完，所以每 1000 步才读一次
        print(f"Misclassification Rate: {err:.4f}")

print(f"time: {(time.perf_counter() - t0) * 1000:.0f} ms")
```

| C++ | Python |
|---|---|
| `#define FLOAT CUDAfloat` 等后端宏 | `device="gpu"` |
| `Layer<FLOAT> L0(1024, d, batchsize)` | `nn.Dense(d, 1024)` |
| `list<Layer<FLOAT>*> trainnn({&L2, &L1, &L0})`，输出层在前 | `nn.Sequential(...)`，输入层在前 |
| `AdamW<FLOAT> optimizer(network.parameters(), options)` | `optim.AdamW(net.parameters(), ...)` |
| `network.eval(X_i)` → `network.backward(...)` → `optimizer.step()` | `net(X_i)` → `net.backward(...)` → `opt.step()` |
| `error_rate(...).to_host().elem(0, 0)` | `nn.error_rate(...).item()` |

数据流和 C++ 版完全一样：
- 训练集留在主机，每个 batch 切出来复制一次到 GPU；
- 前向、反向和 AdamW 更新都在 C++ 里、在显存上完成；
- Python 每一步只调用 5 次左右。

测试集评估时，`net(XT)` 一次性送入 10000 个样本，比训练的 batch 大。C++ 版也是这样做的。

## 5. 矩阵层：`juzhen.Matrix`

```python
A = jz.array(np.random.randn(3, 4).astype(np.float32), device="gpu")
B = jz.randn(4, 2, device="gpu")
C = jz.log(jz.exp(A @ B) + 1) / 5
D = A * A                              # 逐元素乘
C.T, C.shape, C.device                 # .T 是懒转置，不搬数据
C[0:2, :], C[:, [0, 2]]                # 对应 slice / rows / columns
x = C.numpy()
```

- **只支持 float32。** 每个 wheel 只编一个设备类型：GPU 是 `CUDAfloat`、`ROCMfloat`、`MPSfloat` 之一，CPU 是 `float`。
- **生命周期。** Python 对象按值持有 `Matrix<D>`。它的 `shared_ptr` 删除器会把内存还给内存池，而内存池一直活着，所以是安全的。
- **绑定写成模板**，例如 `bind_matrix<D>(m)`，各后端共用一份代码。

### 5.1 数据布局：每行一个样本

讨论中途改过：最早建议沿用特征 × 样本，明确了低门槛的目标之后，改成了每行一个样本（N×d），因为几乎所有教程、numpy、pandas 都是这样。

这几乎没有额外开销：**C 顺序的 N×d numpy 数组，在内存里和 Juzhen 想要的按列存储的 d×N 矩阵逐字节相同。** 所以：
- 复制进来只需要一次 memcpy，再设一下转置标记，逻辑形状就和 numpy 一致；层内部照常用 d×N；
- `.numpy()` 反过来，按转置标记返回 C 顺序或 F 顺序的数组。

## 6. 神经网络层：`juzhen.nn` / `optim`

- **内置层用 C++ 里手写好的显式 backward，不做计算图。** 这和 C++ 的语义一致，绑定层也能保持很薄。
- **Python 的 `Sequential` 持有所有层对象**，通过 `nb::keep_alive` 或一个 Python 列表引用着。它还会把最近一次的输入一直引用到 `backward` 调用为止，满足 C++ 那边对输入指针的要求。
- **`forward` 返回拷贝，不返回 `val` 的引用**，因为下一次 `eval` 会覆盖它。如果核实过每一层的 `eval` 都是赋一个新 buffer 给 `val`，以后可以改成共享 buffer，省掉这次拷贝。
- **只暴露现有的层：** Dense（tanh）、Linear、ReLU、Conv2d、ConvTranspose2d、LayerNorm、TransformerLayer。每个后端的 Conv 特化版本，由编译时选好的设备类型自动选择。
- **其他接口：**
  - `net.jvp(X, tangent)`
  - `jz.save_checkpoint(net, opt, "run.ckpt")`
  - `jz.manual_seed(n)`：CUDA 和 Metal 上对应 `GPUSampler`。讨论时 ROCm 的 GPU 种子是写死的，现在已由 PR #31 修好。
- **优化器更新留在 C++ 的 `AdamW::step()` 里**，不在 Python 里逐层或逐个参数循环调用 C++。

## 7. 自定义层

有三种做法，按推荐顺序：

1. **能用现有层拼出来就拼。** `nn.Sequential` 嵌套就行，不需要写新代码。
2. **在 Python 里继承 `nn.Layer`，用 Juzhen 的矩阵运算写。** 写起来最方便，适合由少量大运算组成的层。
3. **在 C++ 里继承 `Layer<D>`，再编进扩展模块。** 性能最好，适合运算很多、很碎，或者需要自定义 GPU kernel 的层。

### 7.1 Python 自定义层（手写反向）

层的接口和 C++ 的 `Layer<D>` 一一对应：`forward` 对应 `eval`，`backward` 对应 `backward`，`jvp` 对应 `jvp`，参数注册对应 `parameters()`。示例是一个带 SiLU 激活的全连接层，y = z ⊙ σ(z)，z = Wx + b（这里沿用讨论时特征 × 样本的写法）：

```python
class SiLULinear(nn.Layer):
    def __init__(self, n_in, n_out, device="gpu"):
        super().__init__()
        # 注册参数：矩阵、梯度缓冲区和 Adam 状态都在 C++ 侧创建
        self.W = self.param("weight", jz.randn(n_out, n_in, device=device) * 0.01)
        self.b = self.param("bias",   jz.zeros(n_out, 1, device=device), decay=False)

    def forward(self, x):
        ones = jz.ones(1, x.shape[1], device=x.device)
        z = self.W @ x + self.b @ ones          # 偏置按列广播
        s = 1 / (1 + jz.exp(-z))                # sigmoid，用现有运算拼出来
        self._z, self._s = z, s                 # 缓存给 backward 用
        return z * s

    def backward(self, x, dy):
        z, s = self._z, self._s
        dz = dy * (s + z * s * (1 - s))         # SiLU 的导数
        self.add_grad(self.W, dz @ x.T)         # 累加参数梯度
        self.add_grad(self.b, jz.sum(dz, axis=1))
        return self.W.T @ dz                    # 对输入的梯度

    # jvp 可选；没写的话，调用 net.jvp() 时会报错说这一层不支持

net = nn.Sequential(SiLULinear(784, 256), nn.Linear(256, 10), batch_size=32, device="gpu")
opt = optim.AdamW(net.parameters(), lr=1e-4)    # 自定义层的参数也会被 AdamW 更新
jz.save_checkpoint(net, opt, "run.ckpt")        # 也会被保存进 checkpoint
```

背后是怎么接上的：
- **通过 trampoline 类挂进 C++ 的网络。** 绑定层有一个 C++ 类 `PyLayer : Layer<D>`，它的 `eval`、`backward`、`jvp` 都转去调用 Python 的方法（`NB_OVERRIDE`）。所以 C++ 的 `Sequential`、`forward()`、`backward()` 把 Python 层当普通的 `Layer<D>*` 用。
- **`self.param()` 在 C++ 侧创建参数**，包括 `Matrix<D>`、梯度缓冲区和 `adam_state`。这样 AdamW 能原地更新它们，checkpoint 也能找到它们，符合“数据由 Juzhen 内存池管理”的原则。
- **`self.add_grad()` 对应 C++ 的 `add_parameter_gradients`**，同一个参数多次累加时会自动相加。

限制：
- **Python 函数不能放到 GPU 上跑。** C++ 的 `elemwise(func, M)` 接收的是编译期的 C++ lambda，Python 函数没法变成 GPU kernel。所以 Python 层只能用已经暴露的运算拼：`exp`、`log`、`tanh`、`sin`、`cos`、`square`、`sqrt`、加减乘除、`@`、`sum` 等。代价是多几次 kernel 和几个临时矩阵。
- **有 Python 调用开销。** 每层每步前向和反向各要跨一次 Python，层里的每个运算也要跨一次。由 5 到 10 个大矩阵运算组成的层，开销可以忽略；要做几十次小运算的层，应改用第 3 种做法。
- **反向要自己推导。** 所以要提供 `nn.gradcheck(layer, x)`，用有限差分对比解析梯度，思路和 `testAllLayerAPI` 一样。

### 7.2 自动求导（面向初学者）

**2026-11-09 决定：先不做。** 以下是讨论中的设想，留作以后参考。

讨论中途改过：最初说自定义层自己写 `backward`。明确了低门槛的目标之后，认为这对初学者门槛太高，所以**在 Python 层加一个简单的自动求导**：
- 每个 Juzhen 运算执行时记下自己的反向函数，调用 `loss.backward()` 时按相反顺序执行；
- Juzhen 的运算粒度都比较大，一次就是整个矩阵乘法或逐元素运算，所以用 Python 记录计算图的开销不大；
- 初学者只写前向就能定义新层；
- 内置层仍然用 C++ 里手写好的反向，保证速度。自动求导只用在用户自己写的部分。

### 7.3 C++ 自定义层

```cpp
template <class D>
class SiLULinear : public Layer<D> {
    // 像 LinearLayer 一样重写 eval / backward / jvp / parameters
};
```

在绑定代码里加一行 `nb::class_<SiLULinear<D>, Layer<D>>(m, "SiLULinear")`，再重新编译扩展模块。这样能用 `elemwise` 写任意逐元素函数，也能写自己的 CUDA 或 HIP kernel，Python 那边的用法和内置层一样。

最简单的做法是把新层直接加进 Juzhen 的源码树，和内置层一起编译。不建议做成单独的扩展模块：两个扩展模块之间要共享 `Matrix`、`Layer` 这些 C++ 类型，对 ABI 和编译选项的一致性要求很高，容易出问题。

## 8. 数据加载

C++ 现有的 `ml/dataloader.hpp`：
- 从磁盘上的 `.matrix` 文件按顺序流式读取，一次读一个 batch，不打乱；
- 标签是 1×N 的整数；
- 读到结尾自动回到开头，最后一个 batch 可能不满。

Python 提供两种加载器。

**方案一：包装 C++ 的 DataLoader，从磁盘流式读取。** 适合内存放不下的数据集。

```python
from juzhen.data import MatrixFileLoader

train = MatrixFileLoader("datasets/MNIST", "train", batch_size=32, drop_last=True)
test  = MatrixFileLoader("datasets/MNIST", "test",  batch_size=10000)

for epoch in range(5):
    for X_i, y_i in train:                     # 一个 epoch 结束时迭代自然停止
        X_i = X_i.to("gpu")
        Y_i = nn.one_hot(y_i, 10).to("gpu")    # 整数标签 → one-hot
        ...
```

和 C++ 版相比有两处调整：
- 改成 Python 的迭代器语义。C++ 的 `next_batch()` 会无限循环，Python 里 `for ... in loader` 每次遍历一个 epoch。
- 加了 `drop_last=True`，不用像 `demo_cnn_mnist` 那样手动跳过不满的 batch。

**方案二：数据先载入内存，每个 epoch 随机打乱。** MNIST 这类小数据集用这个更好。

```python
from juzhen.data import DataLoader

train = DataLoader(X, Y, batch_size=32, shuffle=True, drop_last=True, device="gpu")
for epoch in range(5):
    for X_i, Y_i in train:                     # 已经在 GPU 上
        ...
```

打乱在主机上完成：每个 epoch 先生成一个随机排列，按排列从主机矩阵里按下标取出一个 batch，再复制到 GPU。主机上的 `Matrix<float>` 本来就能按下标取，所以这个方案在 ROCm 上也能用，绕开了 ROCm 没有按下标取列的问题。

三点设计考虑：
1. **数据要先进 Juzhen 的内存池。** numpy 数组在构造时一次性复制进来，之后每个 batch 都在 Juzhen 内部切分。直接用 PyTorch 的 DataLoader 也行，只是每个 batch 都要经 numpy 复制进 Juzhen 一次。
2. **后台预取要等释放 GIL 之后。** 在那之前可以先提供 `prefetch` 参数：GPU 计算本身是异步的，提前把下一个 batch 复制过去，仍然能重叠一部分时间。CPU 后端就没法重叠了。
3. **训练时最后一个不满的 batch 默认丢掉。** 层在构造时固定了 `batch_size`，所以训练时 `drop_last` 默认为 `True` 最安全。评估时不用丢，前向会按输入的列数自动调整。

另外，讨论中提过一个纯 numpy 的小模块：读写 `.matrix` 文件和 checkpoint，不依赖扩展模块，方便训练后在 Python 里分析和画图。`tests/benchmarkCpuGpu.py` 和 `tests/testElementwiseReduceTorch.py` 里各有一份 `read_matrix`，可以合并进来。

## 9. 性能影响

大部分计算都在 C++ 里，影响集中在下面几处，按影响从大到小排：

1. **强制同步，最容易踩的坑。** `.numpy()`、`.item()`、`print(C)` 和读单个元素，都要等 GPU 把排队的活全做完。典型的坏写法是每一步都 `print(loss.item())`，应该每隔 N 步读一次。这一点 C++ 里也一样，只是 Python 用户更习惯随手打印。
2. **每次调用的固定开销，只在小矩阵上明显。** 每个 Python 运算大约亚微秒到 1 微秒（估计，未实测）。大矩阵和 GPU 后端可以忽略；CPU 后端加很小的矩阵加大量零碎的逐元素运算时会占大头。PyTorch eager 模式也是每个算子经过一次 Python 分派，所以只要计算量放在层里，就不会比 PyTorch 差。
3. **少了右值复用。** C++ 的 `exp(Matrix&&)` 会在原 buffer 上计算，Python 里所有变量都是左值，做不到。计算量和显存读写量基本不变，多出来的是分配和峰值内存：最多多一两个临时矩阵。在意的话，可以加 `A.exp_()`、`A += B` 这类原地操作，或者给常用组合提供融合函数。
4. **跨边界复制。** 每个 batch 从 numpy 进来都要复制，这在 C++ 里本来也要做。要避免的是多复制一次：numpy 应该直接复制到 GPU，不要先进主机上的 `Matrix<float>` 再复制。
5. **GIL。** 第一版全程持有 GIL，C++ 计算时 Python 的其他线程跑不起来。CPU 后端影响明显，GPU 后端影响小。
6. **`Sequential` 返回拷贝。** 每次前向多一次输出矩阵大小的设备内复制，和一层 GEMM 比很小。

一个训练步在 Python 里只有五次左右调用，全部固定开销加起来大约 10 微秒。GEMM、卷积、Transformer 的前向和反向、JVP、AdamW 更新都在 C++ 里，和直接写 C++ 一样快。

## 10. 构建与发布

- **用 nanobind 加 scikit-build-core**，复用现有的 CMakeLists。加一个 `JUZHEN_PYTHON=ON` 选项，调用 `nanobind_add_module`。选 nanobind 而不是 pybind11，是因为它更小、编译更快、每次调用的开销更低，还能用稳定 ABI，一个 wheel 就能覆盖多个 Python 版本（3.12 起）。
- **GPU 后端的编译是主要技术风险。** Juzhen 头文件里有近 20 处设备端 lambda（`__GPU_CPU__` / `__device__`），分布在 `juzhen.hpp`、`layer.hpp`、`loss.hpp`、`util.cuh`、`optimizer_cuda.cuh`。所以在 CUDA 构建里，绑定文件必须是 `.cu`，由 nvcc 编译；ROCm 也一样要用 hipcc。nanobind 头文件在 nvcc 下偶尔会有 C++17 兼容问题。退路是加一层非模板的门面（pimpl）：在 `.cu` 里实现一组普通函数，对外只用不透明句柄，绑定文件本身用普通编译器编译。CPU 版和 Metal 版没有这个问题。
- **按后端分别发包**，例如 `juzhen`（CPU 加 Metal）、`juzhen-cuda12`、`juzhen-rocm`。`jz.backend()` 返回当前后端。先只发 `juzhen`。
- **Windows 要把 OpenBLAS 的 DLL 打进 wheel。**
- **CUDA wheel** 要处理 CUDA runtime 和 cuDNN 的依赖，还要符合 manylinux 规范，不能把 `libcuda` 打包进去。
- **C++ 的输出要转发给 Python。** C++ 往 `std::cout` 写的内容在 Jupyter 里看不到，要像 iOS 的 `CallbackBuf` 那样转到 Python 的 `sys.stdout`。
- **测试用 pytest**，和 numpy、torch 对比数值，可以参照仓库里 `testTransformerTorch.py` 这类对比 torch 的测试。同一个随机种子下，Python 版的结果应该和 C++ 版逐位相同或非常接近。
- **基准测试**，第 1 步做完后加：逐元素运算按矩阵大小扫一遍、一个 MNIST 训练步、和 C++ 原版对比。把 Python 版接进现有的 `benchmark_pytorch_compare`，和 PyTorch 放在一起比。

## 11. 分阶段

| 阶段 | 内容 | 难度 | 主要风险 |
|---|---|---|---|
| 第 0 步（**已完成**） | `ERROR_OUT` 改抛异常；`juzhen::init()` / `shutdown()`；`Memory<D>::release_cached()`。完成后现有 C++ 测试在 CPU、CUDA、ROCm 上照样全部通过 | 低 | 析构路径里不能抛异常（已处理） |
| 第 1 步（**已实现**） | `Matrix`、逐元素运算、numpy 互转、pytest | 中 | 用 nvcc / hipcc 编译 nanobind |
| 第 2 步（**已实现**） | `nn`、`optim`、损失函数、checkpoint，用 MNIST 跑通端到端 | 中 | 层和 `Sequential` 的生命周期 |
| 第 3 步 | 打包成 wheel 发布（**必须做**，先发 CPU 版和 Metal 版）；计算时释放 GIL，另加一把 Juzhen 自己的锁 | 高 | 跨平台、跨 GPU 后端打包 |

**验证是瓶颈。** 每加一个绑定，都要在每个后端上验证。所以先把 CPU 版做好，再逐个加 GPU 后端。

## 12. 已定

- 2026-11-09：**自动求导（7.2）先不做。**
- 2026-11-09：**代码放在本仓库的 `python/` 目录下。**

## 13. 第 1、2 步的实现（2026-11-09）

代码在 `python/`：`src/core.cu` 是 nanobind 绑定，`juzhen/` 是 Python 包，`tests/` 是 pytest，`examples/` 是 C++ 示例的 Python 版。`pip install ./python` 构建，默认 CPU，`-C cmake.define.<后端>=ON` 选 GPU 后端。

和上面计划不同的地方：
- `nn.Dense` 默认激活是 tanh，和 C++ 的 `Layer<D>` 一致；输出层用 `nn.Linear`。所以 4.1 的例子改成了 `jz.Linear(10)`。
- 层对象只是描述，`Sequential` 知道输入维度后才在设备上创建 C++ 层，所以 `jz.Dense(256)` 可以不写输入维度。
- `jz.save_checkpoint(net, path, epoch=, step=)` 不再传优化器：Adam 状态本来就存在层里。
- 多了 `Parameter.assign()`，用来自定义初始化，测试里的数值梯度检查也靠它。
- `C[i, j]` 返回 Python 的 float；其他下标始终返回二维矩阵（`X[0]` 是 1×d）。
- 一维数组导入成 N×1（每个样本一个值），四维图像数组 N×C×H×W 展平成 N×CHW，也就是卷积层要的布局。
- 自定义 Python 层（7.1）还没做。

实现中发现的几点：
- nanobind 默认给扩展加 `/Os`（按体积优化），CPU 后端的逐元素循环因此慢了约 20%，已用 `NOMINSIZE` 关掉。
- Windows 上 Python 不在 PATH 里找扩展依赖的 DLL，构建时把 GPU 运行库目录写进 `juzhen/_build_config.py`，导入时用 `os.add_dll_directory` 加上。
- Windows 上的 ROCm：进程退出时如果 GPU 上还有没做完的活，HIP 运行库会卡死。`juzhen` 在 `atexit` 里先 `synchronize()`。

## 14. 补齐 demo 需要的功能（2026-11-09）

对照 `examples/` 里的 C++ demo，Python 接口缺这些，已补上：
- `jz.max`、`jz.min`、`jz.abs`、`jz.argmax`、`jz.topk`、`jz.softmax`、`jz.logsumexp`，按 axis 计算，在设备上算（Metal 上 `reduce`/`elemwise` 仍经过主机）。
- `nn.masked_softmax_cross_entropy`（离散扩散 demo 要用）。
- `nn.one_hot` 改成在设备上生成：只把 N 个标签拷过去，C++ demo 里各自手写的 one-hot kernel 就不需要了。
- `jz.hstack` / `jz.vstack` 改用库里的实现，不再经过 numpy 和主机。
- `Sequential.state_dict()` / `load_state_dict()`：参数按名字存成 numpy 数组，可以放进层相同、批大小不同的网络（Transformer 训练用批 64，生成用批 1）。
- 网络的输入如果带转置标记（比如 `hstack` 或矩阵乘积的结果），先整理成普通列主序再交给层。Conv、LayerNorm、Transformer 直接读存储，以前这种输入会算错。

已改写的 demo：`cnn_mnist.py`、`rectified_flow.py`、`transformer_charlm.py`、`knn.py`（另有 `fit_sin.py`）。终端界面（FTXUI）换成打印。还没改写：两个离散扩散、TinyStories、word2vec、tsfm、arithmetic、CIFAR rectified flow，以及 6 个网页版。

发现的问题：
- 默认初始化（权重 N(0, 0.001²)，和 C++ 一样）对深的 ReLU 网络太小：rectified flow 的 4 层网络输出几乎恒为零，能不能训练起来看运气（4090 上卡住，ROCm 上碰巧能训练）。`rectified_flow.py` 改用 He 初始化。C++ 的 `demo_rectified` 用同样的默认初始化，也有这个风险。
- C++ 的 `demo_transformer.cu` 和 `demo_rectified.cu` 只为 CUDA 和 Metal 选 GPU 类型，ROCm 构建下其实跑在 CPU 上。Python 版没有这个问题。
