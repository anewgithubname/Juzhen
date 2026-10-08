# AMD Transformer 第一轮优化

本轮实现保留 FP32 和原有模型语义，不改变学习率或数值容差。

## 实现

- AMD 训练 benchmark 使用 HIP event 并等待完成，同时报告包含提交、同步开销的 `wall_mean_ms`。
- `sampled_device_delta_mb` 是设备空闲显存相对初始值的采样差值；不是进程内精确峰值，可能受其他程序影响。
- HIP Softmax 前向融合因果掩码、最大值、指数、归约和归一化；反向融合归约和梯度计算。
- 每个 query 行由 256 个线程协作，使用共享内存归约，不假设 wavefront 大小，支持非整块长度。
- 注意力前向和反向使用每个 head 的 strided-batched GEMM，一次处理所有 batch；直接访问 Q/K/V 和复用的 attention 缓冲区，避免逐 batch 切片复制。
- `benchmarkTrainingStepGeneric` 用同一计时程序保留原始注意力路径，用于 A/B 测量。

## 正确性

PyTorch dump 升级为 `JZTFDMP2`，包含因果模式和三步训练记录。测试比较输出、输入梯度，以及全部 13 组参数和 Adam 一阶、二阶矩。额外配置包括：

| 配置 | d_model / d_k / d_ff | seq / batch / heads | causal |
|---|---|---|---|
| default | 16 / 16 / 32 | 7 / 2 / 4 | true |
| single | 16 / 16 / 32 | 1 / 3 / 4 | true |
| bidirectional | 12 / 15 / 25 | 17 / 3 / 3 | false |
| long | 16 / 16 / 32 | 257 / 2 / 4 | true |

运行已有和新增检查（需要 Python、NumPy、PyTorch；GPU build 需要 HIP）：

```powershell
cmake --build build_health_rocm --target testTransformer testTransformerRef testTransformerTorchDump benchmarkTrainingStep benchmarkTrainingStepGeneric
ctest --test-dir build_health_rocm --output-on-failure -R '^(test11|test13|test14|test15|transformer_.*|benchmark_training_step|benchmark_pytorch_compare)$'
```

在 Windows 上需从配置好编译器、HIP 和 Python PATH 的环境运行。GPU 运行需允许访问设备。PyTorch 在本次环境使用 CPU float64 作为数值参考，不构成 PyTorch GPU 性能对比。

## 实测

2026-09-06，Windows、ROCm 7.1、Radeon 8060S（gfx1151）。固定 benchmark 配置 d128、dk128、ff512、seq64、batch2、heads4。各进程预热 20 步、测量 100 步；原始和优化版交替运行 5 轮，以下是每轮均值的中位数：

| 指标 | 原始注意力 | 优化注意力 | 加速 |
|---|---:|---:|---:|
| GPU event 时间 | 1.147 ms | 0.869 ms | 1.32× |
| 含同步的实际每步时间 | 1.904 ms | 1.299 ms | 1.47× |

复测时设置 `JUZHEN_BENCH_WARMUP=20`、`JUZHEN_BENCH_ITERS=100`，交替运行两个 benchmark 可执行文件。收益仅代表此设备、配置和训练步骤；不代表长序列模型、完整训练收敛或所有 AMD GPU。

以上数字仅对应第一轮注意力优化。

## 第二轮：LayerNorm、Adam 和偏置融合

新增 HIP LayerNorm 前向与输入梯度、Adam 状态更新、偏置广播内核。LayerNorm 保留中心化方差、epsilon=1e-5 和既有缓存约定；Adam 保留超参数和步数递增语义。转置布局继续走通用路径。LayerNorm 的 gamma/beta 梯度归约仍使用原有实现。

同一设备和配置，以第一轮优化后的程序为基线，交替运行 5 轮，每轮预热 20 步、测量 100 步：

| 指标 | 第一轮版本 | 第二轮版本 | 加速 |
|---|---:|---:|---:|
| GPU event 时间 | 0.850 ms | 0.815 ms | 1.04× |
| 含同步的实际每步时间 | 1.320 ms | 0.935 ms | 1.41× |

收益主要体现在减少操作提交、临时矩阵和同步相关开销。不同轮次的系统负载会变化，因此应使用本次交替测试的基线比较，不应直接拼接不同实验的加速倍数。

验证：12 项 AMD Transformer 检查全部通过（包含全部 13 组参数和 Adam 状态的三步对照），另有 CNN 训练对照、U-Net 学习测试通过。CPU 编译与基本 Transformer 回归也通过；未在 CUDA/Metal 硬件上重测。

后续可评估 LayerNorm 协作归约、gamma/beta 梯度归约融合、减少每个 head 的调用次数和长序列内存占用。

## 第三轮：Transformer JVP

ROCm 的注意力方向导数现在使用每个 head 四次 strided-batched GEMM，并复用 Softmax 导数内核和现有 scratch 缓冲区。两项乘积法则项分别通过 beta=0、beta=1 累积，避免逐 batch 切片及额外中间矩阵。前向缓存保持不变，允许在同一个输入上连续计算不同方向的 JVP。LayerNorm JVP 仍使用通用表达式；本轮没有修改卷积 JVP。

在上述 Radeon 8060S 和 d128/dk128/ff512/seq64/batch2/heads4 配置下，交替运行 5 轮，每轮预热 20 次、测量 100 次，取各轮均值中位数（包含 GPU 同步）：

| 模式 | 通用 JVP | 批量 JVP | 加速 |
|---|---:|---:|---:|
| 已有前向缓存，仅计算 JVP | 0.814231 ms | 0.501146 ms | 1.62× |
| 前向 + JVP | 1.066050 ms | 0.805666 ms | 1.32× |

复测目标为 `benchmarkRocmJVPGeneric` 和 `benchmarkRocmJVP`。前者通过 `JUZHEN_ROCM_GENERIC_JVP` 保留通用路径，两者使用相同前向实现和计时方式。未设置环境变量时均预热 20 次、测量 100 次。

AMD 和 CPU 的 `jvp_correctness` 均通过。AMD 对照包含 1/5/257-token 的因果与非因果 Transformer，以及每个配置的两个连续缓存方向；最大相对误差 2.85e-7。有限差分、与反向传播的一致性和 MLP/卷积/转置卷积的现有检查也通过。测试容差未放宽。
