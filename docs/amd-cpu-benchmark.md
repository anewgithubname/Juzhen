# Juzhen CPU / AMD GPU benchmark

## 代码与复现

- `tests/benchmarkCpuGpu.cu`：同一份 Juzhen FP32 源码，在 CPU-only 和 ROCm 构建中分别生成 `benchmarkCpuGpu`。
- `tests/benchmarkCpuGpu.py`：轮换 CPU/GPU 执行顺序，扫描 CPU 线程设置，比较输出和训练后的参数，保存原始日志、矩阵和 JSON 汇总。

从已配置好编译器、HIP 和 Python 环境的终端运行：

```powershell
cmake --build build_health_cpu --target benchmarkCpuGpu
cmake --build build_health_rocm --target benchmarkCpuGpu
python tests/benchmarkCpuGpu.py --cpu build_health_cpu/benchmarkCpuGpu.exe --gpu build_health_rocm/benchmarkCpuGpu.exe --output res/cpu_gpu_benchmark --rounds 3 --iterations 20 --warmup 5 --cpu-threads default,1,4,16
```

Python 比较程序需要 NumPy。Windows 上需将 HIP DLL 目录加入 PATH；GPU 进程需能够访问设备。程序遇到构建/运行失败、非有限输出或数值对照失败时会报错，不输出成功的加速结论。

## 测量口径

- 相同整数 PRNG 生成输入、方向向量和权重，替换两种后端的随机初始化；LayerNorm 的 gamma=1、beta=0。
- 数据与模型预先放入对应设备。统计操作调用到完成的墙钟时间；每个 GPU 样本等待设备完成。计时包含 Juzhen 操作内部的临时分配，不包含初始化、初始数据传输和校验输出拷回 CPU。
- 每个工作负载预热 5 次，测量 20 次。独立进程重复 3 轮，取各轮平均时间的中位数。
- CPU 测试默认、1、4、16 个 OpenBLAS 线程。逐工作负载选取这些设置中最快的中位数作为主要基线；同时保存与默认设置的比较。这个基线是已测配置中的最好值，不声称穷尽所有 CPU 调优方案。
- 训练步骤包含前向、输入/参数梯度和 Adam 更新；每次进程执行 25 次更新。JVP 分别测已有前向缓存的调用，以及前向加 JVP。
- 每轮每个 CPU 设置均与 GPU 比较 36 个矩阵，包括 GEMM 结果、Transformer 输出、JVP、训练输入梯度和全部 13 组更新后的参数。相对 L2 误差阈值为 1e-4。

## 2026-09-07 实测

设备：Ryzen AI MAX+ 395、Radeon 8060S（gfx1151）；Windows、ROCm 7.1。两个构建均为 Release：CPU 使用 GCC/MinGW 和仓库自带 OpenBLAS，GPU 使用 AMD Clang/HIP 和 hipBLAS。因此下面衡量的是本仓库当前实现与构建的实际速度，不是脱离软件实现的硬件峰值。

OpenBLAS 默认启用 32 线程，报告核心名称为 Cooperlake。实际选择的最快线程数见下表。

Transformer 配置均为因果注意力、4 个头：

| 配置 | d_model / d_k / d_ff | seq / batch | 每步 tokens |
|---|---|---|---:|
| small | 128 / 128 / 512 | 64 / 2 | 128 |
| large | 256 / 256 / 1024 | 128 / 4 | 512 |

| 工作负载 | 最快 CPU 线程数 | CPU ms | AMD GPU ms | CPU / GPU |
|---|---:|---:|---:|---:|
| GEMM 512×512 | 16 | 0.460 | 0.186 | 2.47× |
| GEMM 1024×1024 | 16 | 2.730 | 0.969 | 2.82× |
| small 前向 | 1 | 1.253 | 0.362 | 3.46× |
| small 缓存 JVP | 1 | 0.534 | 0.562 | 0.95× |
| small 前向 + JVP | 1 | 1.729 | 0.836 | 2.07× |
| small 训练 | 1 | 2.375 | 0.964 | 2.46× |
| large 前向 | 4 | 10.887 | 0.912 | 11.94× |
| large 缓存 JVP | 4 | 6.198 | 0.865 | 7.17× |
| large 前向 + JVP | 4 | 18.865 | 1.695 | 11.13× |
| large 训练 | 4 | 22.899 | 2.359 | 9.71× |

所有数值对照通过；12 组对照（每组 36 个矩阵）中，最大相对 L2 误差为 1.21e-6。原始数据位于 `res/cpu_gpu_benchmark/results.json`，逐轮日志和输出矩阵保存在其同级子目录。

GPU 没有统一的固定加速倍数：本次较大 Transformer 工作负载获得约 7–12 倍加速；小型缓存 JVP 与优化线程数后的 CPU 基本持平，GPU 约慢 5%。相对默认 32 线程 CPU，小型训练为 6.40×、大型训练为 17.29×；这些数字受 CPU 线程配置影响很大，不宜代替主表的调优后比较。
