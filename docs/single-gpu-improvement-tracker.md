# 单机单卡实验改进追踪

本文档用于追踪 Juzhen 在单机单卡实验场景下的可靠性、可复现性、性能和工程体验改进。

## 状态说明

| 状态 | 含义 |
|---|---|
| 待处理 | 尚未开始 |
| 进行中 | 已开始实现，但尚未达到验收标准 |
| 已完成 | 已实现并通过对应验收 |
| 暂缓 | 当前阶段不实施，并已记录原因 |

只有满足相应验收标准后，任务才能标记为“已完成”。任务状态变化时，应同时更新“最近更新”和进度汇总。

## 当前基线

- 最近更新：2026-07-12
- CPU clean rebuild：成功
- CUDA clean rebuild：成功
- CPU CTest：28 通过，0 失败，2 跳过
- CUDA CTest：26 通过，0 失败，4 跳过
- 合计：54 通过，0 失败，6 跳过，共 60 项
- 已知风险：CUDA 构建仍存在部分仅在其他后端使用的函数警告；已识别的返回引用生命周期警告已修复。

## 进度汇总

| 优先级 | 已完成 | 进行中 | 待处理 | 暂缓 | 总计 |
|---|---:|---:|---:|---:|---:|
| P0 | 3 | 0 | 2 | 0 | 5 |
| P1 | 0 | 0 | 9 | 0 | 9 |
| P2 | 3 | 0 | 4 | 0 | 7 |
| 合计 | 6 | 0 | 15 | 0 | 21 |

## P0：训练可靠性

| ID | 修改建议 | 状态 | 验收标准 | 备注 |
|---|---|---|---|---|
| REL-001 | 修复 `returning reference to local variable` 等生命周期警告 | 已完成 | CPU/CUDA clean rebuild 不再报告已识别的生命周期警告；相关路径有回归测试 | 五个缓存加载器改为按值返回；移除 Transformer forward 未使用变量 |
| REL-002 | 实现完整 checkpoint 保存与恢复 | 已完成 | 可恢复模型、optimizer、epoch/step、随机数状态和数据位置 | 所有层通过统一虚接口声明 checkpoint schema；Transformer 覆盖全部 13 个参数和 13 组 Adam；格式带版本、名称校验、尾部标记和原子替换 |
| REL-003 | 增加 checkpoint 一致性测试 | 已完成 | 连续训练与中断恢复训练在规定容差内得到一致结果 | `checkpoint_resume_consistency` 覆盖 CPU/CUDA 的 Linear 网络和 Transformer block，参数及 Adam 状态最大绝对误差均为 0 |
| REL-004 | 增加 loss、梯度和参数的 NaN/Inf 检测 | 待处理 | 异常发生时立即给出位置、step、张量信息，并保留诊断材料 | 检测应可配置关闭 |
| REL-005 | 增加长时间稳定性测试 | 待处理 | 单卡连续运行预定时长，无显存/主存持续增长，无异常退出 | 首个目标建议为 2 小时 |

## P1：可复现性与实验管理

| ID | 修改建议 | 状态 | 验收标准 | 备注 |
|---|---|---|---|---|
| REP-001 | 统一配置入口 | 待处理 | 核心 demo 使用统一命令行参数或配置文件，不再依赖零散环境变量 | 保留兼容层时需记录弃用计划 |
| REP-002 | 自动创建独立实验目录 | 待处理 | 每次运行独立保存配置、指标、环境信息、checkpoint 和样本 | 目录名应避免冲突 |
| REP-003 | 记录代码与运行环境 | 待处理 | 自动记录 Git commit、工作区状态、GPU、CUDA、cuDNN、编译器和构建选项 | 不记录密钥等敏感环境变量 |
| REP-004 | 统一随机种子管理 | 待处理 | CPU、CUDA、数据加载和数据增强均由一个实验 seed 派生 | 派生规则应固定并记录 |
| REP-005 | 提供 deterministic 模式 | 待处理 | 相同环境和配置下重复运行满足规定的一致性标准 | 文档说明性能影响 |

## P1：显存与训练效率

| ID | 修改建议 | 状态 | 验收标准 | 备注 |
|---|---|---|---|---|
| PERF-001 | 支持 FP16/BF16 混合精度与 loss scaling | 待处理 | 代表性模型数值稳定，吞吐或显存相对 FP32 有明确收益 | 分别记录 GPU 对 BF16 的支持情况 |
| PERF-002 | 支持梯度累积 | 待处理 | 累积结果与等效大 batch 在规定容差内一致 | 正确处理 loss 缩放和 optimizer step |
| PERF-003 | 统计峰值显存并改善 OOM 信息 | 待处理 | 实验摘要包含峰值显存；OOM 时报告建议调整项 | 至少提示 batch size 和梯度累积 |
| PERF-004 | 优化临时矩阵和 workspace 复用 | 待处理 | 基准显示分配次数或峰值显存下降，且数值测试通过 | 重点检查隐式深拷贝和 view/ownership 语义 |

## P2：性能基准与数据管线

| ID | 修改建议 | 状态 | 验收标准 | 备注 |
|---|---|---|---|---|
| BENCH-001 | 建立核心算子 benchmark | 已完成 | 覆盖 GEMM、Conv2D、Attention、LayerNorm 和 optimizer step | `benchmarkCoreOps`；CPU 使用 steady clock，CUDA 使用 event；支持配置 warm-up/iterations；CTest 标签为 `benchmark` |
| BENCH-002 | 建立端到端训练 benchmark | 已完成 | 报告吞吐、P50/P95 step 延迟和峰值显存 | `benchmarkTrainingStep` 固定 Transformer 配置和 seed，覆盖 forward/backward/Adam update；CUDA 使用 event 和 `cudaMemGetInfo` |
| BENCH-003 | 增加 PyTorch 对照 | 已完成 | 相同配置下报告数值误差、吞吐和显存差异 | `benchmarkPytorchCompare.py` 对照相同训练配置的延迟、吞吐和 CUDA 峰值显存；数值验证直接复用 `testTransformerTorch.py` |
| DATA-001 | 支持后台预取和 pinned memory | 待处理 | 数据准备与 GPU 计算可重叠，且吞吐基准有记录 | 需提供关闭开关 |
| DATA-002 | 支持异步 H2D 拷贝 | 待处理 | 使用独立 stream 或明确的流水线，并通过同步正确性测试 | 避免复用尚未传输完成的 host buffer |
| DATA-003 | 增加多线程 DataLoader | 待处理 | worker 数可配置，固定 seed 时可复现 | 覆盖 worker 退出和异常传播 |
| DATA-004 | 分别记录 data time 与 compute time | 待处理 | 训练日志和汇总能识别数据瓶颈与计算瓶颈 | 与统一指标系统结合 |

## P2：构建、测试与诊断

以下事项计入现有任务，不额外加入上方进度总数；完成时应在相关任务备注中链接实现或测试结果。

- 移除对 `/usr/local/cuda-12.6/bin/nvcc` 的硬编码，使用标准 CMake CUDA 编译器发现机制。
- 增加 `CMakePresets.json`，至少提供 CPU Debug、CUDA Debug、CUDA Release 和 sanitizer 配置。
- 将 CTest 的 `test0`、`test1` 等名称替换为描述性名称，并增加 `unit`、`cuda`、`slow`、`parity`、`training` 标签。
- CPU 构建接入 AddressSanitizer 和 UndefinedBehaviorSanitizer。
- 选择小规模 CUDA 测试接入 `compute-sanitizer`。
- CI 持续执行 CPU 测试；条件允许时定期执行 CUDA 测试。

## 建议里程碑

### M1：可靠的单卡长时间训练

- [x] REL-001 生命周期问题清理
- [x] REL-002 完整 checkpoint/resume（统一层接口，含 Transformer）
- [x] REL-003 checkpoint 一致性测试
- [ ] REL-004 NaN/Inf 诊断
- [ ] REL-005 长时间稳定性测试

### M2：可复现、可管理的实验

- [ ] REP-001 统一配置入口
- [ ] REP-002 独立实验目录
- [ ] REP-003 环境与代码版本记录
- [ ] REP-004 统一随机种子
- [ ] REP-005 deterministic 模式

### M3：高效且可比较的训练

- [ ] PERF-001 混合精度
- [ ] PERF-002 梯度累积
- [ ] PERF-003 显存统计和 OOM 诊断
- [x] BENCH-001 核心算子 benchmark
- [x] BENCH-002 端到端 benchmark
- [x] BENCH-003 PyTorch 对照

## 更新记录

| 日期 | 变更 |
|---|---|
| 2026-07-12 | 完成 REL-001–003：修复缓存加载生命周期警告；新增版本化训练 checkpoint；CPU/CUDA 连续训练与恢复训练达到零误差 |
| 2026-07-12 | 清理 BENCH-003 重复实现：删除重复的 dump 解析、权重装载和 parity 计算，改为调用现有 `testTransformerTorch.py` |
| 2026-07-12 | 完成 BENCH-002/003：增加 Transformer 端到端训练 step 延迟分布、吞吐和峰值显存测试，以及 CPU/CUDA PyTorch 性能与数值对照 |
| 2026-07-12 | 完成 BENCH-001：增加 GEMM、Conv2D forward、LayerNorm forward、Attention block forward 和 Adam update benchmark；CPU/CUDA CTest 验证通过 |
| 2026-07-12 | 建立追踪文档；记录 CPU/CUDA rebuild 和 CTest 基线；创建 P0–P2 初始任务列表 |
