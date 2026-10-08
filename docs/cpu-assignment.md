# CPU 线性指派求解器

`ml/assignment.hpp` 提供 `Juzhen::LinearAssignmentSolver`，使用最短增广路形式的 Hungarian 原始-对偶算法。支持有限 FP32 代价矩阵、最小化/最大化和矩形匹配；内部代价、势和松弛量使用 double。不是 ε 近似求解器，也不是 Jonker–Volgenant 实现。

```cpp
#include "ml/assignment.hpp"

Juzhen::LinearAssignmentSolver solver; // 可重复使用，复用内部工作区
auto result = solver.solve(cost);      // cost: Matrix<float>
auto maximum = solver.solve(cost, true);
// result.row_to_col[i]、result.col_to_row[j]：匹配索引
// 矩形问题中未匹配的顶点为 -1；result.cost 为原始代价之和。
```

输入可以是转置视图。每次求解把代价整理为行连续布局，不修改输入。空维度返回空匹配和零代价；NaN、正负无穷均拒绝。不支持通过无穷大表示禁止匹配的边。求解器不限制为 512 阶，但本次基准只测到 512。一个对象不能被多个线程同时调用；批量并行时每个线程使用独立对象。

AMD 程序也可调用此 CPU 求解器：先显式执行 `auto cpu_cost = gpu_cost.to_host();`，再调用 `solve(cpu_cost)`。匹配结果是 CPU 整数数组，GPU 传输时间不包含在下表中。

时间复杂度 `O(min(rows,cols)^2 * max(rows,cols))`；除行连续代价副本外，工作数组为线性空间。

## 正确性与运行

```powershell
cmake --build build_health_cpu --target testAssignment benchmarkAssignment
ctest --test-dir build_health_cpu --output-on-failure -R '^assignment_correctness$'
./build_health_cpu/benchmarkAssignment.exe
```

测试覆盖 784 个 1–7 阶/矩形随机 min/max 问题，每个都与独立穷举最优值比较，并验证转置视图、匹配索引和代价。另测 32/128/512 阶已知最优解、全零重复代价、工作区复用、空矩阵及非法输入。测试在 CPU 构建通过；AMD 构建中的求解仍在 CPU 上执行。

## 2026-09-07 基准

Ryzen AI MAX+ 395，Windows，GCC/MinGW Release（-O3）。求解器单线程，不调用 BLAS。每组生成 12 个输入，前两个预热，后十个计时；计时包含布局转换、输入验证、求解和返回结果的分配，不包含随机代价生成。

| n | 随机代价 | 一维平方距离 | 8 种整数重复代价 | 所有行相同（C[i,j]=j） |
|---:|---:|---:|---:|---:|
| 32 | 0.032 ms | 0.025 ms | 0.020 ms | 0.028 ms |
| 64 | 0.133 ms | 0.149 ms | 0.048 ms | 0.234 ms |
| 128 | 0.575 ms | 0.721 ms | 0.109 ms | 1.749 ms |
| 256 | 2.367 ms | 2.547 ms | 0.378 ms | 14.171 ms |
| 512 | 13.926 ms | 18.442 ms | 1.413 ms | 102.700 ms |

表中为十个样本的中位数。随机、距离和整数代价使用不同随机实例；所有行相同的家族每次结构相同，用于显示退化输入的代价。原始 mean/max 和 checksum 保存在 `build_health_cpu/assignment-benchmark.log`。

这是第一版性能基线，不代表 JV 或其他指派求解器的最佳性能。n=512 的输入结构能带来很大的耗时差异；如果实际任务集中于此规模，应使用业务代价矩阵再评测，必要时继续优化或对照 JV。
