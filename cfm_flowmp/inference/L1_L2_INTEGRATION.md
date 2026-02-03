# L1-L2 集成说明

本文档说明 L1 反应控制层如何接收 L2 (CFM-FlowMP) 的输出，并进行 MPPI 优化。

## 概述

L1 层实现了三层架构中的最底层，负责在 L2 提供的先验附近进行局部优化，确保硬约束满足。

## 数学原理

### 1. 拓扑并行采样 (Topology-Concurrent Sampling)

L2 输出 $K$ 个锚点（Anchors）$\{\bar{\mathbf{u}}^1, \dots, \bar{\mathbf{u}}^K\}$。

L1 实例化 $K$ 个并行 MPPI 优化器。对于第 $m$ 个模式，控制序列采样分布为：

$$\mathbf{u}_i^m \sim \mathcal{N}(\bar{\mathbf{u}}^m, \Sigma_{tube})$$

其中 $\Sigma_{tube}$ 是协方差矩阵，比标准 MPPI 小，限制在"管道"内探索。

### 2. 能量-安全双重代价 (Dual Objective)

总代价函数 $J(\mathbf{u})$：

$$J(\mathbf{u}) = \underbrace{w_1 J_{sem}(\mathbf{x})}_{\text{L3 语义场}} + \underbrace{w_2 J_{tube}(\mathbf{x}, \bar{\mathbf{u}}^m)}_{\text{管道约束}} + \underbrace{w_3 \mathbf{u}^T \mathbf{R} \mathbf{u}}_{\text{能量项}}$$

**管道约束** $J_{tube}$:

$$J_{tube} = \begin{cases} 
0 & \text{if } ||\mathbf{x} - \mathbf{x}_{anchor}|| < r_{tube} \\ 
\infty & \text{otherwise} 
\end{cases}$$

### 3. 最优控制更新 (Update Law)

计算最优控制量（MPPI 经典公式）：

$$\mathbf{u}^*_{k} = \frac{\sum_{i,m} \exp(-\frac{1}{\lambda} J(\mathbf{u}_i^m)) \cdot \mathbf{u}_i^m}{\sum_{i,m} \exp(-\frac{1}{\lambda} J(\mathbf{u}_i^m))}$$

**闭环回流**: 这个 $\mathbf{u}^*_{k}$ 将在下一帧 $k+1$ 成为 L2 的 $\mathbf{z}_{init}$。

## L2 输出格式

L2 层（`TrajectoryGenerator.generate()`）的输出格式：

```python
l2_output = {
    'positions': torch.Tensor,      # [K, T, D] 或 [B*N, T, D]
    'velocities': torch.Tensor,     # [K, T, D]
    'accelerations': torch.Tensor,  # [K, T, D]
}
```

其中：
- `K = B * num_samples`：锚点数量（当 `num_samples > 1` 时）
- `T`：时间步数（轨迹长度）
- `D`：状态维度（通常是 2 或 3）

**注意**：
- 当 `num_samples = 1` 时，格式为 `[B, T, D]`
- 当 `num_samples > 1` 时，格式为 `[B*num_samples, T, D]`，即 `[K, T, D]`

## 使用方法

### 基本使用

```python
from cfm_flowmp.inference import TrajectoryGenerator, GeneratorConfig
from cfm_flowmp.inference import L1ReactiveController, L1Config

# 1. 创建 L2 生成器
generator_config = GeneratorConfig(
    state_dim=2,
    seq_len=64,
    num_samples=5,  # 生成 5 个锚点
)
generator = TrajectoryGenerator(model, generator_config)

# 2. 创建 L1 控制器
l1_config = L1Config(
    n_samples_per_mode=100,  # 每个模式的采样数
    n_timesteps=64,  # 与 L2 输出一致
    tube_radius=0.5,  # 管道半径
    tube_covariance=0.1,  # 管道协方差
    w_semantic=1.0,  # 语义场权重
    w_tube=10.0,  # 管道约束权重
    w_energy=0.1,  # 能量项权重
    temperature=1.0,  # 逆温度 λ
)
l1_controller = L1ReactiveController(config=l1_config)

# 3. 在线规划循环
for frame in range(n_frames):
    # L2 生成锚点
    l2_output = generator.generate(
        start_pos=start_pos,
        goal_pos=goal_pos,
        num_samples=5,  # 生成 5 个锚点
    )
    
    # L1 优化
    l1_controller.initialize_from_l2_output(l2_output)
    l1_result = l1_controller.optimize(n_iterations=10)
    
    # 获取最优控制（用于闭环回流）
    optimal_control = l1_controller.get_next_control(
        l2_output,
        n_iterations=10,
    )
    
    # 执行第一个控制动作，更新状态
    # ...
```

### 完整示例

参见 `example_l1_l2_integration.py` 文件。

## 实现细节

### L1 如何接收 L2 输出

`L1ReactiveController.initialize_from_l2_output()` 方法：

1. **提取锚点位置**：从 L2 输出字典中提取 `'positions'` 键
2. **处理维度**：
   - 如果是 `[B, K, T, D]`，取第一个批次得到 `[K, T, D]`
   - 如果是 `[K, T, D]` 或 `[B*N, T, D]`，直接使用
3. **创建并行优化器**：为每个锚点创建一个 `ParallelMPPIOptimizer`

### MPPI 优化流程

1. **采样**：对每个模式，在锚点附近采样控制序列
   - `u_i^m ~ N(ū^m, Σ_tube)`
2. **评估代价**：计算每个样本的总代价
   - `J(u_i^m) = w1 * J_sem + w2 * J_tube + w3 * u^T R u`
3. **计算权重**：使用重要性采样权重
   - `w_i^m = exp(-J(u_i^m) / λ) / Σ_{j,n} exp(-J(u_j^n) / λ)`
4. **更新控制**：加权平均得到最优控制
   - `u*_k = Σ_{i,m} w_i^m * u_i^m`

### 闭环回流

L1 的最优控制 `u*_k` 通过以下方式回流到 L2：

1. **移位操作**：将控制序列向前移动一步
   - 丢弃第一个动作（已执行）
   - 末尾补零或外推
2. **准备热启动状态**：将移位后的控制转换为完整状态（pos + vel + acc）
3. **加噪**：添加小量噪声以保持探索
4. **CFM 反向注入**：作为 CFM 的初始状态 `z_T`，而不是从纯高斯噪声开始

## 配置参数

### L1Config 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_samples_per_mode` | int | 100 | 每个模式的采样数 |
| `n_timesteps` | int | 50 | 时间步数（应与 L2 输出一致） |
| `tube_radius` | float | 0.5 | 管道半径 r_tube |
| `tube_covariance` | float | 0.1 | 管道协方差 Σ_tube 的标准差 |
| `w_semantic` | float | 1.0 | L3 语义场权重 w1 |
| `w_tube` | float | 10.0 | 管道约束权重 w2 |
| `w_energy` | float | 0.1 | 能量项权重 w3 |
| `temperature` | float | 1.0 | 逆温度 λ |
| `use_warm_start` | bool | True | 是否使用热启动 |

## 注意事项

1. **时间步数一致性**：L1 的 `n_timesteps` 应与 L2 输出的时间步数一致
2. **语义场函数**：需要根据实际应用提供语义场函数（障碍物场、SDF 等）
3. **计算效率**：并行 MPPI 优化可能计算量较大，建议根据实际需求调整采样数和迭代次数
4. **参数调优**：管道半径、协方差、权重等参数需要根据具体应用场景调优

## 参考

- `l1_reactive_control.py`：L1 层实现
- `generator.py`：L2 层实现
- `example_l1_l2_integration.py`：完整示例

