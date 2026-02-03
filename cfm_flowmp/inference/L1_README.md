# L1 反应控制层 (The Reactive Legs)

## 概述

L1 反应控制层是三层架构中的最底层，负责在 L2 提供的先验附近进行局部优化，确保硬约束满足。

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

## 使用方法

### 基本使用

```python
from cfm_flowmp.inference import L1ReactiveController, L1Config

# 1. 创建配置
config = L1Config(
    n_samples_per_mode=100,  # 每个模式的采样数
    tube_radius=0.5,  # 管道半径
    tube_covariance=0.1,  # 管道协方差
    w_semantic=1.0,  # 语义场权重
    w_tube=10.0,  # 管道约束权重
    w_energy=0.1,  # 能量项权重
    temperature=1.0,  # 逆温度
)

# 2. 创建语义场函数（可选）
def semantic_fn(positions):
    # 计算语义场代价
    # positions: [B, T, D]
    # 返回: [B]
    return torch.zeros(positions.shape[0])

# 3. 创建 L1 控制器
l1_controller = L1ReactiveController(
    config=config,
    semantic_fn=semantic_fn,
)

# 4. 从 L2 输出初始化
l2_output = {
    'positions': anchor_positions,  # [K, T, D] - K 个锚点
    'velocities': anchor_velocities,  # [K, T, D] (可选)
}
l1_controller.initialize_from_l2_output(l2_output)

# 5. 执行优化
result = l1_controller.optimize(n_iterations=10)

# 6. 获取最优控制（用于闭环回流）
optimal_control = result['optimal_control']  # [T, D]
```

### 与 L2 层集成

```python
from cfm_flowmp.inference import TrajectoryGenerator, L1ReactiveController

# L2 层生成锚点
generator = TrajectoryGenerator(model, gen_config)
l2_result = generator.generate(
    start_pos=start_pos,
    goal_pos=goal_pos,
    num_samples=K,  # 生成 K 个锚点
)

# 准备 L2 输出
l2_output = {
    'positions': l2_result['positions'],  # [K, T, D]
    'velocities': l2_result['velocities'],
}

# L1 层优化
l1_controller.initialize_from_l2_output(l2_output)
l1_result = l1_controller.optimize(n_iterations=10)

# 获取最优控制（用于下一帧）
next_control = l1_result['optimal_control']
```

## 配置参数

### L1Config

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_samples_per_mode` | int | 100 | 每个模式的采样数 |
| `n_control_points` | int | 10 | 控制点数量 |
| `time_horizon` | float | 5.0 | 时间范围（秒） |
| `n_timesteps` | int | 50 | 时间步数 |
| `tube_radius` | float | 0.5 | 管道半径 $r_{tube}$ |
| `tube_covariance` | float | 0.1 | 管道协方差 $\Sigma_{tube}$ 的标准差 |
| `w_semantic` | float | 1.0 | 语义场权重 $w_1$ |
| `w_tube` | float | 10.0 | 管道约束权重 $w_2$ |
| `w_energy` | float | 0.1 | 能量项权重 $w_3$ |
| `temperature` | float | 1.0 | 逆温度 $\lambda$ |
| `energy_matrix_scale` | float | 1.0 | 能量矩阵 $R$ 的缩放 |

## 代价函数组件

### 1. SemanticFieldCost (语义场代价)

计算 L3 语义场代价。可以是：
- 障碍物场（SDF）
- 语义分割场
- 其他环境感知场

### 2. TubeConstraintCost (管道约束代价)

确保轨迹保持在锚点附近的"管道"内：

$$J_{tube} = \begin{cases} 
0 & \text{if } ||\mathbf{x} - \mathbf{x}_{anchor}|| < r_{tube} \\ 
\infty & \text{otherwise} 
\end{cases}$$

### 3. EnergyCost (能量代价)

惩罚控制输入的能量：

$$J_{energy} = \mathbf{u}^T \mathbf{R} \mathbf{u}$$

## 设计特点

1. **并行优化**: K 个 MPPI 优化器并行运行，每个在对应的锚点附近优化
2. **管道约束**: 硬约束确保轨迹不会偏离锚点太远
3. **双重代价**: 结合语义场和能量项，平衡安全性和效率
4. **闭环回流**: 最优控制可以反馈到 L2 层，实现实时适应
5. **在线热启动（On-Policy 特性）**: 使用上一帧的最优控制作为先验，实现策略延续

## 在线热启动（Warm-Start）机制

### 原理

L1 层支持 **On-Policy** 特性，类似于强化学习中的策略延续：

1. **策略延续**: 在 t 时刻规划出的最优轨迹 $\tau^*_t$，在 $t+1$ 时刻成为强有力的先验
2. **移位操作**: 将 $u^*_t$ 向前移动一步（丢弃第一个动作，末尾补零或外推），得到 $\tilde{u}_{t+1}$
3. **CFM 反向注入**: 在 $t+1$ 时刻生成时，将 $\tilde{u}_{t+1}$ 加噪后作为 CFM 的初始状态 $z_T$，而不是从纯高斯噪声 $\mathcal{N}(0, I)$ 开始

### 效果

- **时间一致性**: CFM 不再是每次"从零思考"，而是"接着上一步的想法继续思考"
- **策略平滑**: 相邻时间步的策略具有连续性，减少抖动
- **更快的收敛**: 利用先验信息，减少探索空间

### 使用方法

```python
# 创建配置（启用热启动）
l1_config = L1Config(
    use_warm_start=True,  # 启用热启动
    warm_start_noise_scale=0.1,  # 热启动时的噪声缩放
    shift_padding_mode="extrapolate",  # 移位填充模式: "zero" 或 "extrapolate"
    # ... 其他参数
)

# 创建控制器
l1_controller = L1ReactiveController(config=l1_config)

# 在线规划循环
for frame in range(n_frames):
    # 获取热启动状态（如果有上一帧）
    warm_start_state = l1_controller.get_warm_start_state()
    
    # L2 层生成（使用热启动状态）
    l2_result = generator.generate(
        start_pos=start_pos,
        goal_pos=goal_pos,
        warm_start_state=warm_start_state,  # CFM 反向注入
    )
    
    # L1 层优化
    l1_controller.initialize_from_l2_output(l2_result)
    l1_result = l1_controller.optimize(n_iterations=10)
    
    # 获取最优控制（会自动保存热启动状态）
    optimal_control = l1_controller.get_next_control(l2_result)
```

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_warm_start` | bool | True | 是否使用热启动 |
| `warm_start_noise_scale` | float | 0.1 | 热启动时的噪声缩放（较小的值保持更多先验信息） |
| `shift_padding_mode` | str | "zero" | 移位填充模式："zero"（补零）或 "extrapolate"（外推） |

## 注意事项

1. **控制序列到轨迹的转换**: 当前实现简化处理，假设控制序列直接对应位置序列。实际应用中可能需要通过动力学模型积分。

2. **语义场函数**: 需要根据实际应用提供语义场函数。可以是障碍物场、语义分割场等。

3. **计算效率**: 并行 MPPI 优化可能计算量较大，建议根据实际需求调整采样数和迭代次数。

4. **参数调优**: 管道半径、协方差、权重等参数需要根据具体应用场景调优。

## 示例

完整示例请参考 `example_l1_l2_integration.py`。

