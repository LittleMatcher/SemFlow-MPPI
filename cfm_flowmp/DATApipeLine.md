# CFM FlowMP 项目数据流和使用方案

## 一、项目概述

CFM FlowMP 是一个基于**条件流匹配（Conditional Flow Matching）**的轨迹规划框架，使用 Transformer 架构学习从专家演示中生成平滑、物理一致的轨迹。

### 核心组件

- **模型架构**：基于 Transformer 的条件向量场预测网络
- **训练方法**：Flow Matching 插值路径和速度场回归
- **推理方法**：RK4 ODE 积分从噪声生成轨迹
- **后处理**：B-spline 平滑保证物理一致性

---

## 二、数据流详解

### 2.1 训练阶段数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                     训练数据流 (Training Flow)                   │
└─────────────────────────────────────────────────────────────────┘

1. 数据加载 (Data Loading)
   │
   ├─ TrajectoryDataset / SyntheticTrajectoryDataset
   │  └─ 输入: 轨迹数据文件 (.npz, .npy, .h5, .pkl)
   │     ├─ positions: [N, T, D] 位置轨迹
   │     ├─ velocities: [N, T, D] 速度轨迹 (可选)
   │     └─ accelerations: [N, T, D] 加速度轨迹 (可选)
   │
   ├─ 数据预处理
   │  ├─ 计算导数 (如果未提供): 使用有限差分
   │  ├─ 归一化 (可选): 计算均值和标准差
   │  └─ 提取条件: start_pos, goal_pos, start_vel
   │
   └─ 输出: DataLoader 批次
      └─ 每个样本包含:
         - positions: [T, D]
         - velocities: [T, D]
         - accelerations: [T, D]
         - start_pos: [D]
         - goal_pos: [D]
         - start_vel: [D]

2. Flow Matching 插值 (Flow Interpolation)
   │
   ├─ FlowInterpolator.interpolate_trajectory()
   │  │
   │  ├─ 采样时间: t ~ Uniform(0, 1) [B]
   │  ├─ 采样噪声: ε ~ N(0, I) [B, T, D]
   │  │
   │  └─ 构建插值状态:
   │     ├─ q_t = t * q_1 + (1-t) * ε_q        (位置插值)
   │     ├─ q_dot_t = t * q_dot_1 + (1-t) * ε_q_dot  (速度插值)
   │     └─ q_ddot_t = t * q_ddot_1 + (1-t) * ε_q_ddot (加速度插值)
   │
   └─ 计算目标向量场:
      ├─ u_target = (q_1 - q_t) / (1-t)      (位置速度场)
      ├─ v_target = (q_dot_1 - q_dot_t) / (1-t)  (速度加速度场)
      └─ w_target = (q_ddot_1 - q_ddot_t) / (1-t) (加速度急动场)

3. 模型前向传播 (Model Forward)
   │
   ├─ FlowMPTransformer
   │  │
   │  ├─ 输入投影: [B, T, 6] → [B, T, hidden_dim]
   │  │  └─ x_t = concat([q_t, q_dot_t, q_ddot_t])
   │  │
   │  ├─ 时间嵌入: t → [B, cond_dim]
   │  │  └─ GaussianFourierProjection
   │  │
   │  ├─ 条件编码: c → [B, cond_dim]
   │  │  └─ ConditionEncoder(start_pos, goal_pos, start_vel)
   │  │
   │  ├─ 组合嵌入: [B, cond_dim] = time_emb + cond_emb
   │  │
   │  ├─ Transformer Blocks (×L层)
   │  │  └─ 每层包含:
   │  │     ├─ AdaLN(cond) → Multi-Head Self-Attention
   │  │     └─ AdaLN(cond) → Feed-Forward Network
   │  │
   │  └─ 输出头: [B, T, hidden_dim] → [B, T, 6]
   │     └─ 预测: [u_pred, v_pred, w_pred]
   │
   └─ 输出: 预测向量场 [B, T, 6]

4. 损失计算 (Loss Computation)
   │
   ├─ FlowMatchingLoss
   │  │
   │  ├─ 分离预测和目标:
   │  │  ├─ u_pred, v_pred, w_pred (从模型输出)
   │  │  └─ u_target, v_target, w_target (从插值结果)
   │  │
   │  └─ 加权 MSE 损失:
   │     L = λ_vel * ||u_pred - u_target||²
   │       + λ_acc * ||v_pred - v_target||²
   │       + λ_jerk * ||w_pred - w_target||²
   │
   └─ 反向传播和优化
      └─ Adam 优化器更新模型参数
```

### 2.2 推理阶段数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                   推理数据流 (Inference Flow)                     │
└─────────────────────────────────────────────────────────────────┘

1. 初始化 (Initialization)
   │
   ├─ 加载训练好的模型
   ├─ 设置生成器配置 (GeneratorConfig)
   │  ├─ solver_type: "rk4"
   │  ├─ num_steps: 20 (或自定义时间表)
   │  └─ use_bspline_smoothing: True
   │
   └─ 输入条件:
      ├─ start_pos: [B, D] 起始位置
      ├─ goal_pos: [B, D] 目标位置
      └─ start_vel: [B, D] 起始速度 (可选)

2. 噪声采样 (Noise Sampling)
   │
   └─ x_0 ~ N(0, I) [B, T, 6]
      └─ 包含: [pos_noise, vel_noise, acc_noise]

3. ODE 求解 (ODE Integration)
   │
   ├─ TrajectoryGenerator.generate()
   │  │
   │  ├─ 创建速度函数: velocity_fn(x_t, t)
   │  │  └─ 包装模型: model(x_t, t, start_pos, goal_pos, start_vel)
   │  │
   │  ├─ RK4Solver.solve()
   │  │  │
   │  │  ├─ 时间调度:
   │  │  │  ├─ 均匀调度: t = [0, dt, 2*dt, ..., 1]
   │  │  │  └─ 8步非均匀调度: [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
   │  │  │
   │  │  └─ RK4 积分步骤 (对每个时间步):
   │  │     k1 = velocity_fn(x_t, t)
   │  │     k2 = velocity_fn(x_t + dt/2 * k1, t + dt/2)
   │  │     k3 = velocity_fn(x_t + dt/2 * k2, t + dt/2)
   │  │     k4 = velocity_fn(x_t + dt * k3, t + dt)
   │  │     x_{t+dt} = x_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
   │  │
   │  └─ 输出: x_1 [B, T, 6] (t=1时的状态)
   │
   └─ 提取组件:
      ├─ positions_raw = x_1[..., :D]
      ├─ velocities_raw = x_1[..., D:2*D]
      └─ accelerations_raw = x_1[..., 2*D:3*D]

4. 后处理 (Post-processing)
   │
   ├─ BSplineSmoother.smooth()
   │  │
   │  ├─ 对每个轨迹:
   │  │  ├─ 拟合 B-spline 曲线 (degree=3)
   │  │  ├─ 重新采样到原始时间点
   │  │  └─ 计算平滑后的速度和加速度
   │  │
   │  └─ 输出: 平滑后的轨迹
   │
   └─ 最终输出:
      ├─ positions: [B, T, D] 平滑位置轨迹
      ├─ velocities: [B, T, D] 平滑速度轨迹
      └─ accelerations: [B, T, D] 平滑加速度轨迹
```

### 2.3 数据格式说明

#### 输入数据格式

```python
# 文件格式: .npz, .npy, .h5, .pkl
{
    'positions': np.ndarray,      # [N, T, D] 位置轨迹
    'velocities': np.ndarray,     # [N, T, D] 速度轨迹 (可选)
    'accelerations': np.ndarray,  # [N, T, D] 加速度轨迹 (可选)
}

# 如果未提供 velocities/accelerations，会自动从 positions 计算
```

#### 模型输入格式

```python
# 训练时
{
    'x_t': torch.Tensor,        # [B, T, 6] 插值状态
    't': torch.Tensor,          # [B] 流时间
    'start_pos': torch.Tensor,  # [B, D] 起始位置
    'goal_pos': torch.Tensor,   # [B, D] 目标位置
    'start_vel': torch.Tensor,  # [B, D] 起始速度 (可选)
}

# 推理时
{
    'x_t': torch.Tensor,        # [B, T, 6] 当前状态
    't': torch.Tensor,         # [B] 当前时间
    'start_pos': torch.Tensor, # [B, D] 起始位置
    'goal_pos': torch.Tensor,  # [B, D] 目标位置
    'start_vel': torch.Tensor, # [B, D] 起始速度 (可选)
}
```

#### 模型输出格式

```python
{
    'output': torch.Tensor,  # [B, T, 6] 预测向量场
    # 分解为:
    # - u_pred: [B, T, D] 位置速度场
    # - v_pred: [B, T, D] 速度加速度场
    # - w_pred: [B, T, D] 加速度急动场
}
```

---

## 三、使用方案

### 3.1 快速开始

#### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或: venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 使用合成数据训练

```bash
# 基础训练
python train.py --synthetic --num_trajectories 5000 --epochs 50

# 使用不同轨迹类型
python train.py --synthetic --trajectory_type bezier --epochs 100
python train.py --synthetic --trajectory_type polynomial --epochs 100
python train.py --synthetic --trajectory_type sine --epochs 100
```

#### 使用自定义数据训练

```bash
# 从文件加载数据
python train.py \
    --data_path /path/to/trajectories.npz \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4

# 自定义模型配置
python train.py \
    --data_path /path/to/data.npz \
    --model_variant large \
    --hidden_dim 512 \
    --num_layers 8 \
    --lr 5e-5
```

#### 生成轨迹

```bash
# 基础生成
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --start 0,0 \
    --goal 2,2 \
    --visualize

# 生成多个样本
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --start 0,0 \
    --goal 2,2 \
    --num_samples 10 \
    --output generated_trajectories.npz

# 使用8步非均匀调度 (推荐，快速推理)
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --start 0,0 \
    --goal 2,2 \
    --solver rk4 \
    --use_8step_schedule

# 自定义ODE求解器设置
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --start 0,0 \
    --goal 2,2 \
    --solver rk4 \
    --num_steps 50
```

### 3.2 Python API 使用

#### 训练 API

```python
import torch
from cfm_flowmp.models import create_flowmp_transformer
from cfm_flowmp.training import CFMTrainer, TrainerConfig, FlowMatchingConfig
from cfm_flowmp.data import SyntheticTrajectoryDataset, create_dataloader

# 1. 创建模型
model = create_flowmp_transformer(
    variant="base",      # "small", "base", "large"
    state_dim=2,         # 状态维度 (2D: x, y)
    max_seq_len=64,      # 轨迹长度
)

# 2. 创建数据集
dataset = SyntheticTrajectoryDataset(
    num_trajectories=5000,
    seq_len=64,
    trajectory_type="bezier",
)

# 3. 创建数据加载器
train_loader = create_dataloader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
)

# 4. 配置训练
flow_config = FlowMatchingConfig(
    state_dim=2,
    lambda_vel=1.0,      # 速度场损失权重
    lambda_acc=1.0,      # 加速度场损失权重
    lambda_jerk=1.0,     # 急动场损失权重
)

trainer_config = TrainerConfig(
    num_epochs=100,
    learning_rate=1e-4,
    device="cuda",
    use_amp=True,
    flow_config=flow_config,
)

# 5. 创建训练器并训练
trainer = CFMTrainer(
    model=model,
    config=trainer_config,
    train_dataloader=train_loader,
    val_dataloader=None,  # 可选验证集
)

trainer.train()
```

#### 推理 API

```python
import torch
from cfm_flowmp.models import create_flowmp_transformer
from cfm_flowmp.inference import TrajectoryGenerator, GeneratorConfig
from cfm_flowmp.inference.generator import create_8step_schedule

# 1. 加载模型
model = create_flowmp_transformer(variant="base")
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. 创建生成器 (推荐使用8步非均匀调度)
gen_config = GeneratorConfig(
    solver_type="rk4",
    time_schedule=create_8step_schedule(),  # [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
    use_bspline_smoothing=True,
    seq_len=64,
    state_dim=2,
)

generator = TrajectoryGenerator(model, gen_config)

# 3. 生成轨迹
start_pos = torch.tensor([[0.0, 0.0]])  # [1, 2]
goal_pos = torch.tensor([[2.0, 2.0]])  # [1, 2]

with torch.no_grad():
    result = generator.generate(
        start_pos=start_pos,
        goal_pos=goal_pos,
        num_samples=5,  # 生成5个样本
    )

# 4. 获取结果
positions = result['positions']      # [5, 64, 2]
velocities = result['velocities']    # [5, 64, 2]
accelerations = result['accelerations']  # [5, 64, 2]
```

#### 自定义数据加载

```python
from cfm_flowmp.data import TrajectoryDataset, create_dataloader

# 从文件加载
dataset = TrajectoryDataset(
    data_path="trajectories.npz",
    normalize=True,  # 自动归一化
    compute_derivatives=True,  # 自动计算导数
)

# 或直接传入数组
import numpy as np
positions = np.random.randn(1000, 64, 2)  # [N, T, D]

dataset = TrajectoryDataset(
    positions=positions,
    normalize=True,
)

# 创建数据加载器
train_loader = create_dataloader(
    dataset,
    batch_size=64,
    shuffle=True,
)
```

### 3.3 完整示例

参考 `example.py` 文件，它展示了完整的训练和推理流程：

```bash
python example.py
```

该示例包括：
1. 创建合成数据集
2. 创建模型
3. 训练模型（短时间演示）
4. 生成轨迹
5. 可视化结果

### 3.4 高级用法

#### 自定义时间调度

```python
from cfm_flowmp.inference import GeneratorConfig

# 自定义非均匀时间调度
custom_schedule = [0.0, 0.5, 0.75, 0.9, 0.95, 0.98, 1.0]

gen_config = GeneratorConfig(
    solver_type="rk4",
    time_schedule=custom_schedule,
    use_bspline_smoothing=True,
)
```

#### 带引导的生成

```python
# 定义障碍物函数
def obstacle_fn(positions):
    """返回障碍物成本（越小越好）"""
    # 示例: 避免中心区域
    center = torch.tensor([1.0, 1.0])
    dist = (positions - center).norm(dim=-1)
    cost = torch.exp(-dist)  # 距离中心越近，成本越高
    return cost.sum()

# 使用引导生成
result = generator.generate_with_guidance(
    start_pos=start_pos,
    goal_pos=goal_pos,
    guidance_scale=2.0,  # 引导强度
    obstacle_fn=obstacle_fn,
)
```

#### 批量生成

```python
# 准备多个条件
conditions = [
    {'start_pos': torch.tensor([0.0, 0.0]), 'goal_pos': torch.tensor([2.0, 2.0])},
    {'start_pos': torch.tensor([1.0, 0.0]), 'goal_pos': torch.tensor([1.0, 2.0])},
    {'start_pos': torch.tensor([0.0, 1.0]), 'goal_pos': torch.tensor([2.0, 1.0])},
]

# 批量生成
results = generator.generate_batch(
    conditions,
    batch_size=32,
)
```

---

## 四、关键参数说明

### 4.1 模型变体

| 变体 | Hidden Dim | Layers | Heads | 参数量 |
|------|-----------|--------|-------|--------|
| small | 128 | 4 | 4 | ~3M |
| base | 256 | 6 | 8 | ~12M |
| large | 512 | 8 | 16 | ~50M |

### 4.2 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 1e-4 | Adam 学习率 |
| weight_decay | 0.01 | L2 正则化 |
| warmup_steps | 1000 | 学习率预热步数 |
| lambda_vel | 1.0 | 速度场损失权重 |
| lambda_acc | 1.0 | 加速度场损失权重 |
| lambda_jerk | 1.0 | 急动场损失权重 |
| batch_size | 64 | 批次大小 |
| gradient_accumulation | 1 | 梯度累积步数 |

### 4.3 推理超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| solver_type | "rk4" | ODE 求解器类型 ("euler", "midpoint", "rk4", "rk45") |
| num_steps | 20 | 均匀时间步数 |
| time_schedule | None | 自定义时间调度 (覆盖 num_steps) |
| use_bspline_smoothing | True | 是否使用 B-spline 平滑 |
| bspline_degree | 3 | B-spline 度数 |

### 4.4 推荐配置

#### 快速训练（开发/测试）
```python
variant = "small"
epochs = 10
batch_size = 32
lr = 1e-3
```

#### 标准训练（生产）
```python
variant = "base"
epochs = 100
batch_size = 64
lr = 1e-

# CFM FlowMP 项目数据流和使用方案

## 目录
1. [项目概述](#项目概述)
2. [数据流详解](#数据流详解)
3. [使用方案](#使用方案)
4. [核心组件说明](#核心组件说明)

---

## 项目概述

CFM FlowMP 是一个基于**条件流匹配（Conditional Flow Matching）**的轨迹规划框架，使用 Transformer 架构学习从专家演示中生成平滑、物理一致的轨迹。

### 核心特性
- **Transformer 架构**：基于 AdaLN 条件化的 Transformer
- **联合预测**：同时预测速度、加速度和加加速度（jerk）场
- **多种 ODE 求解器**：Euler、Midpoint、RK4、自适应 RK45
- **B-spline 平滑**：确保物理一致性
- **混合精度训练**：支持梯度累积和 EMA

---

## 数据流详解

### 1. 训练阶段数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    训练数据流                                │
└─────────────────────────────────────────────────────────────┘

[1] 数据加载 (dataset.py)
    │
    ├─ TrajectoryDataset: 从文件加载专家轨迹
    │   ├─ 支持格式: .npy, .npz, .h5, .pkl
    │   ├─ 输入: positions [N, T, D]
    │   ├─ 自动计算: velocities, accelerations (如果缺失)
    │   └─ 输出: 归一化的轨迹数据
    │
    └─ SyntheticTrajectoryDataset: 生成合成轨迹
        ├─ Bezier 曲线
        ├─ 多项式轨迹
        └─ 正弦运动

[2] 数据预处理
    │
    ├─ 归一化: (x - mean) / std
    ├─ 提取条件: start_pos, goal_pos, start_vel
    └─ 批次化: DataLoader

[3] Flow Matching 训练循环 (flow_matching.py)
    │
    ├─ 步骤 1: 采样流时间 t ~ Uniform(0, 1)
    │
    ├─ 步骤 2: 采样噪声
    │   ├─ ε_q ~ N(0, I)      [B, T, D]  位置噪声
    │   ├─ ε_q_dot ~ N(0, I)  [B, T, D]  速度噪声
    │   └─ ε_q_ddot ~ N(0, I) [B, T, D]  加速度噪声
    │
    ├─ 步骤 3: 状态插值 (线性插值路径)
    │   ├─ q_t = t * q_1 + (1-t) * ε_q
    │   ├─ q_dot_t = t * q_dot_1 + (1-t) * ε_q_dot
    │   └─ q_ddot_t = t * q_ddot_1 + (1-t) * ε_q_ddot
    │
    ├─ 步骤 4: 计算目标向量场
    │   ├─ u_target = (q_1 - q_t) / (1-t)      # 位置速度场
    │   ├─ v_target = (q_dot_1 - q_dot_t) / (1-t)  # 速度加速度场
    │   └─ w_target = (q_ddot_1 - q_ddot_t) / (1-t) # 加速度加加速度场
    │
    ├─ 步骤 5: 模型前向传播 (transformer.py)
    │   ├─ 输入: x_t = [q_t, q_dot_t, q_dot_t] [B, T, 6]
    │   ├─ 条件: start_pos, goal_pos, start_vel
    │   ├─ 时间嵌入: Gaussian Fourier Projection
    │   ├─ Transformer Blocks (AdaLN 条件化)
    │   └─ 输出: [u_pred, v_pred, w_pred] [B, T, 6]
    │
    └─ 步骤 6: 损失计算
        ├─ L_vel = ||u_pred - u_target||²
        ├─ L_acc = ||v_pred - v_target||²
        ├─ L_jerk = ||w_pred - w_target||²
        └─ L_total = λ_vel*L_vel + λ_acc*L_acc + λ_jerk*L_jerk
```

### 2. 推理阶段数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    推理数据流                                │
└─────────────────────────────────────────────────────────────┘

[1] 初始化
    │
    ├─ 输入条件: start_pos [B, D], goal_pos [B, D], start_vel [B, D]
    └─ 采样初始噪声: x_0 ~ N(0, I) [B, T, 6]

[2] ODE 求解 (ode_solver.py)
    │
    ├─ 时间调度:
    │   ├─ 均匀调度: t = [0, dt, 2*dt, ..., 1]
    │   └─ 8步非均匀调度: [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
    │       (早期大步长，后期小步长，保留细节)
    │
    ├─ RK4 积分步骤:
    │   ├─ k1 = model(x_t, t, conditions)
    │   ├─ k2 = model(x_t + dt/2*k1, t + dt/2, conditions)
    │   ├─ k3 = model(x_t + dt/2*k2, t + dt/2, conditions)
    │   ├─ k4 = model(x_t + dt*k3, t + dt, conditions)
    │   └─ x_{t+dt} = x_t + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    │
    └─ 从 t=0 积分到 t=1 → x_1

[3] 提取轨迹
    │
    ├─ positions = x_1[..., :D]      [B, T, D]
    ├─ velocities = x_1[..., D:2*D] [B, T, D]
    └─ accelerations = x_1[..., 2*D:3*D] [B, T, D]

[4] 后处理平滑 (generator.py)
    │
    └─ B-spline 平滑 (可选)
        ├─ 拟合 B-spline 曲线
        ├─ 重新采样平滑轨迹
        └─ 计算平滑后的速度和加速度
```

### 3. 模型架构数据流

```
输入: x_t [B, T, 6] (位置+速度+加速度)
      t [B] (流时间)
      c (start_pos, goal_pos, start_vel)

    ↓
┌─────────────────────────────────────┐
│   Input Projection                  │
│   [B, T, 6] → [B, T, hidden_dim]   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Positional Encoding               │
│   (Sinusoidal/Learned)              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Gaussian Fourier Time Embedding  │
│   t → [B, hidden_dim]               │
└──────────┬──────────────────────────┘
           │
           │    ┌──────────────────────┐
           │    │  Condition Encoder   │
           │    │  c → [B, hidden_dim] │
           │    └──────────┬───────────┘
           │               │
           ↓               ↓
    ┌──────────────────────────┐
    │  Combined Conditioning    │
    │  [time_emb + cond_emb]    │
    └────────────┬──────────────┘
                 │
    ┌─────────────┴─────────────┐
    │                           │
    ↓                           │
┌───────────────────────────────┴─────┐
│  Transformer Blocks (×L)             │
│  ┌────────────────────────────────┐ │
│  │  AdaLN (conditioned on t, c)  │ │
│  │         ↓                      │ │
│  │  Multi-Head Self-Attention     │ │
│  │         ↓                      │ │
│  │  AdaLN (conditioned on t, c)  │ │
│  │         ↓                      │ │
│  │  Feed-Forward Network          │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Final AdaLN                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Output Head                       │
│   [B, T, hidden_dim] → [B, T, 6]    │
│   (u: 速度场, v: 加速度场, w: 加加速度场)│
└─────────────────────────────────────┘
```

---

## 使用方案

### 1. 快速开始

#### 1.1 安装依赖

```bash
pip install -r requirements.txt
```

#### 1.2 使用合成数据训练

```bash
# 使用 Bezier 曲线轨迹训练
python train.py --synthetic --num_trajectories 5000 --epochs 50

# 使用多项式轨迹
python train.py --synthetic --trajectory_type polynomial --epochs 100

# 使用正弦轨迹
python train.py --synthetic --trajectory_type sine --epochs 100
```

#### 1.3 使用自定义数据训练

```bash
# 基本训练
python train.py --data_path /path/to/trajectories.npz --epochs 100

# 自定义模型配置
python train.py \
    --data_path /path/to/data.npz \
    --model_variant large \
    --hidden_dim 512 \
    --num_layers 8 \
    --lr 5e-5 \
    --batch_size 32
```

#### 1.4 生成轨迹

```bash
# 生成单个轨迹
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --start 0,0 \
    --goal 2,2 \
    --visualize

# 生成多个样本
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --start 0,0 \
    --goal 2,2 \
    --num_samples 10 \
    --output generated_trajectories.npz

# 使用 8 步非均匀调度（推荐，快速推理）
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --start 0,0 \
    --goal 2,2 \
    --solver rk4 \
    --use_8step_schedule
```

### 2. Python API 使用

#### 2.1 训练 API

```python
import torch
from cfm_flowmp.models import create_flowmp_transformer
from cfm_flowmp.training import CFMTrainer, TrainerConfig, FlowMatchingConfig
from cfm_flowmp.data import SyntheticTrajectoryDataset, create_dataloader

# 创建模型
model = create_flowmp_transformer(
    variant="base",
    state_dim=2,
    max_seq_len=64,
)

# 创建数据集
dataset = SyntheticTrajectoryDataset(
    num_trajectories=5000,
    seq_len=64,
    trajectory_type="bezier",
)

train_loader = create_dataloader(dataset, batch_size=64)

# 配置训练
flow_config = FlowMatchingConfig(
    state_dim=2,
    lambda_vel=1.0,
    lambda_acc=1.0,
    lambda_jerk=1.0,
)

trainer_config = TrainerConfig(
    num_epochs=100,
    learning_rate=1e-4,
    device="cuda",
    flow_config=flow_config,
)

# 开始训练
trainer = CFMTrainer(model, trainer_config, train_loader)
trainer.train()
```

#### 2.2 推理 API

```python
import torch
from cfm_flowmp.models import create_flowmp_transformer
from cfm_flowmp.inference import TrajectoryGenerator, GeneratorConfig
from cfm_flowmp.inference.generator import create_8step_schedule

# 加载模型
model = create_flowmp_transformer(variant="base")
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 创建生成器（推荐使用 8 步非均匀调度）
gen_config = GeneratorConfig(
    solver_type="rk4",
    time_schedule=create_8step_schedule(),  # [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
    use_bspline_smoothing=True,
)
generator = TrajectoryGenerator(model, gen_config)

# 生成轨迹
start_pos = torch.tensor([[0.0, 0.0]])
goal_pos = torch.tensor([[2.0, 2.0]])

result = generator.generate(
    start_pos=start_pos,
    goal_pos=goal_pos,
    num_samples=5,
)

positions = result['positions']      # [5, 64, 2]
velocities = result['velocities']      # [5, 64, 2]
accelerations = result['accelerations']  # [5, 64, 2]
```

### 3. 数据格式要求

#### 3.1 输入数据格式

```python
# 方式 1: NumPy 数组
{
    'positions': np.ndarray,      # [N, T, D] 位置轨迹
    'velocities': np.ndarray,     # [N, T, D] 速度轨迹（可选）
    'accelerations': np.ndarray,  # [N, T, D] 加速度轨迹（可选）
}

# 方式 2: 直接传入数组
dataset = TrajectoryDataset(
    positions=positions_array,
    velocities=velocities_array,  # 可选
    accelerations=accelerations_array,  # 可选
    normalize=True,
)
```

#### 3.2 支持的文件格式

- `.npy`: 单个 NumPy 数组（仅位置）
- `.npz`: NumPy 压缩文件（多个数组）
- `.h5` / `.hdf5`: HDF5 文件
- `.pkl`: Pickle 文件

### 4. 配置参数说明

#### 4.1 模型变体

| 变体 | Hidden Dim | Layers | Heads | 参数量 |
|------|-----------|--------|-------|--------|
| small | 128 | 4 | 4 | ~3M |
| base | 256 | 6 | 8 | ~12M |
| large | 512 | 8 | 16 | ~50M |

#### 4.2 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 1e-4 | Adam 学习率 |
| weight_decay | 0.01 | L2 正则化 |
| warmup_steps | 1000 | 学习率预热步数 |
| lambda_acc | 1.0 | 加速度损失权重 |
| lambda_jerk | 1.0 | 加加速度损失权重 |
| num_steps (推理) | 20 | ODE 积分步数（均匀） |
| time_schedule (推理) | None | 自定义时间调度 |

---

## 核心组件说明

### 1. 数据模块 (`cfm_flowmp/data/`)

- **TrajectoryDataset**: 加载专家轨迹数据
  - 自动计算速度和加速度（如果缺失）
  - 支持数据归一化
  - 提取起始/目标位置和速度

- **SyntheticTrajectoryDataset**: 生成合成轨迹
  - Bezier 曲线
  - 多项式轨迹
  - 正弦运动

### 2. 训练模块 (`cfm_flowmp/training/`)

- **FlowInterpolator**: 构建插值路径
  - 线性插值：`x_t = t * x_1 + (1-t) * ε`
  - 计算目标向量场

- **FlowMatchingLoss**: 流匹配损失
  - 速度场损失 (u)
  - 加速度场损失 (v)
  - 加加速度场损失 (w)

- **CFMTrainer**: 训练循环
  - 混合精度训练
  - 梯度累积
  - EMA 权重平均
  - 检查点保存

### 3. 推理模块 (`cfm_flowmp/inference/`)

- **RK4Solver**: RK4 ODE 求解器
  - 支持均匀和非均匀时间调度
  - 8 步非均匀调度（推荐）

- **TrajectoryGenerator**: 轨迹生成管道
  - 噪声采样
  - ODE 积分
  - B-spline 平滑

- **BSplineSmoother**: B-spline 平滑
  - 确保物理一致性
  - 减少 ODE 积分误差

### 4. 模型模块 (`cfm_flowmp/models/`)

- **FlowMPTransformer**: 主模型
  - Transformer 编码器
  - AdaLN 条件化
  - 多输出头（速度、加速度、加加速度）

- **Embeddings**: 嵌入层
  - Gaussian Fourier 时间嵌入
  - 条件编码器
  - AdaLN / AdaLN-Zero

---

## 完整示例

### 端到端训练和推理示例

```python
# example.py 展示了完整的训练和推理流程
python example.py
```

该示例包括：
1. 创建合成数据集
2. 训练模型（短时间演示）
3. 生成轨迹
4. 可视化和评估

---

## 算法细节

### Flow Matching 训练算法

对于每个训练步骤：

1. **采样**专家轨迹 `(q_1, q_dot_1, q_ddot_1)` 从数据集
2. **采样**流时间 `t ~ Uniform(0, 1)`
3. **采样**噪声 `ε_q, ε_q_dot, ε_q_ddot ~ N(0, I)`
4. **插值**状态：
   - `q_t = t · q_1 + (1-t) · ε_q`
   - `q_dot_t = t · q_dot_1 + (1-t) · ε_q_dot`
   - `q_ddot_t = t · q_ddot_1 + (1-t) · ε_q_ddot`
5. **计算目标场**（使用插值状态）：
   - `u_target = (q_1 - q_t) / (1-t)`
   - `v_target = (q_dot_1 - q_dot_t) / (1-t)`
   - `w_target = (q_ddot_1 - q_ddot_t) / (1-t)`
6. **前向传播**：`(u_pred, v_pred, w_pred) = Model(x_t, t, c)`
7. **计算损失**：
   ```
   L = ||u_pred - u_target||² 
       + λ_acc * ||v_pred - v_target||² 
       + λ_jerk * ||w_pred - w_target||²
   ```

### ODE 积分（推理）

1. **初始化** `x_0 ~ N(0, I)`
2. **积分**从 `t=0` 到 `t=1` 使用 RK4：
   ```
   for t in time_schedule:
       k1 = mod