# CFM FlowMP 项目数据流和使用方案

## 一、项目概述

CFM FlowMP 是一个基于**条件流匹配（Conditional Flow Matching）**的轨迹规划框架，作为三层架构的中间层（L2），连接上层VLM（L3）语义理解和下层MPPI（L1）局部优化。

### 三层架构定位

```
┌─────────────────────────────────────────────────────────────┐
│  L3: Vision-Language Model (VLM)                            │
│  └─ 语义理解、场景解析、生成语义代价地图                      │
└────────────────────┬────────────────────────────────────────┘
                     │ cost_map [B,C,H,W]
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  L2: Safety-Embedded CFM (本项目)                            │
│  └─ 条件流匹配生成多模态轨迹锚点 (N=64 samples)                │
│     - 输入: 语义代价地图 + 机器人状态 + 目标 + 控制风格权重      │
│     - 输出: N条带完整动力学的轨迹 [p, v, a]                     │
└────────────────────┬────────────────────────────────────────┘
                     │ trajectory_anchors [B×N, T, 2]
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  L1: Model Predictive Path Integral (MPPI)                  │
│  └─ 使用轨迹锚点进行局部优化和实时控制                          │
└─────────────────────────────────────────────────────────────┘
```

### L2层核心组件

- **模型架构**：FlowMPTransformer / FlowMPUNet1D 条件向量场预测
- **代价地图编码器**：CostMapEncoder 将语义地图编码为潜在表示
- **训练方法**：Flow Matching 插值路径和速度场回归
- **推理方法**：RK4 ODE 积分从噪声生成轨迹
- **多模态生成**：每个条件生成N=64条候选轨迹供MPPI选择
- **控制风格调节**：支持 [w_safety, w_energy, w_smooth] 权重动态调整行为

---

## 二、数据流详解

### 2.1 训练阶段数据流

#### 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     训练数据流 (Training Flow)                   │
└─────────────────────────────────────────────────────────────────┘

1. 数据加载与预处理 (Data Loading & Preprocessing)
   │
   ├─ TrajectoryDataset / SyntheticTrajectoryDataset
   │  ├─ 输入: 轨迹数据文件 (.npz, .npy, .h5, .pkl) 或合成轨迹
   │  │
   │  ├─ 数据维度:
   │  │  ├─ 原始数据: positions [N, T, D]
   │  │  │            where: N=轨迹数量, T=时间步数, D=状态维度
   │  │  ├─ 可选项: velocities [N, T, D]
   │  │  └─ 可选项: accelerations [N, T, D]
   │  │
   │  ├─ 预处理步骤:
   │  │  ├─ ① 计算导数 (如果未提供)
   │  │  │   └─ velocity = Δposition / Δtime (中心差分)
   │  │  │   └─ acceleration = Δvelocity / Δtime
   │  │  │
   │  │  ├─ ② 数据归一化 (可选)
   │  │  │   ├─ 计算统计量: mean_pos, std_pos, mean_vel, std_vel
   │  │  │   └─ 归一化: (data - mean) / std
   │  │  │
   │  │  └─ ③ 提取条件信息
   │  │      ├─ start_pos: 轨迹第一个时间步的位置 [T, D] → [D]
   │  │      ├─ goal_pos: 轨迹最后一个时间步的位置 [T, D] → [D]
   │  │      └─ start_vel: 轨迹第一个时间步的速度 [T, D] → [D]
   │  │
   │  └─ 输出格式 (每个样本):
   │     ├─ positions: [T, D]
   │     ├─ velocities: [T, D]
   │     ├─ accelerations: [T, D]
   │     ├─ start_pos: [D]
   │     ├─ goal_pos: [D]
   │     └─ start_vel: [D]
   │
   └─ DataLoader: 组织成批次
      ├─ batch_size: B (通常 32-128)
      ├─ 输出张量形状:
      │  ├─ positions_batch: [B, T, D]
      │  ├─ velocities_batch: [B, T, D]
      │  ├─ accelerations_batch: [B, T, D]
      │  ├─ start_pos_batch: [B, D]
      │  ├─ goal_pos_batch: [B, D]
      │  └─ start_vel_batch: [B, D]
      │
      └─ DataLoader 特性:
         ├─ shuffle=True (随机顺序训练)
         ├─ drop_last=True (丢弃最后不完整批次)
         └─ num_workers (多进程数据加载)

2. Flow Matching 插值与目标计算 (Flow Interpolation)
   │
   ├─ FlowInterpolator.interpolate_trajectory()
   │  │
   │  ├─ 【第一步】采样流时间
   │  │   └─ t ~ Uniform(0, 1) → [B] (每个样本一个时间值)
   │  │   └─ 意义: 在插值路径上的位置
   │  │
   │  ├─ 【第二步】采样高斯噪声
   │  │   ├─ ε_q ~ N(0, I) → [B, T, D] (位置噪声)
   │  │   ├─ ε_q_dot ~ N(0, I) → [B, T, D] (速度噪声)
   │  │   └─ ε_q_ddot ~ N(0, I) → [B, T, D] (加速度噪声)
   │  │
   │  ├─ 【第三步】线性插值状态（从噪声到专家轨迹）
   │  │   │
   │  │   ├─ 位置插值:
   │  │   │  q_t = t * q_1 + (1-t) * ε_q
   │  │   │  形状: [B, T, D]
   │  │   │  含义: t=0时为纯噪声，t=1时为专家轨迹
   │  │   │
   │  │   ├─ 速度插值:
   │  │   │  q_dot_t = t * q_dot_1 + (1-t) * ε_q_dot
   │  │   │  形状: [B, T, D]
   │  │   │
   │  │   └─ 加速度插值:
   │  │      q_ddot_t = t * q_ddot_1 + (1-t) * ε_q_ddot
   │  │      形状: [B, T, D]
   │  │
   │  ├─ 【第四步】合并插值状态
   │  │   └─ x_t = concat([q_t, q_dot_t, q_ddot_t])
   │  │   └─ 形状: [B, T, 6] (对于 D=2 的状态)
   │  │
   │  └─ 【第五步】计算目标向量场
   │     │
   │     ├─ 向量场的定义：速度场 u 描述了从噪声流向专家轨迹的方向
   │     │
   │     ├─ 位置速度场 (Velocity Field):
   │     │  u_target = (q_1 - q_t) / (1-t)
   │     │  形状: [B, T, D]
   │     │  含义: 实现位置从 q_t 到 q_1 的变化速度
   │     │  注意: 分母 (1-t) 是为了在 t→1 时归一化
   │     │
   │     ├─ 速度加速度场 (Acceleration Field):
   │     │  v_target = (q_dot_1 - q_dot_t) / (1-t)
   │     │  形状: [B, T, D]
   │     │  含义: 实现速度从 q_dot_t 到 q_dot_1 的变化速度
   │     │
   │     └─ 加速度急动场 (Jerk Field):
   │        w_target = (q_ddot_1 - q_ddot_t) / (1-t)
   │        形状: [B, T, D]
   │        含义: 实现加速度从 q_ddot_t 到 q_ddot_1 的变化速度
   │
   └─ 输出张量:
      ├─ x_t: [B, T, 6] 插值状态
      ├─ t: [B] 流时间
      ├─ u_target: [B, T, D] 位置速度场目标
      ├─ v_target: [B, T, D] 加速度场目标
      └─ w_target: [B, T, D] 急动场目标

3. 模型前向传播 (Model Forward Pass)
   │
   ├─ 输入准备:
   │  ├─ x_t: [B, T, 6]
   │  ├─ t: [B]
   │  ├─ start_pos: [B, D]
   │  ├─ goal_pos: [B, D]
   │  └─ start_vel: [B, D]
   │
   ├─ FlowMPTransformer 内部流程:
   │  │
   │  ├─ 【第一步】输入投影与嵌入
   │  │   │
   │  │   ├─ 输入线性映射:
   │  │   │  x_t: [B, T, 6] → [B, T, hidden_dim]
   │  │   │  └─ Linear(6, hidden_dim)
   │  │   │
   │  │   ├─ 位置编码 (Positional Encoding):
   │  │   │  └─ PE(pos) = [sin(pos/10000^(2i/d)), cos(...)]
   │  │   │  └─ 添加到每个时间步: x' = x + PE(t)
   │  │   │
   │  │   └─ 输出: [B, T, hidden_dim]
   │  │
   │  ├─ 【第二步】时间嵌入
   │  │   │
   │  │   ├─ Gaussian Fourier Projection:
   │  │   │  ├─ sin(2π*t*b_k) for k in range(num_fourier)
   │  │   │  ├─ cos(2π*t*b_k) for k in range(num_fourier)
   │  │   │  └─ 其中 b_k ~ N(0, σ²)
   │  │   │
   │  │   └─ 输出: time_emb [B, cond_dim]
   │  │
   │  ├─ 【第三步】条件编码
   │  │   │
   │  │   ├─ ConditionEncoder 处理:
   │  │   │  ├─ concat([start_pos, goal_pos, start_vel]) → [B, 5*D]
   │  │   │  ├─ Linear(5*D, hidden_dim)
   │  │   │  └─ 输出: cond_emb [B, cond_dim]
   │  │   │
   │  │   └─ 注意：start_vel 在目标位置时为 [0, 0]
   │  │
   │  ├─ 【第四步】条件融合
   │  │   │
   │  │   ├─ 方法：加法融合或连接后映射
   │  │   │  └─ combined_cond = time_emb + cond_emb
   │  │   │  └─ 形状: [B, hidden_dim]
   │  │   │
   │  │   └─ 作用：用于 AdaLN 的参数生成
   │  │
   │  ├─ 【第五步】Transformer 编码器 (L 层)
   │  │   │
   │  │   └─ 对每一层:
   │  │      │
   │  │      ├─ [A] AdaLN (Adaptive Layer Normalization):
   │  │      │   │
   │  │      │   ├─ 计算 scale γ 和 shift β:
   │  │      │   │  γ = Linear_scale(combined_cond)  [B, hidden_dim]
   │  │      │   │  β = Linear_shift(combined_cond)  [B, hidden_dim]
   │  │      │   │
   │  │      │   ├─ 对所有序列位置应用:
   │  │      │   │  LN_out = (x - μ) / σ
   │  │      │   │  AdaLN_out = γ * LN_out + β
   │  │      │   │
   │  │      │   └─ 输出: [B, T, hidden_dim]
   │  │      │
   │  │      ├─ [B] 多头自注意力 (Multi-Head Self-Attention):
   │  │      │   │
   │  │      │   ├─ 线性映射生成 Q, K, V:
   │  │      │   │  Q = Linear_q(x)  [B, T, hidden_dim]
   │  │      │   │  K = Linear_k(x)  [B, T, hidden_dim]
   │  │      │   │  V = Linear_v(x)  [B, T, hidden_dim]
   │  │      │   │
   │  │      │   ├─ 分头与缩放点积注意力:
   │  │      │   │  分头: Q,K,V 分别分成 num_heads 份
   │  │      │   │  attention = softmax(Q@K^T / √d_k) @ V
   │  │      │   │
   │  │      │   ├─ 合并多头:
   │  │      │   │  output = Linear_out(concat(heads))
   │  │      │   │
   │  │      │   ├─ 残差连接:
   │  │      │   │  out = x + attention_out
   │  │      │   │
   │  │      │   └─ 输出: [B, T, hidden_dim]
   │  │      │
   │  │      ├─ [C] AdaLN (第二次):
   │  │      │   └─ 同上，使用相同的 γ 和 β
   │  │      │   └─ 输出: [B, T, hidden_dim]
   │  │      │
   │  │      └─ [D] 前馈网络 (Feed-Forward Network):
   │  │          │
   │  │          ├─ 结构: Linear(d) → ReLU → Linear(4d) → ReLU → Linear(d)
   │  │          │  (或 GELU 作为激活函数)
   │  │          │
   │  │          ├─ 计算过程:
   │  │          │  ffn_out = Linear_2(ReLU(Linear_1(x)))
   │  │          │
   │  │          ├─ 残差连接:
   │  │          │  out = x + ffn_out
   │  │          │
   │  │          └─ 输出: [B, T, hidden_dim]
   │  │
   │  ├─ 【第六步】最后一层 AdaLN
   │  │   └─ 对整个序列应用最后的自适应归一化
   │  │   └─ 输出: [B, T, hidden_dim]
   │  │
   │  └─ 【第七步】输出头
   │      │
   │      ├─ 输出映射:
   │      │  output = Linear(hidden_dim, 6)  # 映射到 3 个向量场
   │      │
   │      ├─ 分解:
   │      │  u_pred = output[..., :D]  # 位置速度场 [B, T, D]
   │      │  v_pred = output[..., D:2*D]  # 加速度场 [B, T, D]
   │      │  w_pred = output[..., 2*D:3*D]  # 急动场 [B, T, D]
   │      │
   │      └─ 输出: [B, T, 6]
   │
   └─ 模型输出:
      ├─ u_pred: [B, T, D] 预测的位置速度场
      ├─ v_pred: [B, T, D] 预测的加速度场
      └─ w_pred: [B, T, D] 预测的急动场

4. 损失计算与反向传播 (Loss & Optimization)
   │
   ├─ 损失函数计算:
   │  │
   │  ├─ 位置速度场损失:
   │  │  L_u = MSE(u_pred, u_target)
   │  │      = (1/(B*T*D)) * sum((u_pred - u_target)²)
   │  │
   │  ├─ 加速度场损失:
   │  │  L_v = MSE(v_pred, v_target)
   │  │      = (1/(B*T*D)) * sum((v_pred - v_target)²)
   │  │
   │  ├─ 急动场损失:
   │  │  L_w = MSE(w_pred, w_target)
   │  │      = (1/(B*T*D)) * sum((w_pred - w_target)²)
   │  │
   │  └─ 加权总损失:
   │     L_total = λ_u * L_u + λ_v * L_v + λ_w * L_w
   │     (默认: λ_u = λ_v = λ_w = 1.0)
   │
   ├─ 反向传播:
   │  │
   │  ├─ 计算梯度: ∇L w.r.t. 所有参数
   │  │
   │  ├─ 梯度裁剪 (可选):
   │  │  └─ 防止梯度爆炸: norm(∇) → min(norm(∇), clip_value)
   │  │
   │  └─ 梯度累积 (可选):
   │     └─ 多个批次的梯度相加，然后一起优化
   │
   └─ 优化器更新:
      │
      ├─ Adam 优化器:
      │  ├─ m_t = β1 * m_{t-1} + (1-β1) * ∇L  (一阶矩)
      │  ├─ v_t = β2 * v_{t-1} + (1-β2) * (∇L)²  (二阶矩)
      │  ├─ m̂_t = m_t / (1-β1^t)  (偏差修正)
      │  ├─ v̂_t = v_t / (1-β2^t)
      │  └─ θ_{t+1} = θ_t - lr * m̂_t / (√v̂_t + ε)
      │
      ├─ 学习率调度 (可选):
      │  └─ Warmup + Cosine Decay
      │     lr(step) = base_lr * min(step/warmup_steps, cos_decay(step))
      │
      └─ 参数更新完成
         └─ 继续下一批次训练
```

#### 数据维度追踪示例

```python
# 假设参数: B=32, T=64, D=2, hidden_dim=256

# 批次数据
positions: torch.Size([32, 64, 2])
velocities: torch.Size([32, 64, 2])
accelerations: torch.Size([32, 64, 2])

# Flow Matching 阶段
t: torch.Size([32])  # 流时间
x_t: torch.Size([32, 64, 6])  # 拼接: [pos, vel, acc]

# 目标场
u_target: torch.Size([32, 64, 2])  # 位置速度场
v_target: torch.Size([32, 64, 2])  # 加速度场
w_target: torch.Size([32, 64, 2])  # 急动场

# 嵌入
time_emb: torch.Size([32, 256])  # 时间嵌入
cond_emb: torch.Size([32, 256])  # 条件嵌入

# 模型中间
x_embed: torch.Size([32, 64, 256])  # 输入投影
x_pos_enc: torch.Size([32, 64, 256])  # 加入位置编码
x_attn: torch.Size([32, 64, 256])  # 注意力输出
x_ffn: torch.Size([32, 64, 256])  # 前馈输出

# 模型输出
u_pred: torch.Size([32, 64, 2])
v_pred: torch.Size([32, 64, 2])
w_pred: torch.Size([32, 64, 2])
```

---

### 2.2 L2层三层架构集成数据流

#### L3→L2→L1 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│           三层架构数据流 (Three-Tier Integration)               │
└─────────────────────────────────────────────────────────────────┘

【L3层：Vision-Language Model】
   │
   ├─ 输入：
   │  ├─ RGB图像 / 深度图像 [B, 3, H_img, W_img]
   │  ├─ 语言指令 (文本)
   │  └─ 传感器数据（激光雷达、IMU等）
   │
   ├─ 处理：
   │  ├─ 场景理解与语义分割
   │  ├─ 障碍物检测与分类
   │  ├─ 语言指令解析
   │  └─ 风险评估
   │
   └─ 输出给L2：
      ├─ cost_map: [B, C, H, W]  # 语义代价地图
      │  ├─ C通道包含：
      │  │  ├─ 障碍物占用概率 [0, 1]
      │  │  ├─ 可通行性评分 [0, 1]
      │  │  ├─ 风险等级 [0, 1]
      │  │  └─ 其他语义信息...
      │  └─ 空间分辨率: H×W (如 64×64 或 128×128)
      │
      └─ 高级语义标签（可选）

【L2层：Safety-Embedded CFM】本项目核心
   │
   ├─ 输入来源：
   │  ├─ 从L3: cost_map [B, C, H, W]
   │  ├─ 机器人状态: x_curr [B, 6] = [p_x, p_y, v_x, v_y, a_x, a_y]
   │  ├─ 目标状态: x_goal [B, 4] = [p_goal_x, p_goal_y, v_goal_x, v_goal_y]
   │  └─ 控制风格权重: w_style [B, 3] = [w_safety, w_energy, w_smooth]
   │
   ├─ 【步骤1】代价地图编码
   │  │
   │  ├─ CostMapEncoder.forward(cost_map)
   │  │  │
   │  │  ├─ CNN特征提取:
   │  │  │  ├─ Conv2D(C → 32) + ResBlock → [B, 32, H, W]
   │  │  │  ├─ Downsample → [B, 64, H/2, W/2]
   │  │  │  ├─ ResBlock + Downsample → [B, 128, H/4, W/4]
   │  │  │  ├─ ResBlock + Downsample → [B, 256, H/8, W/8]
   │  │  │  └─ ResBlock + Downsample → [B, 512, H/16, W/16]
   │  │  │
   │  │  ├─ 全局池化:
   │  │  │  └─ AdaptiveAvgPool2D → [B, 512]
   │  │  │
   │  │  └─ 投影到潜在空间:
   │  │     └─ Linear(512 → latent_dim) → [B, 256]
   │  │
   │  └─ 输出: e_map [B, 256] 代价地图的潜在表示
   │
   ├─ 【步骤2】条件融合
   │  │
   │  ├─ L2SafetyCFM.prepare_condition()
   │  │  │
   │  │  ├─ 状态编码:
   │  │  │  ├─ x_curr [B, 6] → Linear → [B, 128]
   │  │  │  ├─ x_goal [B, 4] → Linear → [B, 128]
   │  │  │  └─ w_style [B, 3] → Linear → [B, 64]
   │  │  │
   │  │  ├─ 拼接融合:
   │  │  │  combined = cat([e_map, x_curr_enc, x_goal_enc, w_style_enc])
   │  │  │  └─ [B, 256+128+128+64] = [B, 576]
   │  │  │
   │  │  └─ 最终投影:
   │  │     └─ Linear(576 → cond_dim) → [B, cond_dim]
   │  │
   │  └─ 输出: condition [B, cond_dim] 统一条件向量
   │
   ├─ 【步骤3】多模态轨迹生成
   │  │
   │  ├─ L2SafetyCFM.generate_trajectory_anchors()
   │  │  │
   │  │  ├─ 为每个样本生成N=64个候选:
   │  │  │  │
   │  │  │  ├─ 重复条件: condition [B, cond_dim] → [B×N, cond_dim]
   │  │  │  │
   │  │  │  ├─ 采样高斯噪声:
   │  │  │  │  x_0 ~ N(0, I) → [B×N, T, 6]
   │  │  │  │
   │  │  │  ├─ ODE求解 (RK4):
   │  │  │  │  ├─ velocity_fn = λ(x, t): model(x, t, condition)
   │  │  │  │  ├─ 求解: x_0 → x_1 通过 dx/dt = velocity_fn(x, t)
   │  │  │  │  └─ 输出: x_1 [B×N, T, 6]
   │  │  │  │
   │  │  │  └─ 提取动力学分量:
   │  │  │     ├─ trajectories = x_1[..., :2]  [B×N, T, 2]
   │  │  │     ├─ velocities = x_1[..., 2:4]  [B×N, T, 2]
   │  │  │     └─ accelerations = x_1[..., 4:6]  [B×N, T, 2]
   │  │  │
   │  │  ├─ (可选) CBF安全约束:
   │  │  │  └─ 投影轨迹到安全集合
   │  │  │  └─ 保证 safety_margin = 0.1m
   │  │  │
   │  │  └─ (可选) B-spline平滑:
   │  │     └─ 对每条轨迹进行平滑处理
   │  │
   │  └─ 输出: 
   │     ├─ trajectories [B×N, T, 2] 位置轨迹
   │     ├─ velocities [B×N, T, 2] 速度轨迹
   │     ├─ accelerations [B×N, T, 2] 加速度轨迹
   │     └─ metadata (可选): 每条轨迹的质量评分、安全性评估
   │
   └─ 输出给L1：
      └─ trajectory_anchors = {
         'trajectories': [B×64, T, 2],
         'velocities': [B×64, T, 2],
         'accelerations': [B×64, T, 2]
      }

【L1层：Model Predictive Path Integral (MPPI)】
   │
   ├─ 输入来源：
   │  ├─ 从L2: trajectory_anchors (64条候选轨迹)
   │  ├─ 当前机器人状态
   │  └─ 实时传感器数据
   │
   ├─ 处理：
   │  ├─ 轨迹锚点选择:
   │  │  ├─ 评估64条轨迹的局部代价
   │  │  ├─ 选择top-K条作为warm-start
   │  │  └─ 或使用加权组合
   │  │
   │  ├─ 局部采样优化:
   │  │  ├─ 在选定锚点附近采样扰动
   │  │  ├─ 前向rollout评估代价
   │  │  └─ 计算最优控制序列
   │  │
   │  └─ 实时约束检查:
   │     ├─ 动力学约束
   │     ├─ 碰撞检测
   │     └─ 执行器限制
   │
   └─ 输出：
      ├─ 最优控制输入 u_optimal [horizon_length]
      ├─ 预测轨迹
      └─ 执行下一步控制

【关键数据流接口】

L3 → L2:
  - cost_map: [B, C, H, W] torch.Tensor, dtype=float32
  - 包含语义风险、可通行性、障碍物信息

L2 → L1:
  - trajectories: [B×N, T, 2] torch.Tensor, dtype=float32
  - velocities: [B×N, T, 2] torch.Tensor, dtype=float32
  - accelerations: [B×N, T, 2] torch.Tensor, dtype=float32
  - N=64条多模态候选，T=序列长度，2D空间

【控制风格权重效果】

w_style = [w_safety, w_energy, w_smooth]:
  - w_safety高 → 远离障碍物，保守路径
  - w_energy高 → 最短路径，激进行为
  - w_smooth高 → 平滑曲线，舒适运动

示例配置:
  - 保守模式: [0.7, 0.1, 0.2] - 安全优先
  - 平衡模式: [0.4, 0.3, 0.3] - 综合考虑
  - 高效模式: [0.2, 0.5, 0.3] - 效率优先
```

#### L2层训练数据需求

```
训练数据格式 (用于L2层):

必需数据:
├─ trajectories: [N_samples, T, 2]  # 专家轨迹（位置）
├─ velocities: [N_samples, T, 2]    # 速度
├─ accelerations: [N_samples, T, 2] # 加速度
├─ cost_maps: [N_samples, C, H, W]  # 对应的场景代价地图
├─ start_states: [N_samples, 6]     # 初始状态 [p, v, a]
├─ goal_states: [N_samples, 4]      # 目标状态 [p, v]
└─ style_weights: [N_samples, 3]    # 控制风格 [w_safety, w_energy, w_smooth]

可选数据:
├─ scene_images: [N_samples, 3, H_img, W_img]  # RGB图像（如需端到端训练）
├─ safety_margins: [N_samples]                  # 每条轨迹的安全余量
└─ trajectory_quality: [N_samples]              # 轨迹质量标签

数据来源:
1. 人类演示数据（遥操作、VR录制）
2. 专家策略生成（如路径规划算法）
3. 模拟器采样（如Gazebo、Isaac Sim）
4. 真实机器人数据采集
```

---

### 2.3 推理阶段数据流

#### 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                   推理数据流 (Inference Flow)                    │
└─────────────────────────────────────────────────────────────────┘

1. 初始化与输入准备 (Initialization)
   │
   ├─ 加载预训练模型
   │  ├─ 从 checkpoint 文件加载
   │  ├─ 设置为评估模式: model.eval()
   │  ├─ 禁用梯度计算: torch.no_grad()
   │  └─ 可选：移至 GPU
   │
   ├─ 设置生成器配置 (GeneratorConfig)
   │  ├─ solver_type: "rk4" (或 "euler", "midpoint", "rk45")
   │  ├─ time_schedule: 自定义时间步 (或 None 自动生成)
   │  ├─ num_steps: 20 (仅在 time_schedule=None 时使用)
   │  ├─ use_bspline_smoothing: True (B-spline 后处理)
   │  ├─ bspline_degree: 3
   │  └─ state_dim, seq_len: 模型参数
   │
   └─ 输入条件准备
      ├─ start_pos: [B, D] 起始位置
      │  └─ 例如: [[0.0, 0.0], [1.0, 1.0]] for B=2
      │
      ├─ goal_pos: [B, D] 目标位置
      │  └─ 例如: [[2.0, 2.0], [3.0, 1.0]] for B=2
      │
      └─ start_vel: [B, D] (可选) 起始速度
         └─ 默认: [[0.0, 0.0], [0.0, 0.0]]

2. 噪声采样 (Noise Sampling)
   │
   └─ 高斯分布采样
      │
      ├─ 采样初始状态:
      │  x_0 ~ N(0, I) → [B, T, 6]
      │  └─ 分解为:
      │     ├─ pos_0: [B, T, D] ~ N(0, 1)
      │     ├─ vel_0: [B, T, D] ~ N(0, 1)
      │     └─ acc_0: [B, T, D] ~ N(0, 1)
      │
      └─ 目的：为 ODE 求解提供初始条件
         └─ t=0 时，x_0 是纯噪声
         └─ t=1 时，x_1 应该是有效轨迹

3. ODE 求解 (ODE Integration)
   │
   ├─ 【第一步】准备时间调度
   │  │
   │  ├─ 选项 1: 均匀调度
   │  │  └─ time_schedule = np.linspace(0, 1, num_steps)
   │  │  └─ 例: [0.00, 0.05, 0.10, ..., 0.95, 1.00]
   │  │
   │  ├─ 选项 2: 8 步非均匀调度 (推荐，快速推理)
   │  │  └─ time_schedule = [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
   │  │  └─ 特点：早期大步长（快速去噪），后期小步长（精细调整）
   │  │
   │  └─ 选项 3: 自定义调度
   │     └─ 用户提供的自定义时间点列表
   │
   ├─ 【第二步】定义速度函数
   │  │
   │  ├─ velocity_fn(x_t, t) 的作用：
   │  │  └─ 根据当前状态和时间，计算变化率 dx/dt
   │  │
   │  ├─ 实现：
   │  │  │
   │  │  ├─ 扩展时间 t:
   │  │  │  │ 如果 t 是标量，需要扩展为 [B] 的形状
   │  │  │  │ t_expanded: [1] → [B]
   │  │  │  │
   │  │  ├─ 模型前向:
   │  │  │  │ output = model(x_t, t_expanded, start_pos, goal_pos, start_vel)
   │  │  │  │ output: [B, T, 6]
   │  │  │  │
   │  │  └─ 返回：[B, T, 6] 向量场值
   │  │
   │  └─ 目的：提供 ODE 求解器所需的导数信息
   │
   ├─ 【第三步】RK4 ODE 求解
   │  │
   │  ├─ 算法：4 阶 Runge-Kutta 方法
   │  │  │
   │  │  ├─ 初始化: x_current = x_0 [B, T, 6]
   │  │  │
   │  │  └─ 对每一对时间步 (t_i, t_{i+1}):
   │  │     │
   │  │     ├─ dt = t_{i+1} - t_i
   │  │     │
   │  │     ├─ 计算 4 个斜率:
   │  │     │  │
   │  │     │  ├─ k1 = velocity_fn(x_current, t_i)  [B, T, 6]
   │  │     │  │  └─ 在当前点的斜率
   │  │     │  │
   │  │     │  ├─ k2 = velocity_fn(x_current + 0.5*dt*k1, t_i + 0.5*dt)  [B, T, 6]
   │  │     │  │  └─ 在中点处向前看 1/2 步的斜率
   │  │     │  │
   │  │     │  ├─ k3 = velocity_fn(x_current + 0.5*dt*k2, t_i + 0.5*dt)  [B, T, 6]
   │  │     │  │  └─ 另一种中点估计
   │  │     │  │
   │  │     │  └─ k4 = velocity_fn(x_current + dt*k3, t_i + dt)  [B, T, 6]
   │  │     │     └─ 在下一个点的斜率
   │  │     │
   │  │     └─ 加权平均更新:
   │  │        x_{i+1} = x_i + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
   │  │        └─ 加权组合 4 个斜率估计
   │  │
   │  ├─ 精度与成本：
   │  │  ├─ 20 步: O(h^4) 精度，快速
   │  │  ├─ 50 步: O(h^4) 精度，更精确但较慢
   │  │  └─ 8 步非均匀: 最快，保留细节
   │  │
   │  └─ 输出：x_1 [B, T, 6]（t=1 时的状态）
   │
   ├─ 【第四步】提取轨迹分量
   │  │
   │  ├─ x_1: [B, T, 6]
   │  │
   │  ├─ 分解方式:
   │  │  ├─ positions_raw = x_1[..., :D]  [B, T, D]
   │  │  ├─ velocities_raw = x_1[..., D:2*D]  [B, T, D]
   │  │  └─ accelerations_raw = x_1[..., 2*D:3*D]  [B, T, D]
   │  │
   │  └─ 特点：
   │     ├─ 位置是连续的光滑轨迹
   │     ├─ 速度可能有噪声（ODE 积分误差）
   │     └─ 加速度最嘈杂（二阶导数放大误差）
   │
   └─ 【第五步】去噪处理 (可选，若有噪声)
      └─ 对速度和加速度进行平滑处理

4. 后处理：B-spline 平滑 (Post-processing)
   │
   ├─ BSplineSmoother.smooth() 的目的：
   │  ├─ 消除 ODE 积分引入的数值噪声
   │  ├─ 确保速度和加速度的物理一致性
   │  ├─ 减少高频振荡
   │  └─ 改善轨迹的光滑性
   │
   ├─ 处理流程 (对每个样本):
   │  │
   │  ├─ 【位置平滑】:
   │  │  │
   │  │  ├─ 对位置轨迹 positions [T, D]:
   │  │  │  ├─ 对每个维度单独处理:
   │  │  │  │  ├─ 拟合 B-spline 曲线 (degree=3, 通常 cubic spline)
   │  │  │  │  │  └─ positions_1d: [T] → BSpline曲线
   │  │  │  │  │
   │  │  │  │  └─ 重新采样到原始时间点:
   │  │  │  │     ├─ 生成均匀的参数值: u = linspace(0, 1, T)
   │  │  │  │     └─ 在 u 上评估 B-spline
   │  │  │  │
   │  │  │  └─ 输出：平滑位置 pos_smooth [T, D]
   │  │  │
   │  │  └─ 结果：消除了位置噪声
   │  │
   │  ├─ 【速度重新计算】:
   │  │  │
   │  │  ├─ 方法：从平滑位置计算导数
   │  │  │
   │  │  ├─ 实现：
   │  │  │  ├─ 通过中心差分计算速度:
   │  │  │  │  vel[i] = (pos[i+1] - pos[i-1]) / (2 * dt)
   │  │  │  │
   │  │  │  ├─ 边界处理:
   │  │  │  │  vel[0] = (pos[1] - pos[0]) / dt
   │  │  │  │  vel[-1] = (pos[-1] - pos[-2]) / dt
   │  │  │  │
   │  │  │  └─ 输出：速度 vel_smooth [T, D]
   │  │  │
   │  │  └─ 结果：速度与位置一致，消除了不连续
   │  │
   │  └─ 【加速度重新计算】:
   │     │
   │     ├─ 同样从平滑速度计算:
   │     │  acc[i] = (vel[i+1] - vel[i-1]) / (2 * dt)
   │     │
   │     └─ 输出：加速度 acc_smooth [T, D]
   │
   ├─ 平滑参数:
   │  ├─ bspline_degree: 3 (cubic, 通常最优)
   │  ├─ smoothing_factor: 控制拟合紧密度
   │  └─ knots: B-spline 节点数
   │
   └─ 输出张量:
      ├─ positions: [B, T, D] 平滑位置轨迹
      ├─ velocities: [B, T, D] 平滑速度轨迹
      └─ accelerations: [B, T, D] 平滑加速度轨迹

5. 最终输出 (Final Output)
   │
   └─ 结果字典:
      ├─ 'positions': [B, T, D]
      │  └─ 平滑的空间轨迹
      │
      ├─ 'velocities': [B, T, D]
      │  └─ 与位置一致的速度
      │
      ├─ 'accelerations': [B, T, D]
      │  └─ 一致的加速度
      │
      └─ (可选) 'raw_positions': [B, T, D]
         └─ 未平滑的原始输出 (用于调试)
```

#### 数据维度追踪示例

```python
# 推理参数: B=5, T=64, D=2

# 输入条件
start_pos: torch.Size([5, 2])  # 5 个轨迹，2D 位置
goal_pos: torch.Size([5, 2])
start_vel: torch.Size([5, 2])

# 初始噪声
x_0: torch.Size([5, 64, 6])  # 5 个批次，64 时间步，6 维状态

# ODE 求解过程（每个时间步）
t_i: float  # 当前时间
x_current: torch.Size([5, 64, 6])  # 当前状态

# 速度函数调用
velocity_output: torch.Size([5, 64, 6])  # 模型输出的向量场

# RK4 斜率
k1, k2, k3, k4: torch.Size([5, 64, 6])  # 4 个斜率
x_next: torch.Size([5, 64, 6])  # 下一时间步状态

# ODE 求解完成
x_1: torch.Size([5, 64, 6])  # 最终状态

# 分解
positions_raw: torch.Size([5, 64, 2])
velocities_raw: torch.Size([5, 64, 2])
accelerations_raw: torch.Size([5, 64, 2])

# B-spline 平滑后
positions_smooth: torch.Size([5, 64, 2])
velocities_smooth: torch.Size([5, 64, 2])
accelerations_smooth: torch.Size([5, 64, 2])
```

#### 推理算法伪代码

```python
def inference(model, start_pos, goal_pos, start_vel, config):
    B = start_pos.shape[0]  # 批次大小
    T = config.seq_len      # 时间步数
    D = config.state_dim    # 状态维度
    
    # 第一步：采样初始噪声
    x_0 = torch.randn(B, T, 6)
    
    # 第二步：准备时间调度
    if config.use_8step_schedule:
        times = [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
    else:
        times = np.linspace(0, 1, config.num_steps)
    
    # 第三步：定义速度函数
    def velocity_fn(x_t, t):
        # 扩展 t 为批次形状
        t_batch = torch.full((B,), t, device=x_t.device)
        # 模型前向
        output = model(x_t, t_batch, start_pos, goal_pos, start_vel)
        return output  # [B, T, 6]
    
    # 第四步：RK4 ODE 求解
    x_current = x_0
    for i in range(len(times) - 1):
        t_i = times[i]
        t_next = times[i + 1]
        dt = t_next - t_i
        
        # 计算 4 个斜率
        k1 = velocity_fn(x_current, t_i)
        k2 = velocity_fn(x_current + 0.5*dt*k1, t_i + 0.5*dt)
        k3 = velocity_fn(x_current + 0.5*dt*k2, t_i + 0.5*dt)
        k4 = velocity_fn(x_current + dt*k3, t_next)
        
        # 更新状态
        x_current = x_current + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    x_1 = x_current
    
    # 第五步：提取分量
    positions = x_1[..., :D]
    velocities = x_1[..., D:2*D]
    accelerations = x_1[..., 2*D:3*D]
    
    # 第六步：B-spline 平滑（可选）
    if config.use_bspline_smoothing:
        positions = bspline_smooth(positions)
        # 重新计算速度和加速度以保持一致性
        velocities = compute_derivatives(positions)
        accelerations = compute_derivatives(velocities)
    
    return {
        'positions': positions,      # [B, T, D]
        'velocities': velocities,    # [B, T, D]
        'accelerations': accelerations  # [B, T, D]
    }
```

### 2.3 网络架构详解 (Network Architecture)

#### FlowMPTransformer 的完整结构

```
┌─────────────────────────────────────────────────────────────────┐
│                   FlowMPTransformer 架构                        │
└─────────────────────────────────────────────────────────────────┘

【模块 0】输入处理层
├─ 输入: x_t [B, T, 6]  (位置、速度、加速度拼接)
├─ 条件: start_pos [B, D], goal_pos [B, D], start_vel [B, D]
├─ 时间: t [B]
│
├─ 第一步：输入线性投影
│  ├─ Linear(6, hidden_dim)
│  ├─ 作用：将 6 维状态映射到隐藏空间
│  ├─ 维度变化: [B, T, 6] → [B, T, hidden_dim]
│  └─ 参数量: 6 * hidden_dim + hidden_dim (偏置)
│
├─ 第二步：加入位置编码 (Positional Encoding)
│  │
│  ├─ 类型：正弦位置编码（标准 Transformer 方式）
│  │
│  ├─ 公式:
│  │  PE(pos, 2i) = sin(pos / 10000^(2i/d))
│  │  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
│  │  其中 pos 是时间步索引 (0, 1, 2, ..., T-1)
│  │       d 是隐藏维度
│  │
│  ├─ 维度: [T, hidden_dim]
│  │
│  ├─ 添加方式: x' = x + PE
│  │  └─ 逐元素相加，形状广播
│  │
│  └─ 输出: [B, T, hidden_dim]
│
├─ 第三步：时间嵌入 (Time Embedding)
│  │
│  ├─ 方法：Gaussian Fourier 投影
│  │
│  ├─ 过程：
│  │  │
│  │  ├─ 采样频率向量:
│  │  │  ├─ b ~ N(0, σ²) [num_fourier_features]
│  │  │  └─ σ 通常取 1.0 或 16.0（可配置）
│  │  │
│  │  ├─ 傅里叶投影（对每个样本 t ~ [0, 1]）:
│  │  │  ├─ sin_features = sin(2π * t * b)  [num_fourier_features]
│  │  │  ├─ cos_features = cos(2π * t * b)  [num_fourier_features]
│  │  │  └─ concat([sin_features, cos_features])  [2*num_fourier_features]
│  │  │
│  │  └─ 线性层将傅里叶特征映射到隐藏维:
│  │     Linear(2*num_fourier_features, cond_dim)
│  │
│  ├─ 维度变化: [B] → [B, cond_dim]
│  │
│  └─ 作用：
│     ├─ 捕捉时间的周期性信息
│     ├─ 使模型能够学习时间相关的动态
│     └─ 为 AdaLN 提供条件参数
│
├─ 第四步：条件编码 (Condition Encoder)
│  │
│  ├─ 输入拼接:
│  │  └─ concat([start_pos, goal_pos, start_vel])
│  │  └─ 维度: [B, 5*D]  (3 个 D 维向量)
│  │
│  ├─ 线性投影:
│  │  └─ Linear(5*D, cond_dim)
│  │  └─ 维度: [B, cond_dim]
│  │
│  └─ 作用：
│     ├─ 编码起点、目标和初速度信息
│     ├─ 为生成过程提供约束条件
│     └─ 通过 AdaLN 注入到网络各层
│
└─ 第五步：条件融合 (Condition Fusion)
   │
   ├─ 组合时间和条件:
   │  ├─ combined_cond = time_emb + cond_emb
   │  ├─ 维度: [B, cond_dim]
   │  └─ 方式：逐元素相加（或可选连接后线性映射）
   │
   └─ 用途：用于 AdaLN 的参数生成

【模块 1 到 L-1】Transformer 编码器块 (×L 层)

注意：每层的处理步骤完全相同，只有参数不同

┌──────────────────────────────────────────────────────────┐
│ Block i (i = 0, 1, ..., L-1)                             │
│                                                           │
│ 输入: x [B, T, hidden_dim]                              │
│ 条件: combined_cond [B, cond_dim]                        │
│                                                           │
│ ① AdaLN (Adaptive Layer Normalization)                 │
│    │                                                       │
│    ├─ 计算 γ 和 β:                                        │
│    │  ├─ γ = Linear_gamma(combined_cond) [B, hidden_dim] │
│    │  └─ β = Linear_beta(combined_cond) [B, hidden_dim]  │
│    │                                                       │
│    ├─ 对所有序列位置应用自适应归一化:                      │
│    │  ├─ 计算统计量: μ = mean(x, dim=-1) [B, T]          │
│    │  │               σ = std(x, dim=-1) [B, T]           │
│    │  │                                                    │
│    │  ├─ 归一化: x_norm = (x - μ) / σ                    │
│    │  │                                                    │
│    │  └─ 自适应变换: y = γ * x_norm + β                  │
│    │     (γ 和 β 对所有时间步重复应用)                   │
│    │                                                       │
│    └─ 输出: [B, T, hidden_dim]                           │
│                                                           │
│ ② Multi-Head Self-Attention                             │
│    │                                                       │
│    ├─ Q, K, V 投影:                                      │
│    │  ├─ Q = Linear_q(x)  [B, T, hidden_dim]             │
│    │  ├─ K = Linear_k(x)  [B, T, hidden_dim]             │
│    │  └─ V = Linear_v(x)  [B, T, hidden_dim]             │
│    │                                                       │
│    ├─ 分头:                                               │
│    │  ├─ num_heads = hidden_dim / head_dim                │
│    │  ├─ Q: [B, T, hidden_dim] → [B, num_heads, T, head_dim] │
│    │  ├─ K: [B, T, hidden_dim] → [B, num_heads, T, head_dim] │
│    │  └─ V: [B, T, hidden_dim] → [B, num_heads, T, head_dim] │
│    │                                                       │
│    ├─ 缩放点积注意力 (Scaled Dot-Product Attention):      │
│    │  │                                                    │
│    │  ├─ scores = Q @ K^T / √head_dim                    │
│    │  │   维度: [B, num_heads, T, T]                     │
│    │  │                                                    │
│    │  ├─ attention_weights = softmax(scores)              │
│    │  │   维度: [B, num_heads, T, T]                     │
│    │  │                                                    │
│    │  ├─ attention_output = attention_weights @ V         │
│    │  │   维度: [B, num_heads, T, head_dim]              │
│    │  │                                                    │
│    │  └─ 含义：每个位置学习关注其他位置的信息             │
│    │                                                       │
│    ├─ 合并多头:                                           │
│    │  └─ concat(heads) → [B, T, hidden_dim]              │
│    │                                                       │
│    ├─ 输出投影:                                           │
│    │  └─ Linear_out(concat) → [B, T, hidden_dim]         │
│    │                                                       │
│    ├─ 残差连接:                                           │
│    │  └─ output = x + attention_output                    │
│    │                                                       │
│    └─ 输出: [B, T, hidden_dim]                           │
│                                                           │
│ ③ AdaLN (第二次)                                         │
│    │                                                       │
│    ├─ 同① 的过程，使用相同的 γ 和 β                     │
│    │                                                       │
│    └─ 输出: [B, T, hidden_dim]                           │
│                                                           │
│ ④ Feed-Forward Network (FFN)                            │
│    │                                                       │
│    ├─ 结构: Linear → 激活函数 → Linear                    │
│    │                                                       │
│    ├─ 详细过程:                                           │
│    │  ├─ ffn_hidden = Linear1(x)  [B, T, ffn_dim]        │
│    │  │  其中 ffn_dim = 4 * hidden_dim (标准配置)         │
│    │  │                                                    │
│    │  ├─ activated = ReLU(ffn_hidden)  [B, T, ffn_dim]   │
│    │  │  (或 GELU, SiLU 等激活函数)                      │
│    │  │                                                    │
│    │  ├─ ffn_output = Linear2(activated)  [B, T, hidden_dim] │
│    │  │                                                    │
│    │  └─ 含义：逐位置的非线性变换，扩展表示能力           │
│    │                                                       │
│    ├─ 残差连接:                                           │
│    │  └─ output = x + ffn_output                          │
│    │                                                       │
│    └─ 输出: [B, T, hidden_dim]                           │
│                                                           │
│ 本块输出: [B, T, hidden_dim]                             │
└──────────────────────────────────────────────────────────┘

【模块 L】最后的归一化和输出头

├─ 最后 AdaLN 层:
│  ├─ 对整个序列应用最后的自适应归一化
│  └─ 维度: [B, T, hidden_dim] → [B, T, hidden_dim]
│
└─ 输出头 (Output Head):
   │
   ├─ 线性映射:
   │  └─ Linear(hidden_dim, 6)
   │  └─ 维度: [B, T, hidden_dim] → [B, T, 6]
   │
   ├─ 分解输出向量场:
   │  ├─ u_pred = output[..., :D]          # [B, T, D] 位置速度场
   │  ├─ v_pred = output[..., D:2*D]      # [B, T, D] 加速度场
   │  └─ w_pred = output[..., 2*D:3*D]    # [B, T, D] 急动场
   │
   └─ 输出: [B, T, 6] 完整向量场预测
```

#### 参数配置与计算

```python
# 模型变体配置
MODEL_CONFIG = {
    "small": {
        "hidden_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "ffn_dim": 512,  # 4 * hidden_dim
        "head_dim": 32,  # hidden_dim / num_heads
        "cond_dim": 64,
        "num_fourier": 32,
        "num_params": "~3M",
    },
    "base": {
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "ffn_dim": 1024,
        "head_dim": 32,
        "cond_dim": 128,
        "num_fourier": 64,
        "num_params": "~12M",
    },
    "large": {
        "hidden_dim": 512,
        "num_layers": 8,
        "num_heads": 16,
        "ffn_dim": 2048,
        "head_dim": 32,
        "cond_dim": 256,
        "num_fourier": 128,
        "num_params": "~50M",
    },
}

# 参数量计算示例 (base 版本)
# 假设 D=2 (状态维度), T=64 (序列长度)

param_count = {
    "input_projection": 6 * 256 + 256,  # 1.6K
    "pos_encoding": 0,  # 无可学习参数
    "time_embedding": (2*64) * 256 + 256,  # 33K
    "condition_encoder": (5*2) * 128 + 128,  # 1.4K
    
    "per_transformer_block": {
        "adaLN_1": 256 * 128 + 256 * 128,  # 65K (gamma 和 beta)
        "mha_qkv": 3 * 256 * 256,  # 195K
        "mha_out": 256 * 256,  # 65K
        "adaLN_2": 256 * 128 + 256 * 128,  # 65K
        "ffn_1": 256 * 1024 + 1024,  # 263K
        "ffn_2": 1024 * 256 + 256,  # 262K
        "total_per_block": "~918K",
    },
    
    "total_transformer": "918K * 6 layers = ~5.5M",
    
    "output_head": 256 * 6 + 6,  # 1.5K
}
```

#### AdaLN 详解

AdaLN (Adaptive Layer Normalization) 是本架构的关键创新：

```python
# 标准 LayerNorm
def layer_norm(x):
    # x: [B, T, hidden_dim]
    mean = x.mean(dim=-1, keepdim=True)  # [B, T, 1]
    std = x.std(dim=-1, keepdim=True)    # [B, T, 1]
    return (x - mean) / std

# AdaLN
def adaptive_layer_norm(x, cond_emb):
    # x: [B, T, hidden_dim]
    # cond_emb: [B, cond_dim]
    
    # 生成缩放和偏移参数
    gamma = linear_gamma(cond_emb)  # [B, hidden_dim]
    beta = linear_beta(cond_emb)    # [B, hidden_dim]
    
    # 标准归一化
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    x_norm = (x - mean) / std  # [B, T, hidden_dim]
    
    # 自适应变换（广播 gamma 和 beta）
    gamma = gamma.unsqueeze(1)  # [B, 1, hidden_dim]
    beta = beta.unsqueeze(1)    # [B, 1, hidden_dim]
    
    return gamma * x_norm + beta  # [B, T, hidden_dim]

# 优点：
# 1. 时间和条件相关：gamma 和 beta 由 cond_emb 控制
# 2. 位置无关：对序列中的每一步应用相同的 gamma/beta
# 3. 增强条件信息的影响：直接调节各层的激活
```

#### 注意力机制详解

```python
# Multi-Head Self-Attention

def multi_head_attention(x, num_heads=8):
    # x: [B, T, hidden_dim]
    B, T, D = x.shape
    head_dim = D // num_heads
    
    # 1. 线性投影生成 Q, K, V
    Q = linear_q(x)  # [B, T, D]
    K = linear_k(x)  # [B, T, D]
    V = linear_v(x)  # [B, T, D]
    
    # 2. 分头
    # [B, T, D] → [B, T, num_heads, head_dim] → [B, num_heads, T, head_dim]
    Q = Q.view(B, T, num_heads, head_dim).transpose(1, 2)
    K = K.view(B, T, num_heads, head_dim).transpose(1, 2)
    V = V.view(B, T, num_heads, head_dim).transpose(1, 2)
    
    # 3. 计算注意力
    scores = Q @ K.transpose(-2, -1) / sqrt(head_dim)  # [B, num_heads, T, T]
    attention = softmax(scores, dim=-1)  # [B, num_heads, T, T]
    
    # 4. 应用到值
    output = attention @ V  # [B, num_heads, T, head_dim]
    
    # 5. 合并多头
    output = output.transpose(1, 2).contiguous()  # [B, T, num_heads, head_dim]
    output = output.view(B, T, D)  # [B, T, D]
    
    # 6. 最后的线性投影
    return linear_out(output)  # [B, T, D]

# 为什么分头？
# 1. 允许模型在不同的表示子空间中关注不同类型的信息
# 2. 8 个头各自学习不同的注意力模式
# 3. 增加模型容量和表达力
```

### 2.4 数据格式说明

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

#### 完整模型前向传播张量信息流 (Base 配置)

**参数设定**：B=32, T=64, D=2, hidden_dim=256, num_heads=8, num_layers=6

```
┌────────────────────────────────────────────────────────────────────────┐
│               模型前向传播中的张量维度变化 (Base Model)                  │
└────────────────────────────────────────────────────────────────────────┘

【阶段 0】输入处理
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  输入条件张量:                                              │
│  ├─ x_t [32, 64, 6]             ← 拼接的状态向量          │
│  ├─ t [32]                       ← 流时间标量              │
│  ├─ start_pos [32, 2]           ← 起始位置                │
│  ├─ goal_pos [32, 2]            ← 目标位置                │
│  └─ start_vel [32, 2]           ← 起始速度                │
│                                                             │
│  ├─ 输入投影:                                               │
│  │  Linear(6, 256)                                          │
│  │  [32, 64, 6] → [32, 64, 256]                            │
│  │  x_embed                                                 │
│  │                                                           │
│  ├─ 位置编码:                                               │
│  │  PE [64, 256]                                            │
│  │  x_embed = x_embed + PE  (广播: [32, 64, 256])          │
│  │                                                           │
│  ├─ 时间嵌入:                                               │
│  │  Gaussian Fourier: t [32] → [32, 128]                   │
│  │  Linear(128, 128): [32, 128] → [32, 128]                │
│  │  time_emb [32, 128]                                      │
│  │                                                           │
│  ├─ 条件编码:                                               │
│  │  concat([start_pos, goal_pos, start_vel]) [32, 6]      │
│  │  Linear(6, 128): [32, 6] → [32, 128]                    │
│  │  cond_emb [32, 128]                                      │
│  │                                                           │
│  ├─ 条件融合:                                               │
│  │  combined_cond = time_emb + cond_emb                     │
│  │  combined_cond [32, 128]                                 │
│  │                                                           │
│  └─ 阶段 0 输出: [32, 64, 256]  combined_cond [32, 128]    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

【阶段 1】第 1 个 Transformer Block (Block 0)
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  输入: x [32, 64, 256]  条件: combined_cond [32, 128]     │
│                                                             │
│  ┌─ Sub-Block 1A: AdaLN                                    │
│  │  ├─ γ = Linear_gamma(combined_cond) [32, 128] → [32, 256]
│  │  ├─ β = Linear_beta(combined_cond) [32, 128] → [32, 256] │
│  │  ├─ x_norm = LayerNorm(x) [32, 64, 256]                │
│  │  ├─ γ.unsqueeze(1) [32, 1, 256] (广播)                 │
│  │  ├─ β.unsqueeze(1) [32, 1, 256]                         │
│  │  └─ x_ada = γ * x_norm + β [32, 64, 256]               │
│  │      输出: [32, 64, 256]                                 │
│  │                                                           │
│  ├─ Sub-Block 1B: Multi-Head Self-Attention                │
│  │  ├─ Linear_q(x_ada) [32, 64, 256]                       │
│  │  ├─ Linear_k(x_ada) [32, 64, 256]                       │
│  │  ├─ Linear_v(x_ada) [32, 64, 256]                       │
│  │  │                                                        │
│  │  ├─ 分头:                                                 │
│  │  │  ├─ Q [32, 64, 256] → [32, 8, 64, 32]              │
│  │  │  ├─ K [32, 64, 256] → [32, 8, 64, 32]              │
│  │  │  └─ V [32, 64, 256] → [32, 8, 64, 32]              │
│  │  │                                                        │
│  │  ├─ 注意力计算:                                           │
│  │  │  ├─ scores = Q @ K^T / √32                           │
│  │  │  │  [32, 8, 64, 32] @ [32, 8, 32, 64] = [32, 8, 64, 64]
│  │  │  ├─ weights = softmax(scores) [32, 8, 64, 64]       │
│  │  │  └─ attn_out = weights @ V [32, 8, 64, 32]         │
│  │  │                                                        │
│  │  ├─ 合并多头:                                             │
│  │  │  └─ [32, 8, 64, 32] → [32, 64, 256]                 │
│  │  │                                                        │
│  │  ├─ 输出投影:                                             │
│  │  │  └─ Linear_out [32, 64, 256]                         │
│  │  │                                                        │
│  │  └─ 残差连接: x_attn = x_ada + attn_out [32, 64, 256] │
│  │                                                           │
│  ├─ Sub-Block 1C: AdaLN                                    │
│  │  └─ 同 Sub-Block 1A 的过程                              │
│  │      输出: [32, 64, 256]                                 │
│  │                                                           │
│  └─ Sub-Block 1D: Feed-Forward Network                     │
│     ├─ Linear1(x_ada2) [32, 64, 256] → [32, 64, 1024]     │
│     ├─ ReLU() [32, 64, 1024]                               │
│     ├─ Linear2() [32, 64, 1024] → [32, 64, 256]           │
│     ├─ 残差连接: x_ffn = x_ada2 + ffn_out [32, 64, 256]   │
│     └─ Block 0 输出: [32, 64, 256]                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

【阶段 2-6】Block 1, 2, 3, 4, 5 (过程完全相同)
│
├─ 每个 Block 的输出维度: [32, 64, 256]
│
└─ Block 5 的输出: [32, 64, 256]

【阶段 7】最后的 AdaLN 和输出头
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  输入: x [32, 64, 256]  条件: combined_cond [32, 128]     │
│                                                             │
│  ├─ 最后 AdaLN:                                             │
│  │  └─ 同前面的 AdaLN 过程                                  │
│  │      输出: [32, 64, 256]                                 │
│  │                                                           │
│  ├─ 输出投影:                                               │
│  │  Linear(256, 6)                                          │
│  │  [32, 64, 256] → [32, 64, 6]                            │
│  │  raw_output [32, 64, 6]                                  │
│  │                                                           │
│  ├─ 分解向量场:                                             │
│  │  ├─ u_pred = raw_output[..., :2]         [32, 64, 2]   │
│  │  ├─ v_pred = raw_output[..., 2:4]        [32, 64, 2]   │
│  │  └─ w_pred = raw_output[..., 4:6]        [32, 64, 2]   │
│  │                                                           │
│  └─ 最终输出:                                               │
│     {                                                        │
│         'output': [32, 64, 6],    # 完整向量场              │
│         'u_pred': [32, 64, 2],    # 位置速度场              │
│         'v_pred': [32, 64, 2],    # 加速度场                │
│         'w_pred': [32, 64, 2],    # 急动场                  │
│     }                                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

【内存占用估计】(假设 float32)

输入: 32 * 64 * 6 * 4 bytes = 49.2 KB
中间激活 (最大): 32 * 64 * 256 * 4 bytes = 2.1 MB
FFN 中间: 32 * 64 * 1024 * 4 bytes = 8.4 MB
注意力矩阵: 32 * 8 * 64 * 64 * 4 bytes = 4.1 MB
总计激活内存: ~15-20 MB (不计梯度)

【计算复杂度】(FLOPs 估计, Base 模型)

输入投影: 32 * 64 * 6 * 256 = 3.1M FLOPs
位置编码: 0 (查表)
时间嵌入: 32 * 256 * 256 ≈ 2.1M FLOPs
条件编码: 32 * 10 * 128 ≈ 40K FLOPs

每个 Transformer Block:
  ├─ AdaLN (×2): 32 * 64 * 256 * 2 ≈ 1.0M FLOPs
  ├─ 注意力 (QKV): 32 * 64 * 256 * 3 ≈ 1.6M FLOPs
  ├─ 注意力计算: 32 * 8 * 64 * 64 * 32 ≈ 1.0M FLOPs
  ├─ 注意力输出: 32 * 8 * 64 * 64 * 32 ≈ 1.0M FLOPs
  ├─ 输出投影: 32 * 64 * 256 * 256 ≈ 0.1M FLOPs
  └─ FFN: 32 * 64 * 256 * 4 * 2 ≈ 4.2M FLOPs
  每块总计: ~10M FLOPs

6 个 Block: 10M * 6 = 60M FLOPs
输出头: 32 * 64 * 256 * 6 ≈ 31K FLOPs
总计: ~67M FLOPs (单次前向传播)
```

**实际推理时间** (RTX 3090 上的参考值):
- Base 模型, B=1: ~3-5 ms/forward pass
- Base 模型, B=32: ~30-50 ms/forward pass
- 使用 TorchScript 或 ONNX 可再快 1.5-2 倍

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

#### L2层三层架构集成 API

```python
import torch
from cfm_flowmp.models import create_l2_safety_cfm, L2Config

# ==================== L2层配置与创建 ====================

# 1. 创建L2层模型（使用Transformer后端）
config = L2Config(
    model_type="transformer",           # 或 "unet1d"
    state_dim=2,                        # 2D平面
    seq_len=64,                         # 轨迹长度
    cost_map_channels=4,                # 代价地图通道数
    cost_map_size=64,                   # 代价地图分辨率
    hidden_dim=256,                     # Transformer隐藏维度
    num_layers=8,                       # Transformer层数
    num_heads=8,                        # 注意力头数
    cond_dim=256,                       # 条件向量维度
    use_cbf=False,                      # 是否使用CBF约束
    cbf_margin=0.1,                     # CBF安全余量
)

model = create_l2_safety_cfm(config=config)
model.eval()  # 推理模式

# ==================== 从L3接收输入 ====================

# L3层输出：语义代价地图
cost_map = torch.rand(2, 4, 64, 64)  # [B=2, C=4, H=64, W=64]
# 通道含义示例：
#   channel 0: 障碍物占用概率 [0, 1]
#   channel 1: 可通行性评分 [0, 1]
#   channel 2: 风险等级 [0, 1]
#   channel 3: 语义标签编码

# 机器人当前状态 [position_x, position_y, velocity_x, velocity_y, accel_x, accel_y]
x_curr = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 样本1：静止状态
    [1.0, 1.0, 0.5, 0.3, 0.0, 0.0],  # 样本2：运动状态
])  # [B=2, 6]

# 目标状态 [position_x, position_y, velocity_x, velocity_y]
x_goal = torch.tensor([
    [5.0, 5.0, 0.0, 0.0],  # 样本1目标
    [6.0, 4.0, 0.0, 0.0],  # 样本2目标
])  # [B=2, 4]

# 控制风格权重 [w_safety, w_energy, w_smooth]
w_style = torch.tensor([
    [0.7, 0.1, 0.2],  # 样本1：保守模式（安全优先）
    [0.2, 0.5, 0.3],  # 样本2：高效模式（效率优先）
])  # [B=2, 3]

# ==================== L2层生成轨迹锚点 ====================

with torch.no_grad():
    result = model.generate_trajectory_anchors(
        cost_map=cost_map,
        x_curr=x_curr,
        x_goal=x_goal,
        w_style=w_style,
        num_samples=64,  # 每个样本生成64条候选轨迹
    )

# ==================== 输出给L1 MPPI ====================

# result 是一个字典，包含：
trajectories = result['trajectories']      # [B×N, T, 2] = [128, 64, 2]
velocities = result['velocities']          # [B×N, T, 2] = [128, 64, 2]
accelerations = result['accelerations']    # [B×N, T, 2] = [128, 64, 2]

# 对于样本0，获取其64条候选轨迹
sample_0_trajectories = trajectories[:64]   # [64, 64, 2]
sample_0_velocities = velocities[:64]       # [64, 64, 2]
sample_0_accelerations = accelerations[:64] # [64, 64, 2]

print(f"为样本0生成了 {len(sample_0_trajectories)} 条轨迹锚点")
print(f"每条轨迹包含 {sample_0_trajectories.shape[1]} 个时间步")

# ==================== L1 MPPI使用示例 (伪代码) ====================

# MPPI选择最优轨迹锚点作为warm-start
# mppi_optimizer.warm_start(sample_0_trajectories, sample_0_velocities)
# optimal_control = mppi_optimizer.optimize(current_state=x_curr[0])

# ==================== 不同控制风格效果对比 ====================

# 保守模式：w_style = [0.7, 0.1, 0.2]
# - 远离障碍物
# - 保守路径选择
# - 较低速度

# 平衡模式：w_style = [0.4, 0.3, 0.3]
# - 综合考虑安全、效率、平滑
# - 适用于大多数场景

# 高效模式：w_style = [0.2, 0.5, 0.3]
# - 最短路径
# - 较高速度
# - 激进行为
```

#### L2层训练 API

```python
import torch
from torch.utils.data import Dataset, DataLoader
from cfm_flowmp.models import create_l2_safety_cfm, L2Config
from cfm_flowmp.training import FlowMatchingConfig, CFMTrainer, TrainerConfig

# ==================== 自定义L2数据集 ====================

class L2TrajectoryDataset(Dataset):
    """L2层训练数据集，包含代价地图"""
    
    def __init__(self, data_path):
        data = torch.load(data_path)
        
        # 必需字段
        self.trajectories = data['trajectories']      # [N, T, 2]
        self.velocities = data['velocities']          # [N, T, 2]
        self.accelerations = data['accelerations']    # [N, T, 2]
        self.cost_maps = data['cost_maps']            # [N, C, H, W]
        self.start_states = data['start_states']      # [N, 6]
        self.goal_states = data['goal_states']        # [N, 4]
        self.style_weights = data['style_weights']    # [N, 3]
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return {
            'positions': self.trajectories[idx],      # [T, 2]
            'velocities': self.velocities[idx],       # [T, 2]
            'accelerations': self.accelerations[idx], # [T, 2]
            'cost_map': self.cost_maps[idx],          # [C, H, W]
            'start_state': self.start_states[idx],    # [6]
            'goal_state': self.goal_states[idx],      # [4]
            'style_weight': self.style_weights[idx],  # [3]
        }

# ==================== 创建数据加载器 ====================

dataset = L2TrajectoryDataset('l2_training_data.pt')

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# ==================== 创建L2模型 ====================

config = L2Config(
    model_type="transformer",
    state_dim=2,
    seq_len=64,
    cost_map_channels=4,
    cost_map_size=64,
)

model = create_l2_safety_cfm(config=config)

# ==================== 配置训练器 ====================

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
    use_amp=True,
    flow_config=flow_config,
)

# ==================== 训练L2层 ====================

# 注意：需要自定义训练循环以处理cost_map输入
# 或扩展CFMTrainer以支持L2层的特殊输入

for epoch in range(trainer_config.num_epochs):
    model.train()
    
    for batch in train_loader:
        # 提取批次数据
        positions = batch['positions']          # [B, T, 2]
        velocities = batch['velocities']        # [B, T, 2]
        accelerations = batch['accelerations']  # [B, T, 2]
        cost_map = batch['cost_map']            # [B, C, H, W]
        start_state = batch['start_state']      # [B, 6]
        goal_state = batch['goal_state']        # [B, 4]
        style_weight = batch['style_weight']    # [B, 3]
        
        # L2层前向传播（训练模式）
        loss_dict = model.forward(
            trajectories=positions,
            velocities=velocities,
            accelerations=accelerations,
            cost_map=cost_map,
            x_curr=start_state,
            x_goal=goal_state,
            w_style=style_weight,
        )
        
        # 反向传播
        loss = loss_dict['total_loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ==================== 保存模型 ====================

torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
}, 'l2_safety_cfm.pt')
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
lr = 1e-4
```

#### 高精度训练（研究）
```python
variant = "large"
epochs = 200
batch_size = 32
lr = 5e-5
use_amp = False
```

---

## 四、常见问题与优化建议

### 4.1 训练常见问题

#### Q: 训练损失不收敛怎么办？
**A:** 检查以下几点：
1. 学习率是否过高：建议从 1e-4 开始
2. 数据是否已归一化：使用 `normalize=True`
3. 梯度是否爆炸：使用梯度裁剪 `max_grad_norm=1.0`
4. 批次大小：较大的批次 (64-128) 通常更稳定

#### Q: 显存不足怎么办？
**A:** 尝试以下方案：
1. 减少批次大小：`batch_size=32` 或更小
2. 使用梯度累积：`gradient_accumulation_steps=4`
3. 启用混合精度：`use_amp=True`
4. 使用更小的模型：`variant="small"`

#### Q: 训练速度太慢怎么办？
**A:** 优化建议：
1. 增加 `num_workers`：取决于 CPU 核心数
2. 使用更大的批次：`batch_size=128` （显存允许）
3. 启用混合精度：自动快 1.5-2 倍
4. 使用分布式训练：多 GPU 并行

### 4.2 推理常见问题

#### Q: 生成的轨迹不光滑怎么办？
**A:** 
1. 确保启用 B-spline 平滑：`use_bspline_smoothing=True`
2. 增加 ODE 步数：`num_steps=50` （更精细的积分）
3. 使用非均匀调度：`use_8step_schedule=True` （推荐）

#### Q: 推理速度过慢怎么办？
**A:**
1. 使用 8 步非均匀调度：快 3-5 倍
2. 减少序列长度：`seq_len=32`
3. 使用 TorchScript 编译模型
4. 启用 ONNX 推理加速

#### Q: 生成的轨迹与条件不符怎么办？
**A:**
1. 检查输入条件格式是否正确
2. 增加条件权重（如果支持）
3. 使用更强大的模型：`variant="large"`
4. 增加训练时间和数据量

### 4.3 性能优化建议

#### 训练优化
| 方面 | 方法 | 预期收益 |
|------|------|---------|
| 速度 | 混合精度 + 梯度累积 | 1.5-2 倍 |
| 速度 | 分布式训练 (2 GPU) | ~1.8 倍 |
| 显存 | 梯度累积 | 允许更大批次 |
| 精度 | 更大的模型 | +2-3% 性能 |
| 稳定性 | 预热学习率 | 更好的收敛 |

#### 推理优化
| 方案 | 方法 | 速度提升 | 质量损失 |
|------|------|---------|---------|
| 快速 | 8步非均匀调度 | **3-5 倍** | <1% |
| 较快 | 20步均匀调度 | 1 倍 | 0% |
| 精确 | 50步均匀调度 | -2.5 倍 | +0.5% |

### 4.4 调试技巧

```python
# 1. 验证数据格式
positions.shape  # 应为 [N, T, D]
assert positions.min() >= -10 and positions.max() <= 10  # 合理范围

# 2. 检查梯度流动
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"无梯度: {name}")

# 3. 监视损失
if loss.isnan() or loss.isinf():
    print("检测到异常损失！")
    
# 4. 可视化轨迹
import matplotlib.pyplot as plt
plt.plot(positions[0, :, 0], positions[0, :, 1])
plt.show()
```

---

## 五、扩展功能

### 5.1 自定义轨迹数据集

```python
from cfm_flowmp.data import TrajectoryDataset

# 创建自定义数据集
class CustomTrajectoryDataset(TrajectoryDataset):
    def __getitem__(self, idx):
        # 获取数据
        pos = self.positions[idx]
        
        # 自定义预处理
        pos = self.my_custom_preprocessing(pos)
        
        # 返回标准格式
        return {
            'positions': pos,
            'velocities': self.velocities[idx],
            'start_pos': pos[0],
            'goal_pos': pos[-1],
        }
    
    def my_custom_preprocessing(self, pos):
        # 自定义数据增强、变换等
        return pos
```

### 5.2 自定义损失函数

```python
from cfm_flowmp.training import FlowMatchingLoss

class WeightedFlowMatchingLoss(FlowMatchingLoss):
    def forward(self, pred, target, weights=None):
        # 基础损失
        base_loss = super().forward(pred, target)
        
        # 加权损失
        if weights is not None:
            weighted_loss = base_loss * weights
            return weighted_loss.mean()
        
        return base_loss
```

### 5.3 自定义模型输出处理

```python
from cfm_flowmp.inference import TrajectoryGenerator

class CustomTrajectoryGenerator(TrajectoryGenerator):
    def generate(self, start_pos, goal_pos, **kwargs):
        # 标准生成
        result = super().generate(start_pos, goal_pos, **kwargs)
        
        # 自定义后处理
        positions = result['positions']
        
        # 例: 应用约束
        positions = self.apply_constraints(positions)
        
        result['positions'] = positions
        return result
    
    def apply_constraints(self, positions):
        # 确保轨迹在有效范围内
        positions = torch.clamp(positions, min=-10, max=10)
        return positions
```

---

## 六、L2层架构总结

### 6.1 L2层在三层架构中的角色

```
┌────────────────────────────────────────────────────────────────┐
│  完整系统架构：VLM-CFM-MPPI三层规划                             │
└────────────────────────────────────────────────────────────────┘

输入层：传感器数据
  ├─ RGB/深度图像
  ├─ 激光雷达点云
  ├─ IMU/里程计
  └─ 语言指令（文本）
        │
        ↓
┌────────────────────────────────────────────────────────────────┐
│  L3: Vision-Language Model (语义理解层)                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  功能：场景理解、语义分割、风险评估                             │
│  输出：cost_map [B, C, H, W] - 语义代价地图                    │
└────────────────────────────────────────────────────────────────┘
        │ 语义信息
        ↓
┌────────────────────────────────────────────────────────────────┐
│  L2: Safety-Embedded CFM (轨迹生成层) ★ 本项目核心             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                 │
│  子模块：                                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ [1] CostMapEncoder                                      │  │
│  │     └─ CNN: cost_map [B,C,H,W] → e_map [B, 256]       │  │
│  └─────────────────────────────────────────────────────────┘  │
│          │                                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ [2] ConditionFusion                                     │  │
│  │     └─ 融合: e_map + x_curr + x_goal + w_style         │  │
│  │     └─ 输出: condition [B, cond_dim]                   │  │
│  └─────────────────────────────────────────────────────────┘  │
│          │                                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ [3] FlowMPTransformer / FlowMPUNet1D                   │  │
│  │     └─ 条件向量场预测: v(x, t | condition)             │  │
│  └─────────────────────────────────────────────────────────┘  │
│          │                                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ [4] ODE Solver (RK4)                                    │  │
│  │     └─ 求解: dx/dt = v(x, t) from noise → trajectory  │  │
│  └─────────────────────────────────────────────────────────┘  │
│          │                                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ [5] Multi-Modal Sampling                                │  │
│  │     └─ 生成 N=64 条候选轨迹（带完整动力学）             │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  输出：trajectory_anchors                                       │
│    ├─ trajectories [B×64, T, 2]                                │
│    ├─ velocities [B×64, T, 2]                                  │
│    └─ accelerations [B×64, T, 2]                               │
└────────────────────────────────────────────────────────────────┘
        │ 多模态轨迹锚点
        ↓
┌────────────────────────────────────────────────────────────────┐
│  L1: MPPI (实时优化控制层)                                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  功能：局部采样优化、实时控制                                   │
│  输入：从64条锚点中选择/组合                                     │
│  输出：u_optimal - 最优控制序列                                 │
└────────────────────────────────────────────────────────────────┘
        │ 控制指令
        ↓
执行层：机器人控制器
  └─ 电机/舵机执行
```

### 6.2 L2层关键特性

#### 多模态生成能力
```
单个条件 → 64条不同轨迹
  ├─ 探索多种可行路径
  ├─ 适应不同风险偏好
  ├─ 增强鲁棒性
  └─ 为L1 MPPI提供丰富的warm-start选项
```

#### 控制风格调节
```
w_style = [w_safety, w_energy, w_smooth]

┌──────────────┬─────────────────────────────────────────┐
│  权重配置    │  行为特征                                │
├──────────────┼─────────────────────────────────────────┤
│ [0.7,0.1,0.2]│ 保守模式：远离障碍，低速，安全优先      │
│ [0.4,0.3,0.3]│ 平衡模式：综合考虑安全、效率、舒适      │
│ [0.2,0.5,0.3]│ 高效模式：最短路径，高速，效率优先      │
│ [0.3,0.2,0.5]│ 舒适模式：平滑轨迹，减少急动，乘客舒适  │
└──────────────┴─────────────────────────────────────────┘
```

#### 完整动力学输出
```
每条轨迹包含：
  ├─ Position [T, 2]     - 位置轨迹
  ├─ Velocity [T, 2]     - 速度分布（一阶导数）
  └─ Acceleration [T, 2] - 加速度分布（二阶导数）

优势：
  ✓ MPPI可直接使用动力学信息
  ✓ 无需数值微分（避免噪声）
  ✓ 物理一致性保证
  ✓ 便于动力学约束检查
```

### 6.3 L2层数据接口规范

#### 输入接口（从L3接收）

```python
# 必需输入
cost_map: torch.Tensor      # [B, C, H, W] 语义代价地图
  - dtype: torch.float32
  - 值范围: [0, 1] 归一化
  - 通道含义:
    * channel 0: 障碍物占用概率
    * channel 1: 可通行性评分
    * channel 2: 风险等级
    * channel 3+: 其他语义信息

x_curr: torch.Tensor        # [B, 6] 当前机器人状态
  - [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
  - 单位: [m, m, m/s, m/s, m/s², m/s²]

x_goal: torch.Tensor        # [B, 4] 目标状态
  - [pos_x, pos_y, vel_x, vel_y]
  - 单位: [m, m, m/s, m/s]

w_style: torch.Tensor       # [B, 3] 控制风格权重
  - [w_safety, w_energy, w_smooth]
  - 和为1.0，非负
```

#### 输出接口（传给L1）

```python
# 输出字典
result = {
    'trajectories': torch.Tensor,    # [B×N, T, 2] 位置轨迹
    'velocities': torch.Tensor,      # [B×N, T, 2] 速度轨迹
    'accelerations': torch.Tensor,   # [B×N, T, 2] 加速度轨迹
}

# 参数说明
B: 批次大小
N: 每个样本的候选数量（默认64）
T: 轨迹长度（默认64时间步）
2: 2D空间（x, y坐标）
```

### 6.4 性能指标

#### 推理速度
```
配置: RTX 3090, batch_size=1, N=64
  ├─ Transformer (8层, 256维): ~50ms
  ├─ U-Net1D (4层): ~30ms
  └─ 含B-spline平滑: +10ms
```

#### 模型规模
```
FlowMPTransformer:
  ├─ 参数量: ~12M (base配置)
  ├─ 显存占用: ~500MB (推理)
  └─ 训练显存: ~4GB (batch_size=32)

CostMapEncoder:
  ├─ 参数量: ~2M
  └─ 显存占用: ~100MB
```

### 6.5 L2层使用建议

#### 训练数据准备
```
推荐数据规模:
  ├─ 最小: 5,000 轨迹样本（合成数据验证）
  ├─ 基础: 50,000 轨迹样本（真实场景训练）
  └─ 生产: 500,000+ 轨迹样本（复杂环境）

数据多样性:
  ✓ 不同场景类型（室内、室外、混合）
  ✓ 多种障碍物配置
  ✓ 不同控制风格的轨迹
  ✓ 边界情况和困难场景
```

#### 超参数调优
```
关键超参数:
  ├─ learning_rate: 1e-4 (Adam)
  ├─ num_layers: 6-8 (Transformer)
  ├─ hidden_dim: 256-512
  ├─ num_samples: 32-128 (推理时)
  └─ lambda_vel, lambda_acc, lambda_jerk: 均为1.0
```

#### 部署建议
```
实时系统:
  ├─ 使用8步非均匀ODE调度（快速）
  ├─ 批处理多个目标点
  ├─ GPU推理（CUDA）
  ├─ 可选：TensorRT优化
  └─ 异步生成与MPPI优化

离线规划:
  ├─ 使用50步均匀调度（精确）
  ├─ 启用B-spline平滑
  └─ 生成更多候选（N=128）
```

---

## 七、参考资源

### 论文和理论

1. **Flow Matching**: "Flow Matching for Generative Modeling" (2023)
   - 基础理论框架

2. **Diffusion Models**: "Denoising Diffusion Probabilistic Models" (2020)
   - 生成模型基础

3. **Transformers**: "Attention is All You Need" (2017)
   - 模型架构基础

### 相关项目

- [Score-Based Generative Models](https://github.com/yang-song/score_sde)
- [Diffusion-Transformer](https://github.com/openai/guided-diffusion)
- [Motion Matching](https://research.cs.wisc.edu/graphics/Papers/2012/Holden12/)

### 有用链接

- 官方文档：[稍后添加]
- 论文链接：[稍后添加]
- 讨论社区：[稍后添加]

---

## 七、贡献指南

我们欢迎社区的贡献！以下是如何参与的方式：

### 报告 Bug
1. 检查 GitHub Issues 是否已存在相同问题
2. 创建新 Issue，提供以下信息：
   - 复现步骤
   - 预期行为 vs 实际行为
   - Python/PyTorch 版本
   - 错误日志

### 提出功能请求
1. 描述新功能的用途
2. 提供简要的实现思路
3. 讨论可能的 API 设计

### 代码贡献
1. Fork 仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 遵循代码风格 (见下)
4. 提交 PR 并描述改动

### 代码风格指南

```python
# 类的定义
class MyModel(torch.nn.Module):
    """简洁的类文档字符串"""
    
    def __init__(self, config: Dict):
        super().__init__()
        # 初始化代码
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """函数文档说明参数和返回值"""
        return x

# 函数命名：snake_case
def compute_flow_matching_loss(pred, target) -> torch.Tensor:
    pass

# 常数：UPPER_CASE
DEFAULT_LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 512
```

---

## 八、许可和引用

### 许可

本项目采用 [MIT License](LICENSE)

### 引用

如果使用本项目，请在论文中引用：

```bibtex
@misc{cfm-flowmp-2026,
  title={CFM FlowMP: Conditional Flow Matching for Trajectory Planning},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/your-repo}}
}
```

---

**最后更新**: 2026 年 1 月 31 日  
**文档版本**: 2.0  
**项目版本**: v1.0.0

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

**最后更新**: 2026 年 1 月 31 日  
**文档版本**: 2.0  
**项目版本**: v1.0.0