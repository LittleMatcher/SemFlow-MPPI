# CFM FlowMP 数据管道详细文档 (Data Pipeline Documentation)

本文档详细描述了 CFM FlowMP 训练框架的完整数据管道，包括数据格式、预处理、B样条拟合、数据增强和数据加载。

---

## 目录 (Table of Contents)

1. [数据管道概述](#1-数据管道概述)
2. [专家轨迹数据格式](#2-专家轨迹数据格式)
3. [数据预处理流程](#3-数据预处理流程)
4. [B样条轨迹处理](#4-b样条轨迹处理)
5. [导数计算方法](#5-导数计算方法)
6. [数据归一化策略](#6-数据归一化策略)
7. [数据增强技术](#7-数据增强技术)
8. [Dataset实现](#8-dataset实现)
9. [DataLoader配置](#9-dataloader配置)
10. [与训练流程的集成](#10-与训练流程的集成)
11. [代码实现](#11-代码实现)

---

## 1. 数据管道概述

### 1.1 管道架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CFM FlowMP 数据管道                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1: 原始数据采集                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ 仿真轨迹    │  │ 人工演示    │  │ 优化求解    │  │ 真实采集    │    │
│  │ (MPC/RRT)   │  │ (遥操作)    │  │ (CHOMP等)   │  │ (传感器)    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 2: 数据预处理                                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  2.1 轨迹重采样 → 2.2 B样条拟合 → 2.3 导数计算 → 2.4 异常过滤   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 3: 数据增强                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ 几何变换    │  │ 时间扰动    │  │ 噪声注入    │  │ 轨迹拼接    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 4: 数据归一化与打包                                               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  4.1 位置归一化 → 4.2 速度归一化 → 4.3 加速度归一化 → 4.4 条件编码│  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 5: Dataset & DataLoader                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  TrajectoryDataset → DataLoader → Training Loop                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流说明

| 阶段 | 输入 | 输出 | 关键操作 |
|------|------|------|----------|
| Stage 1 | 环境配置 | 原始轨迹点 | 轨迹规划/采集 |
| Stage 2 | 原始轨迹点 | 平滑轨迹+导数 | B样条拟合 |
| Stage 3 | 平滑轨迹 | 增强后的轨迹集 | 几何/时间变换 |
| Stage 4 | 增强轨迹 | 归一化张量 | 标准化处理 |
| Stage 5 | 归一化张量 | Mini-batch | 批量采样 |

---

## 2. 专家轨迹数据格式

### 2.1 基本数据结构

FlowMP 需要的专家轨迹包含位置、速度、加速度三个分量：

```python
# 单条轨迹数据结构
trajectory = {
    'positions': np.ndarray,      # [T, D] 位置序列
    'velocities': np.ndarray,     # [T, D] 速度序列
    'accelerations': np.ndarray,  # [T, D] 加速度序列
    'timestamps': np.ndarray,     # [T] 时间戳（可选）
    'metadata': {                 # 元数据
        'start_pos': np.ndarray,  # [D] 起始位置
        'goal_pos': np.ndarray,   # [D] 目标位置
        'start_vel': np.ndarray,  # [D] 起始速度
        'duration': float,        # 轨迹总时长
        'source': str,            # 数据来源
    }
}
```

### 2.2 维度说明

| 符号 | 含义 | 典型值 |
|------|------|--------|
| T | 轨迹时间步数 | 64, 128, 256 |
| D | 状态空间维度 | 2 (2D), 3 (3D), 7 (7-DOF机械臂) |
| N | 轨迹数量 | 1000 ~ 100000 |
| B | 批次大小 | 32, 64, 128 |

### 2.3 支持的数据格式

```python
# 格式 1: NumPy NPZ 文件（推荐）
# 文件结构: data.npz
{
    'positions': np.ndarray,      # [N, T, D]
    'velocities': np.ndarray,     # [N, T, D]  
    'accelerations': np.ndarray,  # [N, T, D]
}

# 格式 2: HDF5 文件（大规模数据）
# 文件结构: data.h5
/trajectories/
    /positions        # Dataset [N, T, D]
    /velocities       # Dataset [N, T, D]
    /accelerations    # Dataset [N, T, D]
/metadata/
    /start_positions  # Dataset [N, D]
    /goal_positions   # Dataset [N, D]

# 格式 3: 分离文件
trajectories/
    ├── traj_0000.npy  # 单条轨迹 [T, D*3]
    ├── traj_0001.npy
    └── ...
```

### 2.4 数据质量要求

| 检查项 | 要求 | 处理方式 |
|--------|------|----------|
| 轨迹长度 | T ≥ 16 | 过短轨迹插值或丢弃 |
| 数值范围 | 无 NaN/Inf | 过滤异常轨迹 |
| 物理一致性 | v ≈ dq/dt | 重新计算或丢弃 |
| 动力学约束 | |a| < a_max | 过滤超限轨迹 |

---

## 3. 数据预处理流程

### 3.1 轨迹重采样

将不同长度的轨迹统一采样到固定长度 T：

```python
def resample_trajectory(
    positions: np.ndarray,      # [T_orig, D]
    target_length: int = 64,
    method: str = 'linear'      # 'linear', 'cubic', 'bspline'
) -> np.ndarray:
    """
    将轨迹重采样到目标长度。
    
    Args:
        positions: 原始位置序列
        target_length: 目标长度
        method: 插值方法
        
    Returns:
        重采样后的位置序列 [target_length, D]
    """
    from scipy.interpolate import interp1d
    
    T_orig, D = positions.shape
    
    # 原始参数化
    t_orig = np.linspace(0, 1, T_orig)
    # 目标参数化
    t_target = np.linspace(0, 1, target_length)
    
    # 逐维度插值
    resampled = np.zeros((target_length, D))
    for d in range(D):
        if method == 'linear':
            f = interp1d(t_orig, positions[:, d], kind='linear')
        elif method == 'cubic':
            f = interp1d(t_orig, positions[:, d], kind='cubic')
        elif method == 'bspline':
            from scipy.interpolate import splrep, splev
            tck = splrep(t_orig, positions[:, d], k=3, s=0)
            resampled[:, d] = splev(t_target, tck)
            continue
        resampled[:, d] = f(t_target)
    
    return resampled
```

### 3.2 异常值过滤

```python
def filter_trajectories(
    trajectories: List[Dict],
    max_velocity: float = 10.0,
    max_acceleration: float = 50.0,
    min_length: int = 16,
    max_jerk: float = 500.0,
) -> List[Dict]:
    """
    过滤不符合物理约束的轨迹。
    
    Args:
        trajectories: 轨迹列表
        max_velocity: 最大允许速度
        max_acceleration: 最大允许加速度
        min_length: 最小轨迹长度
        max_jerk: 最大允许加加速度
        
    Returns:
        过滤后的轨迹列表
    """
    filtered = []
    stats = {'total': len(trajectories), 'removed': 0, 'reasons': {}}
    
    for traj in trajectories:
        pos = traj['positions']
        vel = traj.get('velocities')
        acc = traj.get('accelerations')
        
        # 检查长度
        if len(pos) < min_length:
            stats['removed'] += 1
            stats['reasons']['too_short'] = stats['reasons'].get('too_short', 0) + 1
            continue
        
        # 检查 NaN/Inf
        if np.any(~np.isfinite(pos)):
            stats['removed'] += 1
            stats['reasons']['nan_inf'] = stats['reasons'].get('nan_inf', 0) + 1
            continue
        
        # 检查速度约束
        if vel is not None:
            vel_norm = np.linalg.norm(vel, axis=-1)
            if np.any(vel_norm > max_velocity):
                stats['removed'] += 1
                stats['reasons']['velocity'] = stats['reasons'].get('velocity', 0) + 1
                continue
        
        # 检查加速度约束
        if acc is not None:
            acc_norm = np.linalg.norm(acc, axis=-1)
            if np.any(acc_norm > max_acceleration):
                stats['removed'] += 1
                stats['reasons']['acceleration'] = stats['reasons'].get('acceleration', 0) + 1
                continue
        
        # 检查加加速度约束
        if acc is not None:
            jerk = np.diff(acc, axis=0)
            jerk_norm = np.linalg.norm(jerk, axis=-1)
            if np.any(jerk_norm > max_jerk):
                stats['removed'] += 1
                stats['reasons']['jerk'] = stats['reasons'].get('jerk', 0) + 1
                continue
        
        filtered.append(traj)
    
    print(f"Filtered {stats['removed']}/{stats['total']} trajectories")
    print(f"Removal reasons: {stats['reasons']}")
    
    return filtered
```

---

## 4. B样条轨迹处理

### 4.1 B样条拟合原理

B样条（B-Spline）提供了一种平滑的轨迹表示方式，具有以下优点：

1. **连续性保证**：k阶B样条保证 C^(k-1) 连续性
2. **局部控制**：修改控制点只影响局部曲线
3. **解析导数**：可直接计算任意阶导数

### 4.2 B样条拟合实现

```python
import numpy as np
from scipy.interpolate import splprep, splev
from typing import Tuple, Dict

class BSplineTrajectoryProcessor:
    """
    B样条轨迹处理器
    
    用于将离散轨迹点拟合为B样条曲线，并计算解析导数。
    这是 FlowMP 数据管道的核心组件。
    
    参考: https://github.com/mkhangg/flow_mp
    """
    
    def __init__(
        self,
        degree: int = 3,           # B样条阶数（3=三次）
        smoothing: float = 0.0,    # 平滑因子（0=插值）
        num_eval_points: int = 64, # 输出点数
    ):
        """
        Args:
            degree: B样条阶数，通常使用3（三次B样条）
            smoothing: 平滑因子，0表示插值，>0表示平滑拟合
            num_eval_points: 重采样输出的点数
        """
        self.degree = degree
        self.smoothing = smoothing
        self.num_eval_points = num_eval_points
    
    def fit_trajectory(
        self,
        positions: np.ndarray,  # [T, D]
        dt: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """
        将离散轨迹拟合为B样条并计算导数。
        
        Args:
            positions: 离散位置点 [T, D]
            dt: 时间步长（用于导数缩放）
            
        Returns:
            包含 positions, velocities, accelerations 的字典
        """
        T, D = positions.shape
        
        # 参数化：使用累积弧长或均匀参数
        u_original = np.linspace(0, 1, T)
        u_eval = np.linspace(0, 1, self.num_eval_points)
        
        # 转置用于 splprep: [D, T]
        positions_T = positions.T
        
        try:
            # 拟合B样条
            # splprep 返回 (tck, u)，其中 tck = (t, c, k)
            # t: 节点向量, c: 控制点, k: 阶数
            tck, u = splprep(
                positions_T,
                u=u_original,
                k=self.degree,
                s=self.smoothing,
            )
            
            # 计算位置（0阶导数）
            pos_eval = np.array(splev(u_eval, tck, der=0)).T  # [T_out, D]
            
            # 计算速度（1阶导数）
            vel_eval = np.array(splev(u_eval, tck, der=1)).T  # [T_out, D]
            
            # 计算加速度（2阶导数）
            acc_eval = np.array(splev(u_eval, tck, der=2)).T  # [T_out, D]
            
            # 缩放导数（从参数空间到时间空间）
            # du/dt = 1 / (T * dt) 对于均匀参数化
            duration = T * dt
            scale_vel = 1.0 / duration
            scale_acc = scale_vel ** 2
            
            vel_eval = vel_eval * scale_vel
            acc_eval = acc_eval * scale_acc
            
            return {
                'positions': pos_eval,
                'velocities': vel_eval,
                'accelerations': acc_eval,
                'tck': tck,  # 保存B样条参数
            }
            
        except Exception as e:
            print(f"B-spline fitting failed: {e}")
            # 回退到数值导数
            return self._numerical_derivatives(positions, dt)
    
    def _numerical_derivatives(
        self,
        positions: np.ndarray,
        dt: float,
    ) -> Dict[str, np.ndarray]:
        """
        使用数值方法计算导数（回退方案）。
        """
        # 重采样位置
        pos_resampled = resample_trajectory(
            positions, 
            self.num_eval_points, 
            method='cubic'
        )
        
        # 数值导数
        velocities = np.gradient(pos_resampled, dt, axis=0)
        accelerations = np.gradient(velocities, dt, axis=0)
        
        return {
            'positions': pos_resampled,
            'velocities': velocities,
            'accelerations': accelerations,
        }
    
    def evaluate_at_time(
        self,
        tck: Tuple,
        t: float,
        duration: float,
    ) -> Dict[str, np.ndarray]:
        """
        在指定时间点评估B样条。
        
        Args:
            tck: B样条参数
            t: 评估时间 [0, duration]
            duration: 轨迹总时长
            
        Returns:
            该时间点的 position, velocity, acceleration
        """
        u = t / duration
        u = np.clip(u, 0, 1)
        
        pos = np.array(splev(u, tck, der=0))
        vel = np.array(splev(u, tck, der=1)) / duration
        acc = np.array(splev(u, tck, der=2)) / (duration ** 2)
        
        return {
            'position': pos,
            'velocity': vel,
            'acceleration': acc,
        }


def process_trajectory_batch(
    raw_trajectories: List[np.ndarray],
    processor: BSplineTrajectoryProcessor,
    dt: float = 0.1,
    num_workers: int = 4,
) -> Dict[str, np.ndarray]:
    """
    批量处理轨迹。
    
    Args:
        raw_trajectories: 原始轨迹列表，每个为 [T_i, D]
        processor: B样条处理器
        dt: 时间步长
        num_workers: 并行处理的工作线程数
        
    Returns:
        处理后的轨迹数组字典
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    results = []
    
    def process_single(traj):
        return processor.fit_trajectory(traj, dt)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single, traj): i 
                   for i, traj in enumerate(raw_trajectories)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                results.append((futures[future], result))
            except Exception as e:
                print(f"Error processing trajectory: {e}")
    
    # 按原始顺序排序
    results.sort(key=lambda x: x[0])
    results = [r[1] for r in results]
    
    # 堆叠成数组
    positions = np.stack([r['positions'] for r in results], axis=0)
    velocities = np.stack([r['velocities'] for r in results], axis=0)
    accelerations = np.stack([r['accelerations'] for r in results], axis=0)
    
    return {
        'positions': positions,      # [N, T, D]
        'velocities': velocities,    # [N, T, D]
        'accelerations': accelerations,  # [N, T, D]
    }
```

### 4.3 B样条参数选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| degree | 3 | 三次B样条，保证 C² 连续 |
| smoothing | 0 ~ T/10 | 0=插值，较大值=更平滑 |
| num_eval_points | 64 | 输出轨迹长度 |

---

## 5. 导数计算方法

### 5.1 解析导数（推荐）

通过B样条的解析导数公式计算：

```python
def bspline_derivative(tck, u, der=1):
    """
    计算B样条的解析导数。
    
    对于k阶B样条，导数是k-1阶B样条：
    B'_{i,k}(u) = k * (B_{i,k-1}(u)/(t_{i+k}-t_i) - B_{i+1,k-1}(u)/(t_{i+k+1}-t_{i+1}))
    """
    from scipy.interpolate import splev
    return splev(u, tck, der=der)
```

### 5.2 数值导数（回退）

当B样条拟合失败时使用数值方法：

```python
def numerical_derivatives(positions: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用中心差分计算数值导数。
    
    Args:
        positions: [T, D] 位置序列
        dt: 时间步长
        
    Returns:
        velocities: [T, D] 速度序列
        accelerations: [T, D] 加速度序列
    """
    T, D = positions.shape
    
    # 速度：中心差分
    velocities = np.zeros_like(positions)
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    velocities[0] = (positions[1] - positions[0]) / dt
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    
    # 加速度：二阶中心差分
    accelerations = np.zeros_like(positions)
    accelerations[1:-1] = (positions[2:] - 2*positions[1:-1] + positions[:-2]) / (dt**2)
    accelerations[0] = accelerations[1]
    accelerations[-1] = accelerations[-2]
    
    return velocities, accelerations


def savitzky_golay_derivatives(
    positions: np.ndarray,
    dt: float,
    window_length: int = 7,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Savitzky-Golay滤波器计算平滑导数。
    
    优点：对噪声更鲁棒
    """
    from scipy.signal import savgol_filter
    
    # 确保window_length为奇数且小于序列长度
    T = positions.shape[0]
    window_length = min(window_length, T - 1)
    if window_length % 2 == 0:
        window_length -= 1
    
    velocities = savgol_filter(positions, window_length, polyorder, deriv=1, delta=dt, axis=0)
    accelerations = savgol_filter(positions, window_length, polyorder, deriv=2, delta=dt, axis=0)
    
    return velocities, accelerations
```

### 5.3 导数验证

```python
def validate_derivatives(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    dt: float,
    tolerance: float = 0.1,
) -> Dict[str, float]:
    """
    验证导数的物理一致性。
    
    通过数值积分验证：
    - 速度积分应近似位置差
    - 加速度积分应近似速度差
    """
    T = positions.shape[0]
    
    # 位置一致性：∫v dt ≈ Δq
    pos_from_vel = positions[0] + np.cumsum(velocities[:-1] * dt, axis=0)
    pos_error = np.mean(np.linalg.norm(positions[1:] - pos_from_vel, axis=-1))
    
    # 速度一致性：∫a dt ≈ Δv  
    vel_from_acc = velocities[0] + np.cumsum(accelerations[:-1] * dt, axis=0)
    vel_error = np.mean(np.linalg.norm(velocities[1:] - vel_from_acc, axis=-1))
    
    return {
        'position_consistency_error': pos_error,
        'velocity_consistency_error': vel_error,
        'is_valid': pos_error < tolerance and vel_error < tolerance,
    }
```

---

## 6. 数据归一化策略

### 6.1 归一化方法

FlowMP 使用逐维度标准化（Z-score normalization）：

```python
class TrajectoryNormalizer:
    """
    轨迹数据归一化器。
    
    对位置、速度、加速度分别进行归一化：
    x_norm = (x - mean) / std
    """
    
    def __init__(self):
        self.stats = {}
    
    def fit(self, data: Dict[str, np.ndarray]):
        """
        计算归一化统计量。
        
        Args:
            data: 包含 'positions', 'velocities', 'accelerations' 的字典
                  每个数组形状为 [N, T, D]
        """
        for key in ['positions', 'velocities', 'accelerations']:
            if key in data:
                arr = data[key]
                # 沿 N 和 T 维度计算统计量
                self.stats[f'{key}_mean'] = arr.mean(axis=(0, 1))  # [D]
                self.stats[f'{key}_std'] = arr.std(axis=(0, 1))    # [D]
                # 防止除零
                self.stats[f'{key}_std'] = np.maximum(self.stats[f'{key}_std'], 1e-6)
    
    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        应用归一化。
        """
        normalized = {}
        for key in ['positions', 'velocities', 'accelerations']:
            if key in data:
                mean = self.stats[f'{key}_mean']
                std = self.stats[f'{key}_std']
                normalized[key] = (data[key] - mean) / std
        return normalized
    
    def inverse_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        反归一化。
        """
        denormalized = {}
        for key in ['positions', 'velocities', 'accelerations']:
            if key in data:
                mean = self.stats[f'{key}_mean']
                std = self.stats[f'{key}_std']
                denormalized[key] = data[key] * std + mean
        return denormalized
    
    def save(self, path: str):
        """保存归一化参数。"""
        np.savez(path, **self.stats)
    
    def load(self, path: str):
        """加载归一化参数。"""
        loaded = np.load(path)
        self.stats = {key: loaded[key] for key in loaded.files}
```

### 6.2 条件归一化

对于起点/终点条件，使用相同的位置归一化参数：

```python
def normalize_conditions(
    start_pos: np.ndarray,  # [N, D]
    goal_pos: np.ndarray,   # [N, D]
    start_vel: np.ndarray,  # [N, D]
    normalizer: TrajectoryNormalizer,
) -> Tuple[np.ndarray, ...]:
    """
    归一化条件变量。
    """
    pos_mean = normalizer.stats['positions_mean']
    pos_std = normalizer.stats['positions_std']
    vel_mean = normalizer.stats['velocities_mean']
    vel_std = normalizer.stats['velocities_std']
    
    start_pos_norm = (start_pos - pos_mean) / pos_std
    goal_pos_norm = (goal_pos - pos_mean) / pos_std
    start_vel_norm = (start_vel - vel_mean) / vel_std
    
    return start_pos_norm, goal_pos_norm, start_vel_norm
```

---

## 7. 数据增强技术

### 7.1 几何变换

```python
class GeometricAugmentation:
    """
    几何数据增强：旋转、平移、缩放、镜像。
    """
    
    def __init__(
        self,
        rotation_range: float = np.pi,      # 旋转角度范围
        translation_range: float = 1.0,     # 平移范围
        scale_range: Tuple[float, float] = (0.8, 1.2),  # 缩放范围
        flip_prob: float = 0.5,             # 镜像概率
    ):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
    
    def __call__(
        self,
        positions: np.ndarray,     # [T, D]
        velocities: np.ndarray,    # [T, D]
        accelerations: np.ndarray, # [T, D]
    ) -> Tuple[np.ndarray, ...]:
        """
        应用随机几何变换。
        """
        D = positions.shape[-1]
        
        # 随机旋转（仅2D/3D）
        if D == 2:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            R = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            positions = positions @ R.T
            velocities = velocities @ R.T
            accelerations = accelerations @ R.T
        
        # 随机平移（仅位置）
        translation = np.random.uniform(
            -self.translation_range, 
            self.translation_range, 
            size=D
        )
        positions = positions + translation
        
        # 随机缩放
        scale = np.random.uniform(*self.scale_range)
        positions = positions * scale
        velocities = velocities * scale
        accelerations = accelerations * scale
        
        # 随机镜像
        if np.random.random() < self.flip_prob:
            flip_axis = np.random.randint(D)
            positions[:, flip_axis] *= -1
            velocities[:, flip_axis] *= -1
            accelerations[:, flip_axis] *= -1
        
        return positions, velocities, accelerations


class TemporalAugmentation:
    """
    时间数据增强：时间缩放、时间反转。
    """
    
    def __init__(
        self,
        time_scale_range: Tuple[float, float] = (0.8, 1.2),
        reverse_prob: float = 0.0,  # 时间反转概率
    ):
        self.time_scale_range = time_scale_range
        self.reverse_prob = reverse_prob
    
    def __call__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """
        应用时间变换。
        
        时间缩放影响导数：
        - 速度缩放为 1/s
        - 加速度缩放为 1/s²
        """
        # 时间缩放
        scale = np.random.uniform(*self.time_scale_range)
        velocities = velocities / scale
        accelerations = accelerations / (scale ** 2)
        
        # 时间反转
        if np.random.random() < self.reverse_prob:
            positions = positions[::-1].copy()
            velocities = -velocities[::-1].copy()
            accelerations = accelerations[::-1].copy()
        
        return positions, velocities, accelerations


class NoiseAugmentation:
    """
    噪声数据增强：添加高斯噪声。
    """
    
    def __init__(
        self,
        position_noise_std: float = 0.01,
        velocity_noise_std: float = 0.05,
        acceleration_noise_std: float = 0.1,
    ):
        self.pos_std = position_noise_std
        self.vel_std = velocity_noise_std
        self.acc_std = acceleration_noise_std
    
    def __call__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """
        添加高斯噪声。
        """
        positions = positions + np.random.randn(*positions.shape) * self.pos_std
        velocities = velocities + np.random.randn(*velocities.shape) * self.vel_std
        accelerations = accelerations + np.random.randn(*accelerations.shape) * self.acc_std
        
        return positions, velocities, accelerations
```

### 7.2 组合增强

```python
class ComposedAugmentation:
    """
    组合多种数据增强。
    """
    
    def __init__(self, augmentations: List, probs: List[float] = None):
        """
        Args:
            augmentations: 增强器列表
            probs: 每种增强的应用概率
        """
        self.augmentations = augmentations
        self.probs = probs or [1.0] * len(augmentations)
    
    def __call__(self, positions, velocities, accelerations):
        for aug, prob in zip(self.augmentations, self.probs):
            if np.random.random() < prob:
                positions, velocities, accelerations = aug(
                    positions, velocities, accelerations
                )
        return positions, velocities, accelerations


# 创建标准增强流水线
def create_standard_augmentation():
    return ComposedAugmentation([
        GeometricAugmentation(
            rotation_range=np.pi,
            translation_range=0.5,
            scale_range=(0.9, 1.1),
            flip_prob=0.3,
        ),
        TemporalAugmentation(
            time_scale_range=(0.9, 1.1),
            reverse_prob=0.0,
        ),
        NoiseAugmentation(
            position_noise_std=0.005,
            velocity_noise_std=0.02,
            acceleration_noise_std=0.05,
        ),
    ], probs=[0.8, 0.5, 0.5])
```

---

## 8. Dataset实现

### 8.1 核心Dataset类

```python
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Callable
import numpy as np

class FlowMPTrajectoryDataset(Dataset):
    """
    FlowMP 轨迹数据集。
    
    提供专家轨迹数据用于 CFM 训练。
    每个样本包含：
    - positions: [T, D] 位置轨迹
    - velocities: [T, D] 速度轨迹
    - accelerations: [T, D] 加速度轨迹
    - start_pos: [D] 起始位置
    - goal_pos: [D] 目标位置
    - start_vel: [D] 起始速度
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        positions: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
        accelerations: Optional[np.ndarray] = None,
        normalizer: Optional[TrajectoryNormalizer] = None,
        augmentation: Optional[Callable] = None,
        compute_derivatives: bool = True,
        dt: float = 0.1,
    ):
        """
        Args:
            data_path: 数据文件路径 (.npz, .h5, .pkl)
            positions: 预加载的位置数据 [N, T, D]
            velocities: 预加载的速度数据 [N, T, D]
            accelerations: 预加载的加速度数据 [N, T, D]
            normalizer: 归一化器
            augmentation: 数据增强函数
            compute_derivatives: 是否从位置计算导数
            dt: 时间步长
        """
        # 加载数据
        if data_path is not None:
            data = self._load_data(data_path)
            positions = data.get('positions', positions)
            velocities = data.get('velocities', velocities)
            accelerations = data.get('accelerations', accelerations)
        
        if positions is None:
            raise ValueError("positions must be provided")
        
        self.positions = positions.astype(np.float32)
        self.dt = dt
        
        # 计算或加载导数
        if velocities is not None:
            self.velocities = velocities.astype(np.float32)
        elif compute_derivatives:
            self.velocities = self._compute_velocities(self.positions, dt)
        else:
            self.velocities = np.zeros_like(self.positions)
        
        if accelerations is not None:
            self.accelerations = accelerations.astype(np.float32)
        elif compute_derivatives:
            self.accelerations = self._compute_accelerations(self.velocities, dt)
        else:
            self.accelerations = np.zeros_like(self.positions)
        
        # 归一化
        self.normalizer = normalizer
        if normalizer is not None:
            data = {
                'positions': self.positions,
                'velocities': self.velocities,
                'accelerations': self.accelerations,
            }
            normalized = normalizer.transform(data)
            self.positions = normalized['positions']
            self.velocities = normalized['velocities']
            self.accelerations = normalized['accelerations']
        
        # 数据增强
        self.augmentation = augmentation
        
        # 数据统计
        self.num_trajectories = len(self.positions)
        self.seq_len = self.positions.shape[1]
        self.state_dim = self.positions.shape[2]
    
    def _load_data(self, path: str) -> Dict[str, np.ndarray]:
        """加载数据文件。"""
        from pathlib import Path
        path = Path(path)
        
        if path.suffix == '.npz':
            data = np.load(path)
            return {key: data[key] for key in data.files}
        
        elif path.suffix in ['.h5', '.hdf5']:
            import h5py
            with h5py.File(path, 'r') as f:
                return {key: f[key][:] for key in f.keys()}
        
        elif path.suffix == '.pkl':
            import pickle
            with open(path, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    def _compute_velocities(self, positions: np.ndarray, dt: float) -> np.ndarray:
        """计算速度（中心差分）。"""
        velocities = np.zeros_like(positions)
        velocities[:, 1:-1] = (positions[:, 2:] - positions[:, :-2]) / (2 * dt)
        velocities[:, 0] = (positions[:, 1] - positions[:, 0]) / dt
        velocities[:, -1] = (positions[:, -1] - positions[:, -2]) / dt
        return velocities
    
    def _compute_accelerations(self, velocities: np.ndarray, dt: float) -> np.ndarray:
        """计算加速度（中心差分）。"""
        return self._compute_velocities(velocities, dt)
    
    def __len__(self) -> int:
        return self.num_trajectories
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本。
        
        Returns:
            Dict containing:
                - positions: [T, D]
                - velocities: [T, D]
                - accelerations: [T, D]
                - start_pos: [D]
                - goal_pos: [D]
                - start_vel: [D]
        """
        positions = self.positions[idx].copy()
        velocities = self.velocities[idx].copy()
        accelerations = self.accelerations[idx].copy()
        
        # 数据增强
        if self.augmentation is not None and self.training:
            positions, velocities, accelerations = self.augmentation(
                positions, velocities, accelerations
            )
        
        # 提取条件
        start_pos = positions[0]
        goal_pos = positions[-1]
        start_vel = velocities[0]
        
        return {
            'positions': torch.from_numpy(positions),
            'velocities': torch.from_numpy(velocities),
            'accelerations': torch.from_numpy(accelerations),
            'start_pos': torch.from_numpy(start_pos),
            'goal_pos': torch.from_numpy(goal_pos),
            'start_vel': torch.from_numpy(start_vel),
        }
    
    def train(self):
        """设置为训练模式（启用数据增强）。"""
        self.training = True
        return self
    
    def eval(self):
        """设置为评估模式（禁用数据增强）。"""
        self.training = False
        return self


class LazyTrajectoryDataset(Dataset):
    """
    惰性加载的轨迹数据集，适用于大规模数据。
    
    数据存储在磁盘上，按需加载。
    """
    
    def __init__(
        self,
        data_dir: str,
        file_pattern: str = "traj_*.npy",
        preprocess_fn: Optional[Callable] = None,
        cache_size: int = 1000,
    ):
        """
        Args:
            data_dir: 数据目录
            file_pattern: 文件匹配模式
            preprocess_fn: 预处理函数
            cache_size: LRU缓存大小
        """
        from pathlib import Path
        import glob
        
        self.data_dir = Path(data_dir)
        self.file_list = sorted(glob.glob(str(self.data_dir / file_pattern)))
        self.preprocess_fn = preprocess_fn
        
        if len(self.file_list) == 0:
            raise ValueError(f"No files found matching {file_pattern} in {data_dir}")
        
        # LRU缓存
        from functools import lru_cache
        self._load_file = lru_cache(maxsize=cache_size)(self._load_file_impl)
    
    def _load_file_impl(self, path: str) -> Dict[str, np.ndarray]:
        """加载单个文件。"""
        data = np.load(path)
        if self.preprocess_fn is not None:
            data = self.preprocess_fn(data)
        return data
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self._load_file(self.file_list[idx])
        return {key: torch.from_numpy(val.astype(np.float32)) 
                for key, val in data.items()}
```

---

## 9. DataLoader配置

### 9.1 标准DataLoader

```python
from torch.utils.data import DataLoader, random_split

def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    创建训练、验证、测试数据加载器。
    
    Args:
        dataset: 轨迹数据集
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        num_workers: 数据加载线程数
        pin_memory: 是否锁页内存
        seed: 随机种子
        
    Returns:
        包含 'train', 'val', 'test' DataLoader 的字典
    """
    # 计算分割大小
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    # 随机分割
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # 丢弃不完整批次
        persistent_workers=num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
```

### 9.2 自定义Collate函数

```python
def trajectory_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义批次整理函数。
    
    处理可变长度轨迹和额外的元数据。
    """
    # 标准字段
    positions = torch.stack([b['positions'] for b in batch], dim=0)
    velocities = torch.stack([b['velocities'] for b in batch], dim=0)
    accelerations = torch.stack([b['accelerations'] for b in batch], dim=0)
    start_pos = torch.stack([b['start_pos'] for b in batch], dim=0)
    goal_pos = torch.stack([b['goal_pos'] for b in batch], dim=0)
    start_vel = torch.stack([b['start_vel'] for b in batch], dim=0)
    
    result = {
        'positions': positions,
        'velocities': velocities,
        'accelerations': accelerations,
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'start_vel': start_vel,
    }
    
    # 可选字段
    if 'env_encoding' in batch[0]:
        result['env_encoding'] = torch.stack([b['env_encoding'] for b in batch], dim=0)
    
    if 'obstacles' in batch[0]:
        result['obstacles'] = [b['obstacles'] for b in batch]
    
    return result
```

---

## 10. 与训练流程的集成

### 10.1 数据管道初始化

```python
def setup_data_pipeline(
    config: Dict,
) -> Tuple[DataLoader, DataLoader, TrajectoryNormalizer]:
    """
    设置完整的数据管道。
    
    Args:
        config: 配置字典
        
    Returns:
        train_loader, val_loader, normalizer
    """
    # 1. 加载原始数据
    raw_data = load_raw_trajectories(config['data_path'])
    
    # 2. B样条处理
    processor = BSplineTrajectoryProcessor(
        degree=config.get('bspline_degree', 3),
        smoothing=config.get('bspline_smoothing', 0),
        num_eval_points=config.get('seq_len', 64),
    )
    processed_data = process_trajectory_batch(
        raw_data,
        processor,
        dt=config.get('dt', 0.1),
    )
    
    # 3. 过滤异常轨迹
    # (已在处理过程中完成)
    
    # 4. 计算归一化参数
    normalizer = TrajectoryNormalizer()
    normalizer.fit(processed_data)
    normalizer.save(config['normalizer_path'])
    
    # 5. 创建数据集
    augmentation = create_standard_augmentation() if config.get('use_augmentation', True) else None
    
    dataset = FlowMPTrajectoryDataset(
        positions=processed_data['positions'],
        velocities=processed_data['velocities'],
        accelerations=processed_data['accelerations'],
        normalizer=normalizer,
        augmentation=augmentation,
    )
    
    # 6. 创建数据加载器
    loaders = create_dataloaders(
        dataset,
        batch_size=config.get('batch_size', 64),
        train_ratio=config.get('train_ratio', 0.9),
        val_ratio=config.get('val_ratio', 0.1),
        test_ratio=0.0,
        num_workers=config.get('num_workers', 4),
    )
    
    return loaders['train'], loaders['val'], normalizer
```

### 10.2 训练循环中的数据使用

```python
def training_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_fn: FlowMatchingLoss,
    interpolator: FlowInterpolator,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    单步训练。
    
    展示数据如何在训练中使用。
    """
    # 移动数据到设备
    positions = batch['positions'].to(device)      # [B, T, D]
    velocities = batch['velocities'].to(device)    # [B, T, D]
    accelerations = batch['accelerations'].to(device)  # [B, T, D]
    start_pos = batch['start_pos'].to(device)      # [B, D]
    goal_pos = batch['goal_pos'].to(device)        # [B, D]
    start_vel = batch['start_vel'].to(device)      # [B, D]
    
    B = positions.shape[0]
    
    # 采样流时间 t ~ Uniform(0, 1)
    t = torch.rand(B, device=device)
    
    # 构建插值状态和目标场
    interp_result = interpolator.interpolate_trajectory(
        q_1=positions,
        q_dot_1=velocities,
        q_ddot_1=accelerations,
        t=t,
    )
    
    x_t = interp_result['x_t']        # [B, T, 6]
    target = interp_result['target']  # [B, T, 6]
    
    # 模型前向传播
    pred = model(
        x_t=x_t,
        t=t,
        start_pos=start_pos,
        goal_pos=goal_pos,
        start_vel=start_vel,
    )
    
    # 计算损失
    loss_dict = loss_fn(pred, target)
    
    return loss_dict
```

---

## 11. 代码实现

### 11.1 完整的数据管道脚本

```python
#!/usr/bin/env python3
"""
CFM FlowMP 数据管道完整实现。

使用方法:
    python data_pipeline.py --input raw_trajectories/ --output processed_data.npz
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# B样条轨迹处理器
# ============================================================

class BSplineTrajectoryProcessor:
    """B样条轨迹处理器，用于平滑和重采样轨迹。"""
    
    def __init__(
        self,
        degree: int = 3,
        smoothing: float = 0.0,
        num_eval_points: int = 64,
    ):
        self.degree = degree
        self.smoothing = smoothing
        self.num_eval_points = num_eval_points
    
    def fit_trajectory(
        self,
        positions: np.ndarray,
        dt: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """拟合B样条并计算导数。"""
        from scipy.interpolate import splprep, splev
        
        T, D = positions.shape
        u_original = np.linspace(0, 1, T)
        u_eval = np.linspace(0, 1, self.num_eval_points)
        
        try:
            tck, u = splprep(positions.T, u=u_original, k=self.degree, s=self.smoothing)
            
            pos_eval = np.array(splev(u_eval, tck, der=0)).T
            vel_eval = np.array(splev(u_eval, tck, der=1)).T
            acc_eval = np.array(splev(u_eval, tck, der=2)).T
            
            duration = T * dt
            vel_eval = vel_eval / duration
            acc_eval = acc_eval / (duration ** 2)
            
            return {
                'positions': pos_eval.astype(np.float32),
                'velocities': vel_eval.astype(np.float32),
                'accelerations': acc_eval.astype(np.float32),
            }
        except Exception as e:
            print(f"B-spline fitting failed: {e}, using numerical derivatives")
            return self._numerical_fallback(positions, dt)
    
    def _numerical_fallback(self, positions: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
        """数值导数回退方案。"""
        from scipy.interpolate import interp1d
        
        T, D = positions.shape
        t_orig = np.linspace(0, 1, T)
        t_new = np.linspace(0, 1, self.num_eval_points)
        
        pos_new = np.zeros((self.num_eval_points, D))
        for d in range(D):
            f = interp1d(t_orig, positions[:, d], kind='cubic')
            pos_new[:, d] = f(t_new)
        
        vel = np.gradient(pos_new, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        
        return {
            'positions': pos_new.astype(np.float32),
            'velocities': vel.astype(np.float32),
            'accelerations': acc.astype(np.float32),
        }


# ============================================================
# 归一化器
# ============================================================

class TrajectoryNormalizer:
    """轨迹数据归一化器。"""
    
    def __init__(self):
        self.stats = {}
    
    def fit(self, data: Dict[str, np.ndarray]):
        """计算归一化统计量。"""
        for key in ['positions', 'velocities', 'accelerations']:
            if key in data:
                arr = data[key]
                self.stats[f'{key}_mean'] = arr.mean(axis=(0, 1))
                self.stats[f'{key}_std'] = np.maximum(arr.std(axis=(0, 1)), 1e-6)
    
    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """应用归一化。"""
        result = {}
        for key in ['positions', 'velocities', 'accelerations']:
            if key in data:
                mean = self.stats.get(f'{key}_mean', 0)
                std = self.stats.get(f'{key}_std', 1)
                result[key] = (data[key] - mean) / std
        return result
    
    def inverse_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """反归一化。"""
        result = {}
        for key in ['positions', 'velocities', 'accelerations']:
            if key in data:
                mean = self.stats.get(f'{key}_mean', 0)
                std = self.stats.get(f'{key}_std', 1)
                result[key] = data[key] * std + mean
        return result
    
    def save(self, path: str):
        np.savez(path, **self.stats)
    
    def load(self, path: str):
        loaded = np.load(path)
        self.stats = {key: loaded[key] for key in loaded.files}


# ============================================================
# 数据增强
# ============================================================

class TrajectoryAugmentation:
    """轨迹数据增强。"""
    
    def __init__(
        self,
        rotation_range: float = np.pi,
        translation_range: float = 0.5,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.01,
        augment_prob: float = 0.5,
    ):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.augment_prob = augment_prob
    
    def __call__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        if np.random.random() > self.augment_prob:
            return positions, velocities, accelerations
        
        D = positions.shape[-1]
        
        # 随机旋转 (2D)
        if D == 2:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle),  np.cos(angle)]])
            positions = positions @ R.T
            velocities = velocities @ R.T
            accelerations = accelerations @ R.T
        
        # 随机平移
        trans = np.random.uniform(-self.translation_range, self.translation_range, D)
        positions = positions + trans
        
        # 随机缩放
        scale = np.random.uniform(*self.scale_range)
        positions = positions * scale
        velocities = velocities * scale
        accelerations = accelerations * scale
        
        # 随机噪声
        positions = positions + np.random.randn(*positions.shape) * self.noise_std
        
        return positions, velocities, accelerations


# ============================================================
# Dataset
# ============================================================

class FlowMPDataset(Dataset):
    """FlowMP轨迹数据集。"""
    
    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        normalizer: Optional[TrajectoryNormalizer] = None,
        augmentation: Optional[TrajectoryAugmentation] = None,
        training: bool = True,
    ):
        self.positions = positions.astype(np.float32)
        self.velocities = velocities.astype(np.float32)
        self.accelerations = accelerations.astype(np.float32)
        self.normalizer = normalizer
        self.augmentation = augmentation
        self.training = training
        
        if normalizer is not None:
            data = self.normalizer.transform({
                'positions': self.positions,
                'velocities': self.velocities,
                'accelerations': self.accelerations,
            })
            self.positions = data['positions']
            self.velocities = data['velocities']
            self.accelerations = data['accelerations']
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        pos = self.positions[idx].copy()
        vel = self.velocities[idx].copy()
        acc = self.accelerations[idx].copy()
        
        if self.training and self.augmentation is not None:
            pos, vel, acc = self.augmentation(pos, vel, acc)
        
        return {
            'positions': torch.from_numpy(pos),
            'velocities': torch.from_numpy(vel),
            'accelerations': torch.from_numpy(acc),
            'start_pos': torch.from_numpy(pos[0]),
            'goal_pos': torch.from_numpy(pos[-1]),
            'start_vel': torch.from_numpy(vel[0]),
        }
    
    def train(self):
        self.training = True
        return self
    
    def eval(self):
        self.training = False
        return self


# ============================================================
# 数据管道主函数
# ============================================================

def process_raw_trajectories(
    input_dir: str,
    output_path: str,
    seq_len: int = 64,
    dt: float = 0.1,
    bspline_smoothing: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    处理原始轨迹数据。
    
    Args:
        input_dir: 原始轨迹目录
        output_path: 输出文件路径
        seq_len: 输出序列长度
        dt: 时间步长
        bspline_smoothing: B样条平滑因子
        
    Returns:
        处理后的数据字典
    """
    import glob
    from tqdm import tqdm
    
    input_dir = Path(input_dir)
    
    # 收集所有轨迹文件
    files = list(input_dir.glob("*.npy")) + list(input_dir.glob("*.npz"))
    print(f"Found {len(files)} trajectory files")
    
    # B样条处理器
    processor = BSplineTrajectoryProcessor(
        degree=3,
        smoothing=bspline_smoothing,
        num_eval_points=seq_len,
    )
    
    # 处理所有轨迹
    all_positions = []
    all_velocities = []
    all_accelerations = []
    
    for file in tqdm(files, desc="Processing trajectories"):
        try:
            data = np.load(file)
            if isinstance(data, np.ndarray):
                positions = data
            else:
                positions = data['positions'] if 'positions' in data else data['pos']
            
            result = processor.fit_trajectory(positions, dt)
            
            all_positions.append(result['positions'])
            all_velocities.append(result['velocities'])
            all_accelerations.append(result['accelerations'])
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # 堆叠数据
    data = {
        'positions': np.stack(all_positions, axis=0),
        'velocities': np.stack(all_velocities, axis=0),
        'accelerations': np.stack(all_accelerations, axis=0),
    }
    
    print(f"Processed {len(all_positions)} trajectories")
    print(f"Shape: {data['positions'].shape}")
    
    # 保存
    np.savez_compressed(output_path, **data)
    print(f"Saved to {output_path}")
    
    return data


def create_data_loaders(
    data_path: str,
    batch_size: int = 64,
    train_ratio: float = 0.9,
    num_workers: int = 4,
    use_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader, TrajectoryNormalizer]:
    """
    创建数据加载器。
    """
    # 加载数据
    data = np.load(data_path)
    positions = data['positions']
    velocities = data['velocities']
    accelerations = data['accelerations']
    
    # 创建归一化器
    normalizer = TrajectoryNormalizer()
    normalizer.fit({
        'positions': positions,
        'velocities': velocities,
        'accelerations': accelerations,
    })
    
    # 创建增强器
    augmentation = TrajectoryAugmentation() if use_augmentation else None
    
    # 创建数据集
    dataset = FlowMPDataset(
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        normalizer=normalizer,
        augmentation=augmentation,
    )
    
    # 分割数据集
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, normalizer


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CFM FlowMP Data Pipeline")
    subparsers = parser.add_subparsers(dest='command')
    
    # 处理命令
    process_parser = subparsers.add_parser('process', help='Process raw trajectories')
    process_parser.add_argument('--input', type=str, required=True, help='Input directory')
    process_parser.add_argument('--output', type=str, required=True, help='Output file')
    process_parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    process_parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    process_parser.add_argument('--smoothing', type=float, default=0.0, help='B-spline smoothing')
    
    # 验证命令
    validate_parser = subparsers.add_parser('validate', help='Validate processed data')
    validate_parser.add_argument('--data', type=str, required=True, help='Data file')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        process_raw_trajectories(
            args.input,
            args.output,
            seq_len=args.seq_len,
            dt=args.dt,
            bspline_smoothing=args.smoothing,
        )
    
    elif args.command == 'validate':
        data = np.load(args.data)
        print("Data validation:")
        for key in data.files:
            arr = data[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            print(f"    min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")


if __name__ == "__main__":
    main()
```

---

## 附录：常见问题

### Q1: B样条拟合失败怎么办？

A: 当轨迹点过少或有重复点时可能失败。解决方案：
1. 确保轨迹点数 > degree + 1
2. 移除重复点
3. 使用数值导数回退

### Q2: 数据增强是否会破坏物理一致性？

A: 几何变换（旋转、平移、缩放）保持物理一致性。噪声注入可能轻微破坏，但适量噪声有助于泛化。

### Q3: 如何处理不同长度的轨迹？

A: 使用B样条重采样到统一长度。对于极端情况（过短/过长），可以过滤或裁剪。

### Q4: 归一化参数如何在推理时使用？

A: 保存训练时的归一化参数，推理时：
1. 输入条件使用相同参数归一化
2. 输出轨迹使用 inverse_transform 反归一化

---

## 参考资料

1. FlowMP 官方实现: https://github.com/mkhangg/flow_mp
2. Scipy B-spline 文档: https://docs.scipy.org/doc/scipy/reference/interpolate.html
3. PyTorch DataLoader 文档: https://pytorch.org/docs/stable/data.html
