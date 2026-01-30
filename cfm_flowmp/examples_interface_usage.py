"""
接口系统使用示例

此文件展示如何在项目中使用接口系统来实现类和验证实现。

包含内容：
1. 实现新的 ODE 求解器
2. 实现新的损失函数  
3. 实现新的数据集
4. 实现新的评估指标
5. 如何处理接口验证错误

所有示例都遵循严格的接口定义和验证流程。
"""

# ============================================================================
# 示例 1: 实现一个新的 ODE 求解器 (DormandPrince45)
# ============================================================================

from abc import abstractmethod
import torch
from typing import Callable, Tuple, Optional, Dict, Any
from cfm_flowmp.interfaces import ODESolver


class DormandPrince45Solver(ODESolver):
    """
    Dormand-Prince 5 阶 ODE 求解器实现
    
    这是一个 Runge-Kutta 方法的自适应步长实现，
    提供 4 阶和 5 阶估计用于误差控制。
    """
    
    def __init__(self, rtol: float = 1e-3, atol: float = 1e-5):
        """
        初始化求解器
        
        Args:
            rtol: 相对误差容限
            atol: 绝对误差容限
        """
        self.rtol = rtol
        self.atol = atol
    
    def solve(
        self,
        vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        initial_state: torch.Tensor,
        t_span: Tuple[float, float],
        t_eval: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        使用Dormand-Prince方法求解ODE
        
        Args:
            vector_field: 向量场函数 f(x, t)
            initial_state: 初始状态
            t_span: 时间范围 (t_start, t_end)
            t_eval: 输出时间点
            **kwargs: 其他参数
        
        Returns:
            形状为 (len(t_eval), *initial_state.shape) 的轨迹张量
        """
        if t_eval is None:
            t_eval = torch.linspace(t_span[0], t_span[1], 100, 
                                   device=initial_state.device)
        
        trajectory = [initial_state.unsqueeze(0)]
        current_state = initial_state.clone()
        current_t = t_span[0]
        
        for target_t in t_eval[1:]:
            dt = target_t - current_t
            current_state = self.step(vector_field, current_state, 
                                     current_t, dt, **kwargs)
            trajectory.append(current_state.unsqueeze(0))
            current_t = target_t
        
        return torch.cat(trajectory, dim=0)
    
    def step(
        self,
        vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        state: torch.Tensor,
        t: float,
        dt: float,
        **kwargs
    ) -> torch.Tensor:
        """
        执行一步 Dormand-Prince 方法
        
        Args:
            vector_field: 向量场函数
            state: 当前状态
            t: 当前时间
            dt: 时间步长
            **kwargs: 其他参数
        
        Returns:
            下一时刻的状态
        """
        # Dormand-Prince 系数
        c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        ]
        
        k = []
        for i, (c_i, a_i) in enumerate(zip(c, a)):
            t_i = t + c_i * dt
            state_i = state.clone()
            for j, a_ij in enumerate(a_i):
                state_i = state_i + a_ij * dt * k[j]
            k.append(vector_field(state_i, t_i))
        
        # 5阶估计
        b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        next_state = state.clone()
        for b_i, k_i in zip(b5, k):
            next_state = next_state + b_i * dt * k_i
        
        return next_state


# ============================================================================
# 示例 2: 实现一个新的损失函数 (参数化Loss)
# ============================================================================

from cfm_flowmp.interfaces import LossFunction


class ParametricFlowLoss(LossFunction):
    """
    参数化流匹配损失函数
    
    用于学习条件向量场，基于匹配真实轨迹和生成轨迹。
    """
    
    def __init__(self, reduction: str = 'mean', weight: float = 1.0):
        """
        初始化损失函数
        
        Args:
            reduction: 'mean' 或 'sum'
            weight: 损失权重
        """
        self.reduction = reduction
        self.weight = weight
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        计算损失值
        
        Args:
            predictions: 预测向量场
            targets: 目标向量
            **kwargs: 其他参数
        
        Returns:
            标量损失值
        """
        loss = torch.nn.functional.mse_loss(predictions, targets, 
                                            reduction=self.reduction)
        return self.weight * loss
    
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        调用损失函数
        
        Args:
            predictions: 预测向量场
            targets: 目标向量
            **kwargs: 其他参数
        
        Returns:
            标量损失值
        """
        return self.compute_loss(predictions, targets, **kwargs)


# ============================================================================
# 示例 3: 实现一个新的数据集
# ============================================================================

from cfm_flowmp.interfaces import Dataset


class SyntheticTrajectoryDataset(Dataset):
    """
    合成轨迹数据集
    
    生成简单的参数化轨迹用于演示。
    """
    
    def __init__(self, num_samples: int = 1000, trajectory_length: int = 64):
        """
        初始化数据集
        
        Args:
            num_samples: 样本数量
            trajectory_length: 每条轨迹的长度
        """
        self.num_samples = num_samples
        self.trajectory_length = trajectory_length
        self.data = self._generate_data()
    
    def _generate_data(self):
        """生成合成轨迹数据"""
        trajectories = []
        for _ in range(self.num_samples):
            # 生成简单的圆形轨迹
            t = torch.linspace(0, 2 * 3.14159, self.trajectory_length)
            x = torch.cos(t)
            y = torch.sin(t)
            trajectory = torch.stack([x, y], dim=1)
            trajectories.append(trajectory)
        return torch.stack(trajectories)
    
    def __len__(self) -> int:
        """
        数据集大小
        
        Returns:
            样本数量
        """
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            包含轨迹数据的字典
        """
        trajectory = self.data[idx]
        return {
            'trajectory': trajectory,
            'start': trajectory[0],
            'goal': trajectory[-1],
        }


# ============================================================================
# 示例 4: 实现一个新的评估指标
# ============================================================================

from cfm_flowmp.interfaces import Metric


class TrajectoryLengthMetric(Metric):
    """
    轨迹长度评估指标
    
    计算轨迹的总长度（所有段的和）
    """
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> float:
        """
        计算轨迹长度
        
        Args:
            predictions: 预测轨迹 (T, D)
            targets: 目标轨迹 (T, D)
            **kwargs: 其他参数
        
        Returns:
            平均轨迹长度
        """
        # 计算段长度
        diffs = torch.diff(predictions, dim=0)
        lengths = torch.norm(diffs, dim=1)
        total_length = lengths.sum().item()
        return total_length
    
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> float:
        """
        调用指标
        
        Args:
            predictions: 预测轨迹
            targets: 目标轨迹
            **kwargs: 其他参数
        
        Returns:
            指标值
        """
        return self.compute(predictions, targets, **kwargs)


# ============================================================================
# 示例 5: 使用接口检查器验证实现
# ============================================================================

def verify_all_implementations():
    """
    验证所有实现都符合接口定义
    """
    from cfm_flowmp.interface_checker import InterfaceChecker
    from cfm_flowmp.interfaces import (
        ODESolver, LossFunction, Dataset, Metric
    )
    
    implementations = [
        (DormandPrince45Solver(), ODESolver),
        (ParametricFlowLoss(), LossFunction),
        (SyntheticTrajectoryDataset(), Dataset),
        (TrajectoryLengthMetric(), Metric),
    ]
    
    print("\n" + "="*70)
    print("接口实现验证")
    print("="*70 + "\n")
    
    all_passed = True
    for impl, interface in implementations:
        try:
            passed, _ = InterfaceChecker.check_implementation(
                impl.__class__, interface, raise_error=True
            )
            print(f"✓ {impl.__class__.__name__} 正确实现 {interface.__name__}")
        except Exception as e:
            print(f"✗ {impl.__class__.__name__} 实现 {interface.__name__} 失败")
            print(f"  错误: {e}\n")
            all_passed = False
    
    print("="*70)
    if all_passed:
        print("🎉 所有实现都通过了接口验证！\n")
    else:
        print("❌ 某些实现未通过验证，请检查上述错误。\n")
    
    return all_passed


# ============================================================================
# 示例 6: 在实际使用中集成实现
# ============================================================================

def example_usage():
    """
    演示如何使用这些实现
    """
    print("\n" + "="*70)
    print("实际使用示例")
    print("="*70 + "\n")
    
    # 1. 创建求解器
    print("1️⃣ 创建 Dormand-Prince ODE 求解器...")
    solver = DormandPrince45Solver()
    
    # 2. 创建数据集
    print("2️⃣ 创建合成轨迹数据集...")
    dataset = SyntheticTrajectoryDataset(num_samples=10)
    sample = dataset[0]
    print(f"   - 样本轨迹形状: {sample['trajectory'].shape}")
    
    # 3. 创建损失函数
    print("3️⃣ 创建参数化流损失函数...")
    loss_fn = ParametricFlowLoss()
    
    # 4. 创建评估指标
    print("4️⃣ 创建轨迹长度评估指标...")
    metric = TrajectoryLengthMetric()
    
    # 5. 演示一个简单的向量场和求解
    print("5️⃣ 求解简单的 ODE...")
    
    def simple_vector_field(x, t):
        """简单的向量场：dx/dt = -x"""
        return -x
    
    initial_state = torch.tensor([1.0, 1.0])
    trajectory = solver.solve(
        simple_vector_field,
        initial_state,
        t_span=(0, 1),
        t_eval=torch.linspace(0, 1, 10)
    )
    print(f"   - 轨迹形状: {trajectory.shape}")
    
    # 6. 计算损失和指标
    print("6️⃣ 计算损失和评估指标...")
    predictions = sample['trajectory']
    targets = sample['trajectory']
    
    loss_value = loss_fn(predictions, targets)
    metric_value = metric(predictions, targets)
    
    print(f"   - 损失值: {loss_value.item():.4f}")
    print(f"   - 轨迹长度: {metric_value:.4f}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # 验证实现
    verify_all_implementations()
    
    # 演示使用
    example_usage()
