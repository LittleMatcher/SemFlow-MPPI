"""
Interface Definitions for CFM FlowMP

定义项目中所有核心类和工具函数的抽象接口。
这个文件作为接口规范，保证各模块API的一致性和可重复使用性。

规范说明：
1. 所有工具类必须继承相应的抽象基类
2. 所有公开方法必须实现相应的接口方法
3. 新增方法前必须检查此文件
4. 新增方法后必须更新此文件
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union, List, Any

try:
    import torch
    import torch.nn as nn
except Exception:  # 允许在未安装 torch 时使用接口工具
    class _DummyTensor:  # noqa: D401 - 简化占位类型
        """占位 Tensor 类型（仅用于类型标注）"""
        pass

    class _DummyModule:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def parameters(self):
            return []

    class _DummyTorch:
        Tensor = _DummyTensor

    torch = _DummyTorch()  # type: ignore[assignment]
    nn = type("nn", (), {"Module": _DummyModule})  # type: ignore[assignment]


# ============ MODEL INTERFACES ============

class BaseModel(ABC, nn.Module):
    """所有模型的基类接口"""
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """前向传播"""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        pass
    
    def get_num_parameters(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())


class EmbeddingBase(ABC, nn.Module):
    """所有嵌入层的基类"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将输入映射到嵌入空间"""
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """返回输出维度"""
        pass


class ConditionalModule(ABC, nn.Module):
    """条件化模块的基类"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量
            condition: 条件张量
        
        Returns:
            条件化处理后的张量
        """
        pass


# ============ INFERENCE INTERFACES ============

class ODESolver(ABC):
    """ODE求解器的基类接口"""
    
    @abstractmethod
    def solve(
        self,
        velocity_fn,
        x_0: torch.Tensor,
        t_span: List[float],
        **kwargs
    ) -> torch.Tensor:
        """
        求解ODE方程 dx/dt = velocity_fn(x, t)
        
        Args:
            velocity_fn: 速度函数 velocity_fn(x_t, t) -> dx_t
            x_0: 初始条件 [B, T, D]
            t_span: 时间点列表
            **kwargs: 求解器特定参数
        
        Returns:
            最终状态 x_T [B, T, D]
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """返回求解器名称"""
        pass


class TrajectoryGeneratorBase(ABC):
    """轨迹生成器的基类接口"""
    
    @abstractmethod
    def generate(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        num_samples: int = 1,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        生成轨迹
        
        Args:
            start_pos: 起始位置 [B, D]
            goal_pos: 目标位置 [B, D]
            num_samples: 生成的样本数
            **kwargs: 生成参数
        
        Returns:
            包含以下键的字典:
            - positions: [B, T, D] 位置轨迹
            - velocities: [B, T, D] 速度
            - accelerations: [B, T, D] 加速度
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        conditions: List[Dict[str, torch.Tensor]],
        batch_size: int = 32,
        **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """批量生成轨迹"""
        pass


class Smoother(ABC):
    """轨迹平滑器的基类"""
    
    @abstractmethod
    def smooth(
        self,
        trajectory: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        平滑轨迹
        
        Args:
            trajectory: [B, T, D] 或 [T, D] 原始轨迹
            **kwargs: 平滑参数
        
        Returns:
            [B, T, D] 或 [T, D] 平滑后的轨迹
        """
        pass


# ============ TRAINING INTERFACES ============

class DataInterpolator(ABC):
    """数据插值器的基类"""
    
    @abstractmethod
    def interpolate(
        self,
        data: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        插值数据并计算目标
        
        Args:
            data: 原始数据 [B, T, D]
            t: 时间步 [B]
            **kwargs: 插值参数
        
        Returns:
            (插值状态, 目标字典)
            插值状态: [B, T, D]
            目标字典包含:
            - target_u: 位置速度场
            - target_v: 加速度场
            - target_w: 急动场
        """
        pass


class LossFunction(ABC, nn.Module):
    """损失函数的基类"""
    
    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        计算损失
        
        Args:
            pred: 模型预测 [B, T, D]
            target: 目标字典
            **kwargs: 损失参数
        
        Returns:
            标量损失值
        """
        pass
    
    @abstractmethod
    def get_component_losses(self) -> Dict[str, float]:
        """获取各分量损失"""
        pass


class Trainer(ABC):
    """训练器的基类"""
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch，返回指标字典"""
        pass
    
    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """验证，返回指标字典"""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """获取训练状态"""
        pass


# ============ DATA INTERFACES ============

class Dataset(ABC):
    """数据集的基类接口"""
    
    @abstractmethod
    def __len__(self) -> int:
        """返回数据集大小"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            包含以下键的字典:
            - positions: [T, D]
            - velocities: [T, D] (可选)
            - accelerations: [T, D] (可选)
            - start_pos: [D]
            - goal_pos: [D]
            - start_vel: [D]
        """
        pass
    
    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """获取数据集统计量"""
        pass


class DataLoader(ABC):
    """数据加载器的基类"""
    
    @abstractmethod
    def __iter__(self):
        """迭代批次"""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """返回批次数量"""
        pass


# ============ UTILITY INTERFACES ============

class Visualizer(ABC):
    """可视化工具的基类"""
    
    @abstractmethod
    def plot(
        self,
        trajectory: torch.Tensor,
        **kwargs
    ) -> Any:
        """
        绘制轨迹
        
        Args:
            trajectory: [B, T, D] 轨迹数据
            **kwargs: 绘图参数
        
        Returns:
            图像对象
        """
        pass


class Metric(ABC):
    """评估指标的基类"""
    
    @abstractmethod
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> float:
        """
        计算指标
        
        Args:
            predictions: 模型预测
            targets: 真值
            **kwargs: 参数
        
        Returns:
            标量指标值
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """返回指标名称"""
        pass
    
    def is_higher_better(self) -> bool:
        """指标是否越大越好（默认True）"""
        return True


# ============ INTERFACE REGISTRY ============

class InterfaceRegistry:
    """接口注册表，用于追踪接口和实现的对应关系"""
    
    _interfaces: Dict[str, type] = {}
    _implementations: Dict[str, List[type]] = {}
    
    @classmethod
    def register_interface(cls, interface_class: type) -> None:
        """注册接口"""
        cls._interfaces[interface_class.__name__] = interface_class
        cls._implementations[interface_class.__name__] = []
    
    @classmethod
    def register_implementation(cls, interface_name: str, impl_class: type) -> None:
        """注册实现"""
        if interface_name not in cls._implementations:
            cls._implementations[interface_name] = []
        cls._implementations[interface_name].append(impl_class)
    
    @classmethod
    def get_implementations(cls, interface_name: str) -> List[type]:
        """获取接口的所有实现"""
        return cls._implementations.get(interface_name, [])
    
    @classmethod
    def get_interface(cls, interface_name: str) -> Optional[type]:
        """获取接口定义"""
        return cls._interfaces.get(interface_name)
    
    @classmethod
    def check_implementation(cls, impl_class: type, interface_class: type) -> bool:
        """检查类是否正确实现了接口"""
        if not issubclass(impl_class, interface_class):
            return False
        
        # 检查所有抽象方法是否已实现
        abstract_methods = interface_class.__abstractmethods__
        for method in abstract_methods:
            if not hasattr(impl_class, method) or getattr(impl_class, method).__isabstractmethod__:
                return False
        
        return True


# 自动注册所有接口
for name, obj in list(globals().items()):
    if isinstance(obj, type) and issubclass(obj, ABC) and obj is not ABC:
        if 'Base' in name or 'Interface' in name or name in [
            'EmbeddingBase', 'ConditionalModule', 'ODESolver', 
            'TrajectoryGeneratorBase', 'Smoother', 'DataInterpolator',
            'LossFunction', 'Trainer', 'Dataset', 'DataLoader',
            'Visualizer', 'Metric'
        ]:
            InterfaceRegistry.register_interface(obj)
