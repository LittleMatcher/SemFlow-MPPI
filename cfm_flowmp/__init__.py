"""
CFM FlowMP: Conditional Flow Matching for Trajectory Planning

A PyTorch implementation of the FlowMP architecture for learning
trajectory generation using Conditional Flow Matching.

Key Components:
- Transformer-based conditional vector field prediction
- Gaussian Fourier time embedding with AdaLN conditioning
- RK4 ODE solver for trajectory generation
- B-spline smoothing for physical consistency

Interface System:
- interfaces.py: Abstract base classes and interface definitions
- interface_checker.py: Tools for validating interface implementations
- INTERFACE_WORKFLOW.md: Development workflow and best practices
"""

__version__ = "0.1.0"
__author__ = "CFM FlowMP Team"

# 核心模块（可选导入：避免在未安装依赖时阻断接口工具）
_core_import_error = None
try:
    from .models import FlowMPTransformer, FlowMPUNet1D
    from .training import CFMTrainer, FlowMatchingLoss
    from .inference import TrajectoryGenerator, RK4Solver
    # L1 反应控制层（可选）
    try:
        from .inference import L1ReactiveController, L1Config
    except ImportError:
        L1ReactiveController = None
        L1Config = None
except Exception as exc:  # 允许在缺少依赖时继续使用接口系统
    FlowMPTransformer = None
    FlowMPUNet1D = None
    CFMTrainer = None
    FlowMatchingLoss = None
    TrajectoryGenerator = None
    RK4Solver = None
    L1ReactiveController = None
    L1Config = None
    _core_import_error = exc

# 接口系统
from .interfaces import (
    InterfaceRegistry,
    BaseModel,
    EmbeddingBase,
    ConditionalModule,
    ODESolver,
    TrajectoryGeneratorBase,
    Smoother,
    DataInterpolator,
    LossFunction,
    Trainer,
    Dataset,
    DataLoader,
    Visualizer,
    Metric,
)

from .interface_checker import (
    InterfaceChecker,
    InterfaceValidationError,
    check_implementation,
    print_interface_report,
    print_implementation_template,
)

__all__ = [
    # 核心模块
    "FlowMPTransformer",
    "FlowMPUNet1D",
    "CFMTrainer", 
    "FlowMatchingLoss",
    "TrajectoryGenerator",
    "RK4Solver",
    "L1ReactiveController",
    "L1Config",
    
    # 接口系统
    "InterfaceRegistry",
    "BaseModel",
    "EmbeddingBase",
    "ConditionalModule",
    "ODESolver",
    "TrajectoryGeneratorBase",
    "Smoother",
    "DataInterpolator",
    "LossFunction",
    "Trainer",
    "Dataset",
    "DataLoader",
    "Visualizer",
    "Metric",
    
    # 接口工具
    "InterfaceChecker",
    "InterfaceValidationError",
    "check_implementation",
    "print_interface_report",
    "print_implementation_template",
]
