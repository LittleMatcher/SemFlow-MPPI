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

import importlib

__version__ = "0.1.0"
__author__ = "CFM FlowMP Team"

# 核心模块延迟导入：避免在仅使用脚本/接口工具时触发 torch 等重依赖初始化
_CORE_EXPORTS = {
    "FlowMPTransformer": ("models", "FlowMPTransformer"),
    "FlowMPUNet1D": ("models", "FlowMPUNet1D"),
    "SemanticDiffusionConfig": ("models", "SemanticDiffusionConfig"),
    "SemanticConditionedDiffusionPlanner": ("models", "SemanticConditionedDiffusionPlanner"),
    "create_semantic_diffusion_planner": ("models", "create_semantic_diffusion_planner"),
    "CFMTrainer": ("training", "CFMTrainer"),
    "FlowMatchingLoss": ("training", "FlowMatchingLoss"),
    "TrajectoryGenerator": ("inference", "TrajectoryGenerator"),
    "RK4Solver": ("inference", "RK4Solver"),
    "L1ReactiveController": ("inference", "L1ReactiveController"),
    "L1Config": ("inference", "L1Config"),
}

_INTERFACE_EXPORTS = {
    "InterfaceRegistry": ("interfaces", "InterfaceRegistry"),
    "BaseModel": ("interfaces", "BaseModel"),
    "EmbeddingBase": ("interfaces", "EmbeddingBase"),
    "ConditionalModule": ("interfaces", "ConditionalModule"),
    "ODESolver": ("interfaces", "ODESolver"),
    "TrajectoryGeneratorBase": ("interfaces", "TrajectoryGeneratorBase"),
    "Smoother": ("interfaces", "Smoother"),
    "DataInterpolator": ("interfaces", "DataInterpolator"),
    "LossFunction": ("interfaces", "LossFunction"),
    "Trainer": ("interfaces", "Trainer"),
    "Dataset": ("interfaces", "Dataset"),
    "DataLoader": ("interfaces", "DataLoader"),
    "Visualizer": ("interfaces", "Visualizer"),
    "Metric": ("interfaces", "Metric"),
    "InterfaceChecker": ("interface_checker", "InterfaceChecker"),
    "InterfaceValidationError": ("interface_checker", "InterfaceValidationError"),
    "check_implementation": ("interface_checker", "check_implementation"),
    "print_interface_report": ("interface_checker", "print_interface_report"),
    "print_implementation_template": ("interface_checker", "print_implementation_template"),
}

_LAZY_EXPORTS = {**_CORE_EXPORTS, **_INTERFACE_EXPORTS}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, symbol_name = _LAZY_EXPORTS[name]
        try:
            module = importlib.import_module(f".{module_name}", __name__)
            value = getattr(module, symbol_name)
            globals()[name] = value
            return value
        except Exception as exc:
            raise ImportError(
                f"Failed to import '{symbol_name}' from cfm_flowmp.{module_name}. "
                "Please check optional dependencies (e.g. torch)."
            ) from exc
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))

__all__ = [
    # 核心模块
    "FlowMPTransformer",
    "FlowMPUNet1D",
    "SemanticDiffusionConfig",
    "SemanticConditionedDiffusionPlanner",
    "create_semantic_diffusion_planner",
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
