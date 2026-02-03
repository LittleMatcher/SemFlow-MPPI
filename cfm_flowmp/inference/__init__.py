"""
CFM FlowMP Inference Module

Contains:
- RK4Solver: Runge-Kutta 4th order ODE solver
- TrajectoryGenerator: Complete trajectory generation pipeline
- L1ReactiveController: L1 reaction control layer for local optimization
"""

from .ode_solver import RK4Solver, EulerSolver, AdaptiveRK45Solver
from .generator import TrajectoryGenerator, GeneratorConfig

# L1 Reactive Control Layer
try:
    from .l1_reactive_control import (
        L1ReactiveController,
        L1Config,
        DualObjectiveCost,
        SemanticFieldCost,
        TubeConstraintCost,
        EnergyCost,
    )
    __all__ = [
        "RK4Solver",
        "EulerSolver", 
        "AdaptiveRK45Solver",
        "TrajectoryGenerator",
        "GeneratorConfig",
        "L1ReactiveController",
        "L1Config",
        "DualObjectiveCost",
        "SemanticFieldCost",
        "TubeConstraintCost",
        "EnergyCost",
    ]
except ImportError:
    __all__ = [
        "RK4Solver",
        "EulerSolver", 
        "AdaptiveRK45Solver",
        "TrajectoryGenerator",
        "GeneratorConfig",
    ]
