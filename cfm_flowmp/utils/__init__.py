"""
CFM FlowMP Utilities

Contains:
- Visualization utilities
- Metric computation
- Configuration helpers
"""

from .visualization import visualize_trajectory, visualize_flow_field
from .metrics import compute_metrics

__all__ = [
    "visualize_trajectory",
    "visualize_flow_field",
    "compute_metrics",
]
