"""
CFM FlowMP Utilities

Contains:
- Visualization utilities
- Metric computation
- Configuration helpers
- Results management
- Batch inference
"""

from .visualization import visualize_trajectory, visualize_flow_field
from .metrics import compute_metrics

from .config_loader import (
    load_config_from_yaml,
    save_config_to_yaml,
    config_to_dotmap,
    merge_configs,
    get_tensor_args,
    create_results_dir,
)

from .results_manager import (
    PlanningResult,
    ResultsManager,
)

from .batch_inference import (
    BatchInferencePipeline,
)

__all__ = [
    # Visualization
    "visualize_trajectory",
    "visualize_flow_field",
    # Metrics
    "compute_metrics",
    # Config (new)
    "load_config_from_yaml",
    "save_config_to_yaml",
    "config_to_dotmap",
    "merge_configs",
    "get_tensor_args",
    "create_results_dir",
    # Results (new)
    "PlanningResult",
    "ResultsManager",
    # Batch inference (new)
    "BatchInferencePipeline",
]
