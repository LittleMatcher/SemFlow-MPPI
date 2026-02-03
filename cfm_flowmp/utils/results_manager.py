"""
结果管理和保存工具
"""

import os
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class PlanningResult:
    """规划结果数据结构"""
    
    # 输入
    start_pos: np.ndarray
    goal_pos: np.ndarray
    start_vel: Optional[np.ndarray] = None
    
    # L2 输出
    l2_anchor_positions: Optional[np.ndarray] = None
    l2_anchor_velocities: Optional[np.ndarray] = None
    
    # L1 输出
    l1_optimal_control: Optional[np.ndarray] = None
    l1_best_control: Optional[np.ndarray] = None
    l1_best_mode: Optional[int] = None
    
    # 代价和指标
    l1_best_cost: Optional[float] = None
    l1_mean_cost: Optional[float] = None
    l1_all_costs: Optional[np.ndarray] = None
    
    # 时间统计
    t_l2_generation: float = 0.0
    t_l1_optimization: float = 0.0
    t_total: float = 0.0
    
    # 元数据
    frame_id: Optional[int] = None
    warm_start_used: bool = False
    config: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanningResult':
        """从字典创建"""
        return cls(**data)


class ResultsManager:
    """
    结果管理器
    
    负责保存和加载规划结果，支持批量处理。
    """
    
    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: 结果保存目录
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.trajectories_dir = self.results_dir / "trajectories"
        self.metrics_dir = self.results_dir / "metrics"
        self.visualizations_dir = self.results_dir / "visualizations"
        self.configs_dir = self.results_dir / "configs"
        
        for dir_path in [self.trajectories_dir, self.metrics_dir, 
                        self.visualizations_dir, self.configs_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_result(
        self,
        result: PlanningResult,
        filename: Optional[str] = None,
        save_format: str = "pt",  # "pt", "pkl", "npz"
    ) -> str:
        """
        保存单个规划结果
        
        Args:
            result: 规划结果
            filename: 文件名（可选，自动生成）
            save_format: 保存格式
            
        Returns:
            saved_path: 保存路径
        """
        if filename is None:
            if result.frame_id is not None:
                filename = f"result_frame_{result.frame_id:03d}.{save_format}"
            else:
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"result_{timestamp}.{save_format}"
        
        save_path = self.trajectories_dir / filename
        
        if save_format == "pt":
            # PyTorch 格式
            torch.save(result.to_dict(), save_path, _use_new_zipfile_serialization=True)
        elif save_format == "pkl":
            # Pickle 格式
            with open(save_path, 'wb') as f:
                pickle.dump(result.to_dict(), f)
        elif save_format == "npz":
            # NumPy 格式（只保存数组数据）
            np.savez(
                save_path,
                start_pos=result.start_pos,
                goal_pos=result.goal_pos,
                optimal_control=result.l1_optimal_control,
                best_control=result.l1_best_control,
                best_cost=result.l1_best_cost,
                mean_cost=result.l1_mean_cost,
            )
        else:
            raise ValueError(f"Unsupported save_format: {save_format}")
        
        return str(save_path)
    
    def load_result(self, filename: str, load_format: str = "pt") -> PlanningResult:
        """
        加载规划结果
        
        Args:
            filename: 文件名
            load_format: 加载格式
            
        Returns:
            result: 规划结果
        """
        load_path = self.trajectories_dir / filename
        
        if load_format == "pt":
            data = torch.load(load_path, map_location='cpu', weights_only=False)
        elif load_format == "pkl":
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
        elif load_format == "npz":
            data = np.load(load_path, allow_pickle=True)
            # 转换为字典格式
            data = {key: data[key] for key in data.keys()}
        else:
            raise ValueError(f"Unsupported load_format: {load_format}")
        
        return PlanningResult.from_dict(data)
    
    def save_batch_results(
        self,
        results: List[PlanningResult],
        batch_name: str = "batch",
    ) -> str:
        """
        保存批量结果
        
        Args:
            results: 结果列表
            batch_name: 批次名称
            
        Returns:
            saved_path: 保存路径
        """
        batch_path = self.trajectories_dir / f"{batch_name}.pt"
        
        results_dict = {
            'results': [r.to_dict() for r in results],
            'n_results': len(results),
            'batch_name': batch_name,
        }
        
        torch.save(results_dict, batch_path, _use_new_zipfile_serialization=True)
        
        return str(batch_path)
    
    def save_metrics(
        self,
        metrics: Dict[str, Any],
        filename: str = "metrics.yaml",
    ) -> str:
        """
        保存指标
        
        Args:
            metrics: 指标字典
            filename: 文件名
            
        Returns:
            saved_path: 保存路径
        """
        import yaml
        
        save_path = self.metrics_dir / filename
        
        with open(save_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        
        return str(save_path)
    
    def save_config(
        self,
        config: Any,
        filename: str = "config.yaml",
    ) -> str:
        """
        保存配置
        
        Args:
            config: 配置对象或字典
            filename: 文件名
            
        Returns:
            saved_path: 保存路径
        """
        from .config_loader import save_config_to_yaml
        
        save_path = self.configs_dir / filename
        save_config_to_yaml(config, str(save_path))
        
        return str(save_path)

