"""
配置加载和管理工具
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    from dotmap import DotMap
except ImportError:
    # Fallback: use dict if dotmap not available
    DotMap = dict


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    从 YAML 文件加载配置
    
    Args:
        config_path: YAML 配置文件路径
        
    Returns:
        config_dict: 配置字典
    """
    config_path = os.path.expandvars(config_path)  # 支持环境变量
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict


def save_config_to_yaml(config: Any, save_path: str):
    """
    保存配置到 YAML 文件
    
    Args:
        config: 配置对象（dataclass 或 dict）
        save_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换为字典
    if hasattr(config, '__dict__'):
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        else:
            config_dict = config.__dict__
    else:
        config_dict = config
    
    # 保存到 YAML
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def config_to_dotmap(config: Any) -> DotMap:
    """
    将配置转换为 DotMap（便于访问）
    
    Args:
        config: 配置对象或字典
        
    Returns:
        dotmap_config: DotMap 对象
    """
    if isinstance(config, DotMap):
        return config
    
    if hasattr(config, '__dict__'):
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        else:
            config_dict = config.__dict__
    else:
        config_dict = config
    
    return DotMap(config_dict)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    合并配置（override_config 覆盖 base_config）
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        merged_config: 合并后的配置
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_tensor_args(device: str = "cuda", dtype: str = "float32") -> Dict[str, Any]:
    """
    获取统一的 tensor 参数（参考 torch_robotics 的设计）
    
    Args:
        device: 设备 ("cuda", "cpu", "cuda:0" 等)
        dtype: 数据类型 ("float32", "float64" 等)
        
    Returns:
        tensor_args: 包含 device 和 dtype 的字典
    """
    import torch
    
    # 处理设备
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print(f"Warning: CUDA not available, using CPU instead")
    
    # 处理数据类型
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)
    
    return {
        "device": torch.device(device),
        "dtype": torch_dtype,
    }


def create_results_dir(base_dir: str = "results", experiment_name: str = None) -> str:
    """
    创建结果目录
    
    Args:
        base_dir: 基础目录
        experiment_name: 实验名称（可选）
        
    Returns:
        results_dir: 结果目录路径
    """
    import time
    
    if experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    results_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    return results_dir

