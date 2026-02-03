"""
批量推理工具

"""

import time
import torch
import numpy as np
from typing import List, Dict, Optional, Callable
from tqdm import tqdm

from ..inference import TrajectoryGenerator, L1ReactiveController
from .results_manager import ResultsManager, PlanningResult


class BatchInferencePipeline:
    """
    批量推理管道
    
    支持批量处理多个规划任务，自动管理结果保存。
    """
    
    def __init__(
        self,
        l2_generator: TrajectoryGenerator,
        l1_controller: L1ReactiveController,
        results_manager: ResultsManager,
        tensor_args: Optional[Dict] = None,
    ):
        """
        Args:
            l2_generator: L2 轨迹生成器
            l1_controller: L1 反应控制器
            results_manager: 结果管理器
            tensor_args: Tensor 参数（device, dtype）
        """
        self.l2_generator = l2_generator
        self.l1_controller = l1_controller
        self.results_manager = results_manager
        
        if tensor_args is None:
            tensor_args = {"device": "cuda" if torch.cuda.is_available() else "cpu", 
                          "dtype": torch.float32}
        self.tensor_args = tensor_args
        self.device = tensor_args["device"]
    
    def plan_single(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        start_vel: Optional[np.ndarray] = None,
        frame_id: Optional[int] = None,
        n_l2_samples: int = 5,
        n_l1_iterations: int = 10,
        use_warm_start: bool = True,
        verbose: bool = False,
    ) -> PlanningResult:
        """
        规划单个任务
        
        Args:
            start_pos: 起始位置 [D]
            goal_pos: 目标位置 [D]
            start_vel: 起始速度 [D] (可选)
            frame_id: 帧 ID（用于在线规划）
            n_l2_samples: L2 生成的锚点数量
            n_l1_iterations: L1 优化迭代次数
            use_warm_start: 是否使用热启动
            verbose: 是否打印详细信息
            
        Returns:
            result: 规划结果
        """
        # 转换为 torch tensor
        start_pos_t = torch.tensor(start_pos, **self.tensor_args).unsqueeze(0)
        goal_pos_t = torch.tensor(goal_pos, **self.tensor_args).unsqueeze(0)
        start_vel_t = None
        if start_vel is not None:
            start_vel_t = torch.tensor(start_vel, **self.tensor_args).unsqueeze(0)
        
        # 获取热启动状态
        warm_start_state = None
        if use_warm_start:
            warm_start_state = self.l1_controller.get_warm_start_state()
        
        # L2 生成
        t_l2_start = time.perf_counter()
        with torch.no_grad():
            l2_results = []
            for k in range(n_l2_samples):
                l2_result = self.l2_generator.generate(
                    start_pos=start_pos_t,
                    goal_pos=goal_pos_t,
                    start_vel=start_vel_t,
                    num_samples=1,
                    warm_start_state=warm_start_state,
                )
                l2_results.append(l2_result)
        
        # 合并锚点
        anchor_positions = torch.stack(
            [r['positions'][0] for r in l2_results], dim=0
        )  # [K, T, D]
        anchor_velocities = torch.stack(
            [r['velocities'][0] for r in l2_results], dim=0
        )
        
        t_l2 = time.perf_counter() - t_l2_start
        
        # 准备 L2 输出
        l2_output = {
            'positions': anchor_positions,
            'velocities': anchor_velocities,
        }
        
        # L1 优化
        t_l1_start = time.perf_counter()
        self.l1_controller.initialize_from_l2_output(l2_output)
        l1_result = self.l1_controller.optimize(
            n_iterations=n_l1_iterations,
            verbose=verbose,
        )
        optimal_control = self.l1_controller.get_next_control(
            l2_output, n_iterations=n_l1_iterations
        )
        t_l1 = time.perf_counter() - t_l1_start
        
        # 创建结果
        result = PlanningResult(
            start_pos=start_pos,
            goal_pos=goal_pos,
            start_vel=start_vel,
            l2_anchor_positions=anchor_positions.cpu().numpy(),
            l2_anchor_velocities=anchor_velocities.cpu().numpy(),
            l1_optimal_control=optimal_control.cpu().numpy(),
            l1_best_control=l1_result['best_control'].cpu().numpy(),
            l1_best_mode=l1_result['best_mode'],
            l1_best_cost=l1_result['best_cost'],
            l1_mean_cost=l1_result['mean_cost'],
            l1_all_costs=l1_result['all_costs'].cpu().numpy(),
            t_l2_generation=t_l2,
            t_l1_optimization=t_l1,
            t_total=t_l2 + t_l1,
            frame_id=frame_id,
            warm_start_used=warm_start_state is not None,
        )
        
        return result
    
    def plan_batch(
        self,
        tasks: List[Dict[str, np.ndarray]],
        save_results: bool = True,
        batch_name: str = "batch",
        verbose: bool = True,
        **kwargs,
    ) -> List[PlanningResult]:
        """
        批量规划
        
        Args:
            tasks: 任务列表，每个任务包含 start_pos, goal_pos, start_vel (可选)
            save_results: 是否保存结果
            batch_name: 批次名称
            verbose: 是否显示进度条
            **kwargs: 传递给 plan_single 的其他参数
            
        Returns:
            results: 规划结果列表
        """
        results = []
        
        iterator = tqdm(tasks, desc=f"Planning batch: {batch_name}") if verbose else tasks
        
        for i, task in enumerate(iterator):
            result = self.plan_single(
                start_pos=task['start_pos'],
                goal_pos=task['goal_pos'],
                start_vel=task.get('start_vel'),
                frame_id=i,
                verbose=False,
                **kwargs,
            )
            results.append(result)
        
        # 保存批量结果
        if save_results:
            self.results_manager.save_batch_results(results, batch_name)
        
        return results
    
    def plan_online(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        n_frames: int = 10,
        update_start_pos_fn: Optional[Callable] = None,
        save_results: bool = True,
        **kwargs,
    ) -> List[PlanningResult]:
        """
        在线规划（多帧，使用热启动）
        
        Args:
            start_pos: 初始起始位置
            goal_pos: 目标位置
            n_frames: 帧数
            update_start_pos_fn: 更新起始位置的函数（可选）
            save_results: 是否保存结果
            **kwargs: 传递给 plan_single 的其他参数
            
        Returns:
            results: 规划结果列表
        """
        results = []
        current_start_pos = start_pos.copy()
        
        for frame in range(n_frames):
            result = self.plan_single(
                start_pos=current_start_pos,
                goal_pos=goal_pos,
                frame_id=frame,
                use_warm_start=(frame > 0),  # 第一帧不使用热启动
                **kwargs,
            )
            results.append(result)
            
            # 更新起始位置（使用最优控制的第一步）
            if update_start_pos_fn is not None:
                current_start_pos = update_start_pos_fn(result)
            else:
                # 默认：使用最优控制的第一步
                if result.l1_optimal_control is not None:
                    current_start_pos = result.l1_optimal_control[0]
            
            # 保存单个结果
            if save_results:
                self.results_manager.save_result(result)
        
        return results

