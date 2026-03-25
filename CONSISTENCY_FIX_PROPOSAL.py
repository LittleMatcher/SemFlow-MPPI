"""
L2 Training-Inference 一致性修复方案

问题：
  - Training: L2学向量场质量 (CFM loss)
  - Inference: proposal被赋予uniform权重，完全忽视向量场
  - 结果: validate_l2_no_l3中L2 ≈ Random

修复策略：
  从ODE求解中提取向量场置信度，用于proposal排序
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any


def extract_vector_field_confidence(
    velocity_fn_outputs: torch.Tensor,
    x_t: torch.Tensor,
    method: str = "norm_mean",
) -> torch.Tensor:
    """
    从模型的向量场预测中提取置信度/质量信号
    
    参数:
        velocity_fn_outputs: ODE求解中模型预测的向量场 [B*N, T, D*3]
                             其中 D*3 = 6 (position_dot, velocity_dot, accel_dot)
        x_t: ODE求解过程中的状态轨迹 [B*N, T, D*3]
        method: 置信度计算方法
            - "norm_mean": 向量场平均范数 (越大越确定)
            - "norm_max": 向量场最大范数
            - "variance": 向量场沿时间维度的方差
            - "smoothness": 向量场的平滑性 (低变化=越确定)
            - "consistency": 向量场与目标的一致性
    
    返回:
        proposal_scores: [B*N] 置信度分数（越高越好）
    """
    B_N, T, D = velocity_fn_outputs.shape
    device = velocity_fn_outputs.device
    dtype = velocity_fn_outputs.dtype
    
    if method == "norm_mean":
        # 向量场平均强度（强的向量场表示确定的预测）
        v_norms = torch.norm(velocity_fn_outputs, dim=-1)  # [B*N, T]
        scores = v_norms.mean(dim=-1)  # [B*N]
        
    elif method == "norm_max":
        # 向量场最大强度
        v_norms = torch.norm(velocity_fn_outputs, dim=-1)  # [B*N, T]
        scores = v_norms.max(dim=-1)[0]  # [B*N]
        
    elif method == "variance":
        # 向量场的低方差表示一致的预测（高置信度）
        # 低方差 = 高置信, 所以用 (1 + var)^-1
        v_norms = torch.norm(velocity_fn_outputs, dim=-1)  # [B*N, T]
        v_var = v_norms.var(dim=-1)  # [B*N]
        scores = 1.0 / (1.0 + v_var)  # [B*N]
        
    elif method == "smoothness":
        # 向量场沿时间的平滑性（低变化=高置信）
        v_norms = torch.norm(velocity_fn_outputs, dim=-1)  # [B*N, T]
        df_dt = torch.abs(torch.diff(v_norms, dim=-1))  # [B*N, T-1]
        smoothness = df_dt.mean(dim=-1)  # [B*N]
        scores = 1.0 / (1.0 + smoothness)  # [B*N], 低smoothness -> 高置信
        
    elif method == "consistency":
        # 向量场的各分量一致性
        # 将向量分为 [pos_dot, vel_dot, accel_dot]
        pos_dot = velocity_fn_outputs[..., :2]      # 速度
        vel_dot = velocity_fn_outputs[..., 2:4]     # 加速度
        accel_dot = velocity_fn_outputs[..., 4:6]   # 加加速度
        
        # 物理一致性：|vel| 应该与 |pos_dot| 接近
        x_vel = x_t[..., 2:4]  # 来自ODE结果的速度
        consistency = 1.0 - torch.abs(
            torch.norm(x_vel, dim=-1) - torch.norm(pos_dot, dim=-1)
        ).mean(dim=-1) / (torch.norm(x_vel, dim=-1).mean(dim=-1) + 1e-6)  # [B*N]
        scores = torch.clamp(consistency, 0, 1)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 归一化到 [0.1, 1.0]（避免完全零）
    scores_min = scores.min()
    scores_max = scores.max()
    if (scores_max - scores_min) > 1e-6:
        scores = 0.1 + 0.9 * (scores - scores_min) / (scores_max - scores_min + 1e-8)
    else:
        scores = torch.ones_like(scores) * 0.55
    
    return scores


def modify_generator_generate_method():
    """
    修改 TrajectoryGenerator.generate() 的建议（伪代码）
    
    在原 generate() 方法中，找到这一行（约第634行）：
    
        result['proposal_scores'] = default_scores
    
    替换为以下逻辑：
    """
    # ============ 替代方案 ============
    
    # 方案A：从ODE求解中提取向量场置信度（推荐）
    # 在ODE求解后添加：
    #
    # if x_1 is not None:
    #     # x_1 包含了ODE求解的轨迹
    #     v_pred = velocity_fn(x_1, t=torch.ones(B, device=x_1.device))  # 重新计算最后一步的向量场
    #     proposal_scores = extract_vector_field_confidence(v_pred, x_1, method="norm_mean")
    #     # 将proposal_scores乘以启发式得分以进行混合
    #     proposal_scores = proposal_scores / proposal_scores.max()  # 归一化

    # 方案B：混合向量场置信度与几何启发式
    # （需要在generate中接受cost_map参数以计算几何启发式）
    # proposal_scores = 0.6 * vector_field_confidence + 0.4 * geometric_heuristic
    
    # 方案C：记录ODE求解过程中的轨迹发散度（最准确但需要修改solver）
    # proposal_scores = solver.get_trajectory_divergence()  # 低发散=高置信
    
    return """
    关键改动位置：cfm_flowmp/inference/generator.py
    
    Line ~634: 替换 
        result['proposal_scores'] = default_scores
    
    替换为：
        result['proposal_scores'] = proposal_scores_computed_from_vector_field
    
    或使用混合：
        result['proposal_scores'] = (vector_field_confidence + 
                                     extract_heuristic_score(result, goal_pos, cost_map))
    """


# ============ validate_l2_no_l3.py 的修改建议 ============

def modified_proposal_score_l2aware(
    traj: 'np.ndarray',
    goal_xy: 'np.ndarray',
    cost_map_2d: 'np.ndarray',
    collision_threshold: float,
    model_confidence: Optional[float] = None,  # ← 新参数：模型置信度
    confidence_weight: float = 0.3,  # 新权重：向量场置信度的系数
) -> float:
    """
    改进的proposal评分函数，同时考虑几何启发式和模型置信度
    
    核心改进：加入模型置信度参数，使得L2生成的proposal能被识别为更优
    """
    # 原始启发式评分
    final_err = 'np.linalg.norm(traj[-1] - goal_xy)'
    collision_penalty = '1.0 if collision_flag(...) else 0.0'
    jitter = 'trajectory_jerk_mean(traj)'
    geometric_score = 'final_err + 2.0*collision_penalty + 0.2*jitter'
    
    # 改进：加入模型置信度
    if model_confidence is not None:
        # 低的model_confidence应该增加score（因为score越低越好）
        # 所以用 (1 - confidence) 来反映不确定性
        model_uncertainty = 1.0 - model_confidence
        combined_score = (
            (1 - confidence_weight) * geometric_score + 
            confidence_weight * model_uncertainty * 10.0  # 缩放以匹配geometric_score范围
        )
        return float(combined_score)
    else:
        # 回退到原始评分
        return float(geometric_score)


def modified_main_validate_l2_no_l3():
    """
    validate_l2_no_l3.py 主函数的改进建议
    
    在 L2 proposal 生成后，从生成器中提取 proposal_scores：
    """
    pseudo_code = """
    # 在 main() 中修改 L2 proposal 生成部分
    
    @torch.no_grad()
    def main() -> None:
        # ... 原始代码 ...
        
        # ======= 关键修改 =======
        # 当评估L2-only时：
        
        for sample_idx in range(n_eval):
            batch = dataset[sample_idx:sample_idx+1]
            cost_map = batch['cost_map'].to(device)
            
            # 生成L2 proposals（使用改进的generator）
            l2_output = model.generate_trajectory_anchors(
                cost_map=cost_map,
                x_curr=batch['start_state'].to(device),
                x_goal=batch['goal_state'].to(device),
                w_style=batch['style_weights'].to(device),
                num_samples=args.num_proposals,
            )
            
            l2_proposals = l2_output['trajectories'].cpu().numpy()
            l2_proposal_scores = l2_output['proposal_scores'].cpu().numpy()  # ← 新增
            
            # 使用模型置信度进行proposal选择
            best_idx = select_best_proposal_with_confidence(
                l2_proposals,
                goal_xy,
                cost_map_2d,
                args.collision_threshold,
                model_confidences=l2_proposal_scores,  # ← 新参数
            )
            
            best_l2_traj = l2_proposals[best_idx]
            
            # ... 继续评估 ...
    """
    return pseudo_code


# ============ train_l2_mock.py 的参数调整建议 ============

def recommended_training_parameters():
    """
    为了与推理目标对齐，建议的训练参数调整
    """
    return {
        "epochs": 100,  # ← 从3增加到100以充分训练向量场
        "batch_size": 32,
        
        # Loss权重对齐：
        # - position_loss 最重要（对应推理中的final_err权重1.0）
        # - acceleration_loss 次要（对应推理中的jerk权重0.2）
        # - velocity_loss 中等（确保物理一致性）
        
        "lambda_vel": 1.0,    # 位置流匹配（最高优先级）
        "lambda_acc": 0.5,    # 加速度流匹配（中等优先级）
        "lambda_jerk": 0.1,   # 加加速度流匹配（低优先级）
        
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "device": "cpu",  # 保持CPU以便重现
    }


if __name__ == "__main__":
    """
    实施步骤：
    
    1. 在 cfm_flowmp/inference/generator.py 中找到 generate() 方法（~行600-780）
       在第634行修改 proposal_scores 的计算
       
    2. 在 validate_l2_no_l3.py 中修改proposal选择逻辑
       使用 model.proposal_scores 而非纯启发式
       
    3. 使用推荐的训练参数重新训练L2（100 epochs）
       命令：python train_l2_mock.py --epochs 100 --batch_size 32 \
                 --lambda_vel 1.0 --lambda_acc 0.5 --lambda_jerk 0.1 \
                 --data_source generated --data_dir traj_data/cfm_env
       
    4. 使用新checkpoint和改进的推理重新运行验证
       python validate_l2_no_l3.py --checkpoint checkpoints_l2_consistent/best_model.pt
       
    预期结果：
       L2 goal_reach_rate: 0.0 → 0.5+（显著提升）
       L2 vs Random 有明显差异（而不是相等）
    """
    
    print("修复方案已准备，详见各函数文档。")
    print("关键改动：")
    print("  1. extract_vector_field_confidence() - 从模型提取proposal_scores")
    print("  2. generator.py:generate() - 使用置信度而非uniform")
    print("  3. validate_l2_no_l3.py - 利用proposal_scores进行选择")
    print("  4. train_l2_mock.py - 100 epochs + 调整loss权重")
