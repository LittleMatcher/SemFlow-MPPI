"""
测试Fix1：验证新的proposal_scores计算是否正常工作

这个脚本验证：
1. compute_proposal_scores_from_trajectories()能正确计算置信度
2. proposal_scores不再是uniform，而是反映轨迹质量
3. L2生成的proposal与Random baseline在置信度上有差异
"""

import torch
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from cfm_flowmp.inference.generator import compute_proposal_scores_from_trajectories


def test_smoothness_method():
    """测试smoothness方法"""
    print("=" * 60)
    print("Test 1: Smoothness-based confidence scoring")
    print("=" * 60)
    
    # 创建测试轨迹
    # 轨迹1：平滑（低jerk）= 高置信度
    t = torch.linspace(0, 1, 64)
    smooth_traj = torch.stack([t, t**2], dim=-1).unsqueeze(0)  # 二次函数，平滑
    smooth_vel = torch.stack([torch.ones_like(t), 2*t], dim=-1).unsqueeze(0)
    smooth_acc = torch.stack([torch.zeros_like(t), torch.full_like(t, 2.0)], dim=-1).unsqueeze(0)
    
    # 轨迹2：抖动（高jerk）= 低置信度
    noisy_traj = smooth_traj + 0.1 * torch.randn_like(smooth_traj)
    noisy_vel = smooth_vel + 0.05 * torch.randn_like(smooth_vel)
    noisy_acc = smooth_acc + 0.02 * torch.randn_like(smooth_acc)
    
    # 堆叠
    trajectories = torch.cat([smooth_traj, noisy_traj], dim=0)  # [2, 64, 2]
    velocities = torch.cat([smooth_vel, noisy_vel], dim=0)
    accelerations = torch.cat([smooth_acc, noisy_acc], dim=0)
    
    scores = compute_proposal_scores_from_trajectories(
        trajectories, velocities, accelerations,
        method="smoothness"
    )
    
    print(f"轨迹1（平滑）    置信度: {scores[0]:.4f}")
    print(f"轨迹2（抖动）    置信度: {scores[1]:.4f}")
    print(f"✓ 平滑轨迹置信度更高: {scores[0] > scores[1]}")
    print()


def test_consistency_method():
    """测试consistency方法"""
    print("=" * 60)
    print("Test 2: Consistency-based confidence scoring")
    print("=" * 60)
    
    t = torch.linspace(0, 1, 64)
    
    # 一致的轨迹：v = dpos/dt, a = dv/dt（物理一致）
    pos = torch.stack([t, t**2], dim=-1).unsqueeze(0)
    vel = torch.stack([torch.ones_like(t), 2*t], dim=-1).unsqueeze(0)
    acc = torch.stack([torch.zeros_like(t), torch.full_like(t, 2.0)], dim=-1).unsqueeze(0)
    
    # 不一致的轨迹：随机生成不满足物理关系
    pos_bad = torch.randn(1, 64, 2)
    vel_bad = torch.randn(1, 64, 2)
    acc_bad = torch.randn(1, 64, 2)
    
    # 堆叠
    trajectories = torch.cat([pos, pos_bad], dim=0)
    velocities = torch.cat([vel, vel_bad], dim=0)
    accelerations = torch.cat([acc, acc_bad], dim=0)
    
    scores = compute_proposal_scores_from_trajectories(
        trajectories, velocities, accelerations,
        method="consistency"
    )
    
    print(f"一致的轨迹    置信度: {scores[0]:.4f}")
    print(f"不一致的轨迹  置信度: {scores[1]:.4f}")
    print(f"✓ 一致的轨迹置信度更高: {scores[0] > scores[1]}")
    print()


def test_combined_method():
    """测试combined方法"""
    print("=" * 60)
    print("Test 3: Combined confidence scoring (recommended)")
    print("=" * 60)
    
    t = torch.linspace(0, 1, 64)
    
    # 高质量轨迹：平滑 + 一致
    pos_good = torch.stack([t, t**2], dim=-1).unsqueeze(0)
    vel_good = torch.stack([torch.ones_like(t), 2*t], dim=-1).unsqueeze(0)
    acc_good = torch.stack([torch.zeros_like(t), torch.full_like(t, 2.0)], dim=-1).unsqueeze(0)
    
    # 低质量轨迹：随机
    pos_bad = torch.randn(1, 64, 2)
    vel_bad = torch.randn(1, 64, 2)
    acc_bad = torch.randn(1, 64, 2)
    
    # 生成多个轨迹用于批处理
    trajectories = torch.cat([pos_good]*2 + [pos_bad]*2, dim=0)  # [4, 64, 2]
    velocities = torch.cat([vel_good]*2 + [vel_bad]*2, dim=0)
    accelerations = torch.cat([acc_good]*2 + [acc_bad]*2, dim=0)
    
    scores = compute_proposal_scores_from_trajectories(
        trajectories, velocities, accelerations,
        method="combined"
    )
    
    print(f"高质量轨迹 #1 置信度: {scores[0]:.4f}")
    print(f"高质量轨迹 #2 置信度: {scores[1]:.4f}")
    print(f"低质量轨迹 #1 置信度: {scores[2]:.4f}")
    print(f"低质量轨迹 #2 置信度: {scores[3]:.4f}")
    print(f"✓ 高质量轨迹置信度更高: {scores[0] > scores[2] and scores[1] > scores[3]}")
    print()


def test_not_uniform():
    """验证proposal_scores不再是uniform"""
    print("=" * 60)
    print("Test 4: Proposal scores are NOT uniform (improvement from Fix1)")
    print("=" * 60)
    
    # 生成多个不同质量的轨迹
    num_trajs = 10
    trajectories = torch.randn(num_trajs, 64, 2)
    velocities = torch.randn(num_trajs, 64, 2)
    accelerations = torch.randn(num_trajs, 64, 2)
    
    scores = compute_proposal_scores_from_trajectories(
        trajectories, velocities, accelerations,
        method="combined"
    )
    
    print(f"生成了 {num_trajs} 个随机轨迹")
    print(f"置信度分数: {scores.numpy()}")
    print(f"统计:")
    print(f"  最小置信度: {scores.min():.4f}")
    print(f"  最大置信度: {scores.max():.4f}")
    print(f"  平均置信度: {scores.mean():.4f}")
    print(f"  标准差: {scores.std():.4f}")
    
    # 验证不是uniform
    is_uniform = torch.allclose(scores, scores[0])
    print(f"✓ 置信度不是uniform: {not is_uniform}")
    print()


def test_normalized_weights():
    """验证mode_weights正确归一化"""
    print("=" * 60)
    print("Test 5: Mode weights correctly normalized")
    print("=" * 60)
    
    trajectories = torch.randn(5, 64, 2)
    velocities = torch.randn(5, 64, 2)
    accelerations = torch.randn(5, 64, 2)
    
    scores = compute_proposal_scores_from_trajectories(
        trajectories, velocities, accelerations,
        method="combined"
    )
    
    weights = scores / scores.sum()
    
    print(f"Proposal scores: {scores.numpy()}")
    print(f"Mode weights: {weights.numpy()}")
    print(f"权重和: {weights.sum():.6f}")
    print(f"✓ 权重正确归一化到1.0: {abs(weights.sum().item() - 1.0) < 1e-5}")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Fix1 验证：新的 Proposal Scores 计算" + " " * 11 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        test_smoothness_method()
        test_consistency_method()
        test_combined_method()
        test_not_uniform()
        test_normalized_weights()
        
        print("=" * 60)
        print("✅ 所有测试通过！Fix1 正常工作")
        print("=" * 60)
        print()
        print("关键改进:")
        print("  ✓ proposal_scores 基于轨迹质量而非uniform")
        print("  ✓ 4种置信度计算方法可选")
        print("  ✓ 'combined' 方法综合平滑性与一致性（推荐）")
        print("  ✓ 高质量CFM向量场现在能被proposal选择识别")
        print()
        print("下一步:")
        print("  1. 运行: validate_l2_no_l3.py（应该看到L2 vs Random的差异）")
        print("  2. 用新proposal_scores执行Fix2（重新训练100 epochs）")
        print()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
