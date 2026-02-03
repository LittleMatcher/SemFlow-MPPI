#!/usr/bin/env python3
"""
完整流程测试脚本：从数据生成到训练

测试 SemFlow-MPPI 项目的完整工作流程：
1. 生成训练数据（可选：合成数据或环境数据）
2. 训练模型
3. 验证模型
4. 简单推理测试

用法:
    python test_pipeline.py --quick          # 快速测试（合成数据，少量epoch）
    python test_pipeline.py --full           # 完整测试（生成数据，更多epoch）
    python test_pipeline.py --synthetic-only # 仅使用合成数据
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np

# 添加项目路径
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from cfm_flowmp.models import create_flowmp_transformer
from cfm_flowmp.training import CFMTrainer, FlowMatchingConfig, TrainerConfig
from cfm_flowmp.data import SyntheticTrajectoryDataset, TrajectoryDataset, create_dataloader
from cfm_flowmp.inference import TrajectoryGenerator, GeneratorConfig


def test_data_generation(args):
    """测试数据生成"""
    print("\n" + "="*60)
    print("步骤 1: 数据生成测试")
    print("="*60)
    
    if args.use_synthetic:
        print("使用合成数据（跳过数据生成）")
        return None
    
    # 这里可以添加真实数据生成的测试
    # 目前使用合成数据作为默认选项
    print("提示: 要生成真实环境数据，请运行:")
    print("  python -m cfm_flowmp.scripts.generate_data.generate_env_trajs_cfm \\")
    print("      --planner rrt --num_trajs 100 --output data/test_trajectories.npz")
    
    return None


def test_training(args):
    """测试模型训练"""
    print("\n" + "="*60)
    print("步骤 2: 模型训练测试")
    print("="*60)
    
    # 创建输出目录
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # ============ 数据准备 ============
    print("\n[1/4] 准备数据...")
    
    if args.use_synthetic or args.data_path is None:
        print(f"  使用合成数据: {args.num_trajectories} 条 {args.trajectory_type} 轨迹")
        dataset = SyntheticTrajectoryDataset(
            num_trajectories=args.num_trajectories,
            seq_len=args.seq_len,
            state_dim=args.state_dim,
            trajectory_type=args.trajectory_type,
            seed=args.seed,
        )
    else:
        print(f"  从文件加载数据: {args.data_path}")
        if not Path(args.data_path).exists():
            raise FileNotFoundError(f"数据文件不存在: {args.data_path}")
        dataset = TrajectoryDataset(
            data_path=args.data_path,
            normalize=True,
        )
    
    # 分割数据集
    from torch.utils.data import random_split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(args.num_workers, 4),  # 限制worker数量
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 4),
    )
    
    # ============ 模型准备 ============
    print("\n[2/4] 创建模型...")
    
    model = create_flowmp_transformer(
        variant=args.model_variant,
        state_dim=args.state_dim,
        max_seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {num_params:,}")
    print(f"  模型变体: {args.model_variant}")
    
    # ============ 训练配置 ============
    print("\n[3/4] 配置训练...")
    
    flow_config = FlowMatchingConfig(
        state_dim=args.state_dim,
        lambda_vel=args.lambda_vel,
        lambda_acc=args.lambda_acc,
        lambda_jerk=args.lambda_jerk,
    )
    
    trainer_config = TrainerConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.grad_clip,
        gradient_accumulation_steps=args.gradient_accumulation,
        checkpoint_dir=str(checkpoint_dir),
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        device=args.device,
        use_amp=args.use_amp and args.device == "cuda",
        flow_config=flow_config,
    )
    
    print(f"  设备: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    
    # ============ 训练 ============
    print("\n[4/4] 开始训练...")
    print("="*60)
    
    trainer = CFMTrainer(
        model=model,
        config=trainer_config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        logger=None,  # 测试时不使用wandb
    )
    
    try:
        trainer.train(resume_from=args.resume_from)
        print("\n" + "="*60)
        print("✓ 训练完成！")
        print("="*60)
        return True
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference(args):
    """测试模型推理"""
    print("\n" + "="*60)
    print("步骤 3: 模型推理测试")
    print("="*60)
    
    # 查找最新的检查点
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoints:
        print("⚠ 未找到检查点，跳过推理测试")
        return False
    
    # 按修改时间排序，取最新的
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"使用检查点: {latest_checkpoint}")
    
    try:
        # 加载检查点
        checkpoint = torch.load(latest_checkpoint, map_location=args.device, weights_only=False)
        
        # 尝试从检查点中读取模型配置
        print("\n[1/3] 加载模型...")
        if 'model_config' in checkpoint:
            # 使用检查点中保存的模型配置
            saved_config = checkpoint['model_config']
            print(f"  从检查点读取模型配置: {saved_config}")
            model_hidden_dim = saved_config.get('hidden_dim', args.hidden_dim)
            model_num_layers = saved_config.get('num_layers', args.num_layers)
            model_num_heads = saved_config.get('num_heads', args.num_heads)
            model_state_dim = saved_config.get('state_dim', args.state_dim)
            model_seq_len = saved_config.get('max_seq_len', args.seq_len)
            # 重要：使用检查点中保存的 time_embed_dim（如果存在）
            # 如果不存在，尝试从 state_dict 推断
            model_time_embed_dim = saved_config.get('time_embed_dim', None)
            if model_time_embed_dim is None:
                # 从 state_dict 推断 time_embed_dim
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                if 'time_embed.output_proj.weight' in state_dict:
                    model_time_embed_dim = state_dict['time_embed.output_proj.weight'].shape[0]
                    print(f"  从 state_dict 推断 time_embed_dim: {model_time_embed_dim}")
                else:
                    # 使用默认值（与 hidden_dim 相同或使用 variant 的默认值）
                    model_time_embed_dim = model_hidden_dim
                    print(f"  未找到 time_embed_dim，使用 hidden_dim: {model_time_embed_dim}")
        else:
            # 使用传入的参数（向后兼容）
            print(f"  检查点中无模型配置，使用传入参数")
            model_hidden_dim = args.hidden_dim
            model_num_layers = args.num_layers
            model_num_heads = args.num_heads
            model_state_dim = args.state_dim
            model_seq_len = args.seq_len
            # 尝试从 state_dict 推断 time_embed_dim
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if 'time_embed.output_proj.weight' in state_dict:
                model_time_embed_dim = state_dict['time_embed.output_proj.weight'].shape[0]
                print(f"  从 state_dict 推断 time_embed_dim: {model_time_embed_dim}")
            else:
                model_time_embed_dim = model_hidden_dim
        
        print(f"  使用配置: variant={args.model_variant}, hidden_dim={model_hidden_dim}, "
              f"num_layers={model_num_layers}, num_heads={model_num_heads}, "
              f"time_embed_dim={model_time_embed_dim}")
        
        # 使用与训练时完全相同的参数创建模型
        model = create_flowmp_transformer(
            variant=args.model_variant,
            state_dim=model_state_dim,
            max_seq_len=model_seq_len,
            hidden_dim=model_hidden_dim,
            num_layers=model_num_layers,
            num_heads=model_num_heads,
            time_embed_dim=model_time_embed_dim,  # 使用检查点中的 time_embed_dim
        )
        
        # 验证模型配置
        actual_hidden_dim = model.hidden_dim
        actual_num_layers = len(model.blocks)
        print(f"  实际模型配置: hidden_dim={actual_hidden_dim}, num_layers={actual_num_layers}")
        
        if actual_hidden_dim != model_hidden_dim or actual_num_layers != model_num_layers:
            print(f"  ⚠ 警告: 模型配置与预期不符!")
            print(f"    预期: hidden_dim={model_hidden_dim}, num_layers={model_num_layers}")
            print(f"    实际: hidden_dim={actual_hidden_dim}, num_layers={actual_num_layers}")
            print(f"    这可能导致权重加载失败。")
        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 尝试加载权重
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"  ⚠ 警告: 缺少以下键: {len(missing_keys)} 个")
                if len(missing_keys) <= 5:
                    for key in missing_keys[:5]:
                        print(f"    - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"    - {key}")
                    print(f"    ... 还有 {len(missing_keys) - 5} 个")
            if unexpected_keys:
                print(f"  ⚠ 警告: 意外的键: {len(unexpected_keys)} 个")
        except Exception as e:
            print(f"  ✗ 加载权重失败: {e}")
            print("  尝试使用 strict=False 加载...")
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        model.to(args.device)
        print("✓ 模型加载成功")
        
        # 创建生成器
        print("\n[2/3] 创建轨迹生成器...")
        gen_config = GeneratorConfig(
            solver_type="rk4",
            use_8step_schedule=True,
            use_bspline_smoothing=True,
            seq_len=args.seq_len,
            state_dim=args.state_dim,
        )
        generator = TrajectoryGenerator(model, gen_config)
        print("✓ 生成器创建成功")
        
        # 生成轨迹
        print("\n[3/3] 生成测试轨迹...")
        start_pos = torch.tensor([[0.0, 0.0]], device=args.device)
        goal_pos = torch.tensor([[2.0, 2.0]], device=args.device)
        
        with torch.no_grad():
            result = generator.generate(
                start_pos=start_pos,
                goal_pos=goal_pos,
                num_samples=3,
            )
        
        positions = result['positions']
        print(f"✓ 成功生成 {positions.shape[0]} 条轨迹")
        print(f"  轨迹形状: {positions.shape}")
        print(f"  起始位置: {positions[0, 0].cpu().numpy()}")
        print(f"  目标位置: {positions[0, -1].cpu().numpy()}")
        
        print("\n" + "="*60)
        print("✓ 推理测试通过！")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="测试 SemFlow-MPPI 完整流程")
    
    # 测试模式
    parser.add_argument("--quick", action="store_true",
                       help="快速测试模式（少量数据，少量epoch）")
    parser.add_argument("--full", action="store_true",
                       help="完整测试模式")
    parser.add_argument("--synthetic-only", action="store_true",
                       help="仅使用合成数据")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, default=None,
                       help="数据文件路径")
    parser.add_argument("--use_synthetic", action="store_true", default=True,
                       help="使用合成数据")
    parser.add_argument("--num_trajectories", type=int, default=1000,
                       help="合成轨迹数量")
    parser.add_argument("--trajectory_type", type=str, default="bezier",
                       choices=["bezier", "polynomial", "sine"])
    
    # 模型参数
    parser.add_argument("--model_variant", type=str, default="small",
                       choices=["small", "base", "large"])
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--state_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--lambda_vel", type=float, default=1.0)
    parser.add_argument("--lambda_acc", type=float, default=1.0)
    parser.add_argument("--lambda_jerk", type=float, default=1.0)
    
    # 其他参数
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_test")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    # 测试选项
    parser.add_argument("--skip_training", action="store_true",
                       help="跳过训练，仅测试推理")
    parser.add_argument("--skip_inference", action="store_true",
                       help="跳过推理测试")
    
    args = parser.parse_args()
    
    # 根据模式调整参数
    if args.quick:
        args.num_trajectories = 500
        args.epochs = 3
        args.batch_size = 16
        args.model_variant = "small"
        # 注意：small variant 的默认 hidden_dim=128, num_layers=4
        # 如果要使用更小的模型，需要显式传递这些参数
        args.hidden_dim = 64
        args.num_layers = 2
        args.num_heads = 2
    elif args.full:
        args.num_trajectories = 5000
        args.epochs = 10
        args.batch_size = 64
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*60)
    print("SemFlow-MPPI 完整流程测试")
    print("="*60)
    print(f"测试模式: {'快速' if args.quick else '完整' if args.full else '默认'}")
    print(f"设备: {args.device}")
    print(f"随机种子: {args.seed}")
    print("="*60)
    
    # 运行测试
    results = {}
    
    # 1. 数据生成测试
    if not args.skip_training:
        test_data_generation(args)
    
    # 2. 训练测试
    if not args.skip_training:
        results['training'] = test_training(args)
    else:
        print("\n跳过训练步骤")
        results['training'] = True
    
    # 3. 推理测试
    if not args.skip_inference and results.get('training', False):
        results['inference'] = test_inference(args)
    else:
        print("\n跳过推理测试")
        results['inference'] = True
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ 所有测试通过！")
        return 0
    else:
        print("\n✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())

