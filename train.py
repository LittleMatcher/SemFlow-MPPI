#!/usr/bin/env python3
"""
CFM FlowMP Training Script

Train a Flow Matching model for trajectory generation.

Usage:
    python train.py --config config.yaml
    python train.py --epochs 100 --batch_size 64 --lr 1e-4

Example with synthetic data:
    python train.py --synthetic --num_trajectories 5000 --epochs 50
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cfm_flowmp.models import FlowMPTransformer, create_flowmp_transformer
from cfm_flowmp.training import CFMTrainer, FlowMatchingConfig, TrainerConfig
from cfm_flowmp.data import TrajectoryDataset, SyntheticTrajectoryDataset, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train CFM FlowMP model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to trajectory data file")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data for training")
    parser.add_argument("--num_trajectories", type=int, default=5000,
                       help="Number of synthetic trajectories")
    parser.add_argument("--trajectory_type", type=str, default="bezier",
                       choices=["bezier", "polynomial", "sine"],
                       help="Type of synthetic trajectories")
    
    # Model arguments
    parser.add_argument("--model_variant", type=str, default="base",
                       choices=["small", "base", "large"],
                       help="Model size variant")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Transformer hidden dimension")
    parser.add_argument("--num_layers", type=int, default=6,
                       help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=64,
                       help="Trajectory sequence length")
    parser.add_argument("--state_dim", type=int, default=2,
                       help="State dimension (typically 2 for 2D)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Number of warmup steps")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping norm")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                       help="Gradient accumulation steps")
    
    # Loss weights
    parser.add_argument("--lambda_vel", type=float, default=1.0,
                       help="Weight for velocity field loss")
    parser.add_argument("--lambda_acc", type=float, default=1.0,
                       help="Weight for acceleration field loss")
    parser.add_argument("--lambda_jerk", type=float, default=1.0,
                       help="Weight for jerk field loss")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=5000,
                       help="Steps between checkpoints")
    parser.add_argument("--eval_interval", type=int, default=1000,
                       help="Steps between evaluations")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--use_amp", action="store_true", default=True,
                       help="Use automatic mixed precision")
    
    # Logging
    parser.add_argument("--wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="cfm-flowmp",
                       help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="W&B run name")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("=" * 60)
    print("CFM FlowMP Training")
    print("=" * 60)
    
    # ============ Data Setup ============
    print("\n[1/4] Setting up data...")
    
    if args.synthetic or args.data_path is None:
        print(f"Using synthetic {args.trajectory_type} trajectories")
        dataset = SyntheticTrajectoryDataset(
            num_trajectories=args.num_trajectories,
            seq_len=args.seq_len,
            state_dim=args.state_dim,
            trajectory_type=args.trajectory_type,
            seed=args.seed,
        )
    else:
        print(f"Loading data from {args.data_path}")
        dataset = TrajectoryDataset(
            data_path=args.data_path,
            normalize=True,
        )
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # ============ Model Setup ============
    print("\n[2/4] Setting up model...")
    
    model = create_flowmp_transformer(
        variant=args.model_variant,
        state_dim=args.state_dim,
        max_seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model variant: {args.model_variant}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Heads: {args.num_heads}")
    
    # ============ Training Setup ============
    print("\n[3/4] Setting up training...")
    
    # Flow matching config
    flow_config = FlowMatchingConfig(
        state_dim=args.state_dim,
        lambda_vel=args.lambda_vel,
        lambda_acc=args.lambda_acc,
        lambda_jerk=args.lambda_jerk,
    )
    
    # Trainer config
    trainer_config = TrainerConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.grad_clip,
        gradient_accumulation_steps=args.gradient_accumulation,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        device=args.device,
        use_amp=args.use_amp and args.device == "cuda",
        flow_config=flow_config,
    )
    
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  AMP: {trainer_config.use_amp}")
    
    # Setup logger
    logger = None
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            logger = wandb
            print("  W&B logging enabled")
        except ImportError:
            print("  W&B not installed, skipping logging")
    
    # Create trainer
    trainer = CFMTrainer(
        model=model,
        config=trainer_config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        logger=logger,
    )
    
    # ============ Training ============
    print("\n[4/4] Starting training...")
    print("=" * 60)
    
    trainer.train(resume_from=args.resume_from)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("=" * 60)
    
    if logger:
        wandb.finish()


if __name__ == "__main__":
    main()
