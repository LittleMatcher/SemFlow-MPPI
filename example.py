#!/usr/bin/env python3
"""
CFM FlowMP Example Script

A simple example demonstrating the complete training and inference pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

# Import CFM FlowMP modules
from cfm_flowmp.models import create_flowmp_transformer
from cfm_flowmp.training import CFMTrainer, TrainerConfig, FlowMatchingConfig
from cfm_flowmp.inference import TrajectoryGenerator, GeneratorConfig
from cfm_flowmp.data import SyntheticTrajectoryDataset, create_dataloader


def main():
    print("=" * 60)
    print("CFM FlowMP - Complete Example")
    print("=" * 60)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # ============ 1. Create Synthetic Dataset ============
    print("\n[1] Creating synthetic trajectory dataset...")
    
    dataset = SyntheticTrajectoryDataset(
        num_trajectories=1000,  # Small dataset for demo
        seq_len=64,
        state_dim=2,
        trajectory_type="bezier",
        noise_std=0.01,
        seed=42,
    )
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = create_dataloader(train_dataset, batch_size=32, num_workers=0)
    val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # ============ 2. Create Model ============
    print("\n[2] Creating FlowMP Transformer model...")
    
    model = create_flowmp_transformer(
        variant="small",  # Use small model for demo
        state_dim=2,
        max_seq_len=64,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # ============ 3. Training ============
    print("\n[3] Training model (short demo)...")
    
    # Configure training
    flow_config = FlowMatchingConfig(
        state_dim=2,
        lambda_vel=1.0,
        lambda_acc=1.0,
        lambda_jerk=1.0,
    )
    
    trainer_config = TrainerConfig(
        num_epochs=5,  # Short training for demo
        batch_size=32,
        learning_rate=1e-3,
        warmup_steps=100,
        device=device,
        use_amp=device == "cuda",
        checkpoint_dir="./demo_checkpoints",
        log_interval=50,
        eval_interval=100,
        save_interval=500,
        flow_config=flow_config,
    )
    
    trainer = CFMTrainer(
        model=model,
        config=trainer_config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )
    
    # Train
    trainer.train()
    
    # ============ 4. Inference ============
    print("\n[4] Generating trajectories...")
    
    # Create generator
    gen_config = GeneratorConfig(
        solver_type="rk4",
        num_steps=20,
        state_dim=2,
        seq_len=64,
        use_bspline_smoothing=True,
    )
    
    generator = TrajectoryGenerator(model, gen_config)
    
    # Generate trajectories
    start_pos = torch.tensor([[0.0, 0.0]], device=device)
    goal_pos = torch.tensor([[2.0, 2.0]], device=device)
    
    with torch.no_grad():
        result = generator.generate(
            start_pos=start_pos,
            goal_pos=goal_pos,
            num_samples=3,
        )
    
    positions = result['positions']
    velocities = result['velocities']
    accelerations = result['accelerations']
    
    print(f"  Generated {positions.shape[0]} trajectories")
    print(f"  Position shape: {positions.shape}")
    print(f"  Velocity shape: {velocities.shape}")
    
    # ============ 5. Visualize Results ============
    print("\n[5] Saving visualization...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        from cfm_flowmp.utils.visualization import visualize_trajectory
        from cfm_flowmp.utils.metrics import compute_metrics
        
        # Compute metrics
        metrics = compute_metrics(result)
        print("  Trajectory Metrics:")
        for key, value in list(metrics.items())[:5]:  # Print first 5 metrics
            print(f"    {key}: {value:.4f}")
        
        # Visualize
        fig = visualize_trajectory(
            positions=positions,
            velocities=velocities,
            start_pos=start_pos.squeeze(),
            goal_pos=goal_pos.squeeze(),
            title="Generated Trajectories (Demo)",
            save_path="demo_trajectories.png",
        )
        plt.close()
        
        print("  Visualization saved to: demo_trajectories.png")
        
    except ImportError as e:
        print(f"  Visualization skipped (matplotlib not available): {e}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
