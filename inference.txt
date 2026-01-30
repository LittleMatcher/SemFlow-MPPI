#!/usr/bin/env python3
"""
CFM FlowMP Inference Script

Generate trajectories using a trained Flow Matching model.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt --start 0,0 --goal 2,2
    python inference.py --checkpoint checkpoints/best_model.pt --num_samples 10 --output trajectories.npz
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cfm_flowmp.models import FlowMPTransformer, create_flowmp_transformer
from cfm_flowmp.inference import TrajectoryGenerator, GeneratorConfig
from cfm_flowmp.utils.visualization import visualize_trajectory, visualize_flow_field
from cfm_flowmp.utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Generate trajectories with CFM FlowMP")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--model_variant", type=str, default="base",
                       choices=["small", "base", "large"],
                       help="Model size variant")
    
    # Generation arguments
    parser.add_argument("--start", type=str, default="0,0",
                       help="Start position (comma-separated)")
    parser.add_argument("--goal", type=str, default="2,2",
                       help="Goal position (comma-separated)")
    parser.add_argument("--start_vel", type=str, default=None,
                       help="Start velocity (comma-separated, optional)")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of trajectories to generate")
    parser.add_argument("--num_steps", type=int, default=20,
                       help="Number of ODE integration steps")
    parser.add_argument("--solver", type=str, default="rk4",
                       choices=["euler", "midpoint", "rk4", "rk45"],
                       help="ODE solver type")
    parser.add_argument("--seq_len", type=int, default=64,
                       help="Trajectory sequence length")
    
    # Smoothing
    parser.add_argument("--no_smoothing", action="store_true",
                       help="Disable B-spline smoothing")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (.npz)")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Visualize generated trajectories")
    parser.add_argument("--save_plot", type=str, default=None,
                       help="Save visualization to file")
    parser.add_argument("--visualize_flow", action="store_true",
                       help="Visualize the flow field")
    
    # Hardware
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str, model_variant: str = "base"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        if hasattr(config, 'flow_config'):
            state_dim = config.flow_config.state_dim
        else:
            state_dim = 2
    else:
        state_dim = 2
    
    # Create model
    model = create_flowmp_transformer(
        variant=model_variant,
        state_dim=state_dim,
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def parse_coordinates(coord_str: str) -> torch.Tensor:
    """Parse comma-separated coordinates."""
    coords = [float(x.strip()) for x in coord_str.split(',')]
    return torch.tensor(coords, dtype=torch.float32)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CFM FlowMP Trajectory Generation")
    print("=" * 60)
    
    # ============ Load Model ============
    print(f"\n[1/3] Loading model from {args.checkpoint}...")
    
    model = load_model(args.checkpoint, args.device, args.model_variant)
    print(f"  Model loaded successfully")
    print(f"  Device: {args.device}")
    
    # ============ Setup Generator ============
    print("\n[2/3] Setting up generator...")
    
    gen_config = GeneratorConfig(
        solver_type=args.solver,
        num_steps=args.num_steps,
        seq_len=args.seq_len,
        use_bspline_smoothing=not args.no_smoothing,
        num_samples=args.num_samples,
    )
    
    generator = TrajectoryGenerator(model, gen_config)
    
    print(f"  Solver: {args.solver}")
    print(f"  ODE steps: {args.num_steps}")
    print(f"  Smoothing: {not args.no_smoothing}")
    
    # ============ Generate Trajectories ============
    print("\n[3/3] Generating trajectories...")
    
    # Parse conditions
    start_pos = parse_coordinates(args.start).unsqueeze(0).to(args.device)
    goal_pos = parse_coordinates(args.goal).unsqueeze(0).to(args.device)
    
    start_vel = None
    if args.start_vel:
        start_vel = parse_coordinates(args.start_vel).unsqueeze(0).to(args.device)
    
    print(f"  Start: {start_pos.squeeze().tolist()}")
    print(f"  Goal: {goal_pos.squeeze().tolist()}")
    if start_vel is not None:
        print(f"  Start velocity: {start_vel.squeeze().tolist()}")
    print(f"  Generating {args.num_samples} sample(s)...")
    
    # Generate
    with torch.no_grad():
        result = generator.generate(
            start_pos=start_pos,
            goal_pos=goal_pos,
            start_vel=start_vel,
            num_samples=args.num_samples,
            return_raw=True,
        )
    
    positions = result['positions']
    velocities = result['velocities']
    accelerations = result['accelerations']
    
    print(f"  Generated trajectory shape: {positions.shape}")
    
    # ============ Compute Metrics ============
    metrics = compute_metrics(result)
    print("\n  Trajectory Metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.4f}")
    
    # ============ Save Results ============
    if args.output:
        print(f"\n  Saving to {args.output}...")
        np.savez(
            args.output,
            positions=positions.cpu().numpy(),
            velocities=velocities.cpu().numpy(),
            accelerations=accelerations.cpu().numpy(),
            start_pos=start_pos.cpu().numpy(),
            goal_pos=goal_pos.cpu().numpy(),
        )
        print("  Saved!")
    
    # ============ Visualize ============
    if args.visualize:
        print("\n  Generating visualization...")
        
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig = visualize_trajectory(
            positions=positions,
            velocities=velocities,
            start_pos=start_pos.squeeze(),
            goal_pos=goal_pos.squeeze(),
            title=f"Generated Trajectories (n={args.num_samples})",
            save_path=args.save_plot,
        )
        
        if args.save_plot:
            print(f"  Plot saved to {args.save_plot}")
        else:
            plt.savefig("generated_trajectory.png", dpi=150, bbox_inches='tight')
            print("  Plot saved to generated_trajectory.png")
        
        plt.close()
    
    # ============ Visualize Flow Field ============
    if args.visualize_flow:
        print("\n  Generating flow field visualization...")
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Visualize at different time steps
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            fig = visualize_flow_field(
                model=model,
                t=t,
                start_pos=start_pos.squeeze(),
                goal_pos=goal_pos.squeeze(),
                grid_range=(-1, 3),
                seq_len=args.seq_len,
                save_path=f"flow_field_t{t:.2f}.png",
            )
            plt.close()
        
        print("  Flow field plots saved")
    
    print("\n" + "=" * 60)
    print("Generation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
