"""
Example: Using Warm Start (Predict-Revert-Refine) Strategy
============================================================

This example demonstrates how to use the Warm Start mechanism to accelerate
L2 trajectory generation and ensure temporal consistency in a closed-loop
control scenario.

Warm Start Strategy Overview:
1. **Predict**: Shift the optimal trajectory from t-1 forward in time
2. **Revert**: Apply OT interpolation to create latent state at τ_warm (e.g., 0.8)
3. **Refine**: Solve CFM ODE from τ_warm to 1.0 (reducing computation by ~80%)

This follows principles from:
- SDEdit: Image editing by denoising diffusion models at intermediate timesteps
- MPC: Model Predictive Control with temporal consistency
- On-Policy RL: Policy continuation across timesteps
"""

import torch
import numpy as np
from typing import Dict, Optional

from cfm_flowmp.inference.generator import TrajectoryGenerator, GeneratorConfig
from cfm_flowmp.inference.l1_reactive_control import L1ReactiveController, L1Config

# Some installations may have an earlier TrajectoryGenerator that lacks
# the full L2 warm-start API. Detect this so examples can degrade gracefully.
HAS_L2_WARM_START_API = hasattr(TrajectoryGenerator, "generate_with_warm_start")


def warm_start_control_loop_example():
    """
    Complete example of using Warm Start in a closed-loop control scenario.
    
    Scenario:
    - Robot navigating through a dynamic environment
    - At each timestep, we need to replan the trajectory
    - Use Warm Start to accelerate replanning and maintain temporal consistency
    """
    
    # ============ Setup ============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # L2 Generator Configuration
    gen_config = GeneratorConfig(
        state_dim=2,              # 2D position (x, y)
        seq_len=64,               # 64 timesteps
        solver_type="rk4",
        num_steps=20,             # Standard: 20 steps for full generation
        use_bspline_smoothing=True,
        enable_warm_start=False,  # We'll use explicit warm start
    )
    
    # L1 Controller Configuration
    l1_config = L1Config(
        n_samples_per_mode=100,
        n_control_points=10,
        time_horizon=5.0,
        n_timesteps=64,
        tube_radius=0.5,
        use_warm_start=True,      # Enable warm start in L1
        warm_start_noise_scale=0.1,
        shift_padding_mode="extrapolate",
        device=device,
    )
    
    # Create mock model (replace with your trained model)
    # For this example, we'll use a simple identity model
    class MockModel(torch.nn.Module):
        def __init__(self, state_dim):
            super().__init__()
            self.state_dim = state_dim
            
        def forward(self, x_t, t, start_pos=None, goal_pos=None, 
                   start_vel=None, goal_vel=None, env_encoding=None):
            # Simple linear interpolation velocity field
            # In practice, this would be your trained FlowMP model
            B, T, D = x_t.shape
            D_pos = D // 3  # Position dimension
            
            # Extract current positions
            pos = x_t[..., :D_pos]
            
            # Target: interpolate towards goal
            if goal_pos is not None:
                goal_expanded = goal_pos.unsqueeze(1).expand(-1, T, -1)
                velocity_field = goal_expanded - pos
            else:
                velocity_field = torch.zeros_like(pos)
            
            # Full velocity field (positions, velocities, accelerations)
            v_full = torch.cat([
                velocity_field,
                torch.zeros_like(velocity_field),
                torch.zeros_like(velocity_field)
            ], dim=-1)
            
            return v_full
    
    model = MockModel(state_dim=2).to(device)
    
    # Create generator and controller
    generator = TrajectoryGenerator(model, gen_config)
    l1_controller = L1ReactiveController(l1_config)
    
    print("=" * 80)
    print("Warm Start Control Loop Example")
    print("=" * 80)
    
    # ============ Control Loop Simulation ============
    
    # Initial conditions
    start_pos = torch.tensor([0.0, 0.0], device=device)
    goal_pos = torch.tensor([1.0, 1.0], device=device)
    
    # Warm start parameters
    tau_warm = 0.8  # Start ODE integration at t=0.8
    refine_steps = 5  # Only 5 steps from 0.8 to 1.0 (vs 20 for full generation)
    
    # Storage for previous optimal trajectory
    prev_optimal_traj = None
    
    # Simulate 5 control timesteps
    num_timesteps = 5
    
    for timestep in range(num_timesteps):
        print(f"\n{'─' * 80}")
        print(f"Timestep {timestep}")
        print(f"{'─' * 80}")
        
        # Update start position (simulate robot movement)
        if timestep > 0:
            # Move start position slightly towards goal
            progress = timestep / num_timesteps
            start_pos = torch.tensor([progress * 0.2, progress * 0.2], device=device)
        
        # ============ Trajectory Generation (L2) ============
        
        if timestep == 0 or prev_optimal_traj is None:
            # First timestep: Generate from scratch (no warm start available)
            print("\n[L2] Generating trajectory from scratch (t=0 → t=1)")
            print(f"  Method: Standard CFM with {gen_config.num_steps} ODE steps")
            
            l2_output = generator.generate(
                start_pos=start_pos.unsqueeze(0),
                goal_pos=goal_pos.unsqueeze(0),
                return_raw=True,  # Need raw output for warm start cache
            )
            
            generation_info = {
                'method': 'standard',
                'ode_steps': gen_config.num_steps,
                'time_range': '0.0 → 1.0',
            }
        
        else:
            # Subsequent timesteps: Use Warm Start
            print("\n[L2] Generating trajectory with Warm Start")
            print(f"  Step 1 (Predict): Shift previous optimal trajectory")
            print(f"  Step 2 (Revert): Apply OT interpolation at τ={tau_warm}")
            print(f"  Step 3 (Refine): Solve ODE from τ={tau_warm} to 1.0 "
                  f"with {refine_steps} steps")
            
            # Step 1 & 2: Prepare warm start latent (handled in L1)
            z_tau = l1_controller.prepare_warm_start_latent(
                prev_opt_traj=prev_optimal_traj,
                tau_warm=tau_warm,
            )
            
            print(f"  ✓ Warm start latent z_τ shape: {z_tau.shape}")
            
            # Step 3: Generate with warm start
            l2_output = generator.generate_with_warm_start(
                condition={
                    'start_pos': start_pos.unsqueeze(0),
                    'goal_pos': goal_pos.unsqueeze(0),
                },
                warm_start_latent=z_tau,
                t_start=tau_warm,
                steps=refine_steps,
            )
            
            generation_info = {
                'method': 'warm_start',
                'ode_steps': refine_steps,
                'time_range': f'{tau_warm} → 1.0',
                'speedup': f'{gen_config.num_steps / refine_steps:.1f}x',
            }
        
        print(f"\n  Generation Info:")
        for key, value in generation_info.items():
            print(f"    {key}: {value}")
        
        # ============ L1 MPPI Optimization ============
        
        print(f"\n[L1] Running MPPI optimization")
        
        # Initialize L1 with L2 output
        l1_controller.initialize_from_l2_output(l2_output)
        
        # Run MPPI optimization
        mppi_result = l1_controller.optimize(
            n_iterations=10,
            verbose=False,
        )
        
        optimal_control = mppi_result['optimal_control']  # [T, D]
        best_cost = mppi_result['best_cost']
        
        print(f"  ✓ Optimal control shape: {optimal_control.shape}")
        print(f"  ✓ Best cost: {best_cost:.4f}")
        
        # ============ Update for Next Iteration ============
        
        # Store optimal trajectory for next timestep's warm start
        # Convert positions to full state (pos + vel + acc) if needed
        if optimal_control.shape[-1] == 2:
            # Only positions, need to compute vel & acc
            positions = optimal_control  # [T, 2]
            
            dt = l1_config.time_horizon / l1_config.n_timesteps
            velocities = torch.diff(positions, dim=0) / dt
            velocities = torch.cat([velocities, velocities[-1:]], dim=0)
            
            accelerations = torch.diff(velocities, dim=0) / dt
            accelerations = torch.cat([accelerations, accelerations[-1:]], dim=0)
            
            prev_optimal_traj = torch.cat([positions, velocities, accelerations], dim=-1)
        else:
            prev_optimal_traj = optimal_control
        
        print(f"\n  ✓ Stored optimal trajectory for next timestep: {prev_optimal_traj.shape}")
    
    print(f"\n{'=' * 80}")
    print("Control Loop Complete!")
    print(f"{'=' * 80}")
    print(f"\nSummary:")
    print(f"  - First timestep: Full generation ({gen_config.num_steps} steps)")
    print(f"  - Subsequent timesteps: Warm start ({refine_steps} steps)")
    print(f"  - Computational speedup: ~{gen_config.num_steps / refine_steps:.1f}x")
    print(f"  - Temporal consistency: Maintained through trajectory shifting")


def simple_warm_start_example():
    """
    Simplified example showing just the core Warm Start API calls.
    """
    print("\n" + "=" * 80)
    print("Simple Warm Start API Example")
    print("=" * 80)
    
    device = torch.device('cpu')
    
    # Mock previous optimal trajectory
    # Shape: [T, D*3] where T=64, D=2 (pos + vel + acc = 6 total)
    T, D = 64, 2
    prev_optimal_traj = torch.randn(T, D * 3, device=device)
    
    # Create L1 controller
    l1_config = L1Config(
        n_timesteps=T,
        time_horizon=5.0,
        use_warm_start=True,
        warm_start_noise_scale=0.1,
        shift_padding_mode="extrapolate",
    )
    l1_controller = L1ReactiveController(l1_config)
    
    # ============ L1 Methods ============
    
    print("\n[L1] Warm Start Methods:")
    
    # 1. Shift trajectory (Predict)
    print("\n1. shift_trajectory(trajectory)")
    shifted_traj = l1_controller.shift_trajectory(prev_optimal_traj)
    print(f"   Input shape:  {prev_optimal_traj.shape}")
    print(f"   Output shape: {shifted_traj.shape}")
    print(f"   Operation: Forward shift by 1 timestep with padding")
    
    # 2. Prepare warm start latent (Revert)
    print("\n2. prepare_warm_start_latent(prev_opt_traj, tau_warm)")
    tau_warm = 0.8
    z_tau = l1_controller.prepare_warm_start_latent(
        prev_opt_traj=prev_optimal_traj,
        tau_warm=tau_warm,
    )
    print(f"   Input shape:  {prev_optimal_traj.shape}")
    print(f"   Output shape: {z_tau.shape}")
    print(f"   τ_warm: {tau_warm}")
    # Textual formula; avoid undefined Python variables inside f-string
    print("   Formula: z_τ = τ * shift(u*_(t-1)) + (1-τ) * ε, ε ~ N(0,I)")
    
    # ============ L2 Methods ============
    
    # L2 warm-start part requires newer TrajectoryGenerator API
    if HAS_L2_WARM_START_API:
        print("\n[L2] Warm Start Methods:")
        
        # Create generator with mock model
        class MockModel(torch.nn.Module):
            def forward(self, x_t, t, **kwargs):
                return torch.zeros_like(x_t)
        
        gen_config = GeneratorConfig(
            state_dim=D,
            seq_len=T,
            solver_type="rk4",
            num_steps=20,
        )
        generator = TrajectoryGenerator(MockModel(), gen_config)
        
        # 3. Generate with warm start (Refine)
        print("\n3. generate_with_warm_start(condition, warm_start_latent, t_start, steps)")
        
        start_pos = torch.randn(1, D, device=device)
        goal_pos = torch.randn(1, D, device=device)
        
        result = generator.generate_with_warm_start(
            condition={
                'start_pos': start_pos,
                'goal_pos': goal_pos,
            },
            warm_start_latent=z_tau,
            t_start=tau_warm,
            steps=5,
        )
        
        print(f"   Input (z_τ) shape: {z_tau.shape}")
        print(f"   Output positions:  {result['positions'].shape}")
        print(f"   Output velocities: {result['velocities'].shape}")
        print(f"   Output accel:      {result['accelerations'].shape}")
        print(f"   Integration range: t={tau_warm} → t=1.0")
        print(f"   ODE steps: 5 (vs 20 for standard generation)")
        print(f"   Speedup: ~4x")
    else:
        print("\n[L2] Warm Start Methods:")
        print("   Skipped: TrajectoryGenerator in this build has no "
              "`generate_with_warm_start` method; only L1 warm-start API is demonstrated.")
    
    print("\n" + "=" * 80)


def benchmark_warm_start_speedup():
    """
    Benchmark the computational speedup from using Warm Start.
    """
    print("\n" + "=" * 80)
    print("Warm Start Speedup Benchmark")
    print("=" * 80)
    
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Setup
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 128)
            
        def forward(self, x_t, t, **kwargs):
            # Simulate some computation
            B, T, D = x_t.shape
            x_flat = x_t.reshape(B * T, D)
            out = self.linear(x_flat)
            return out.reshape(B, T, D)
    
    model = MockModel().to(device)
    
    gen_config = GeneratorConfig(
        state_dim=2,
        seq_len=64,
        solver_type="rk4",
        num_steps=20,
        use_bspline_smoothing=False,  # Disable for fair timing
    )
    
    generator = TrajectoryGenerator(model, gen_config)
    
    # Test conditions
    B = 4  # Batch size
    start_pos = torch.randn(B, 2, device=device)
    goal_pos = torch.randn(B, 2, device=device)
    
    # Warm-up
    _ = generator.generate(start_pos, goal_pos)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark 1: Standard generation
    print("\n1. Standard Generation (t=0 → t=1):")
    print(f"   ODE steps: {gen_config.num_steps}")
    
    num_runs = 10
    times_standard = []
    
    for _ in range(num_runs):
        start_time = time.time()
        result = generator.generate(start_pos, goal_pos, return_raw=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_standard.append(time.time() - start_time)
    
    avg_time_standard = np.mean(times_standard)
    std_time_standard = np.std(times_standard)
    print(f"   Time: {avg_time_standard*1000:.2f} ± {std_time_standard*1000:.2f} ms")
    
    # Benchmark 2: Warm start generation
    print("\n2. Warm Start Generation (t=0.8 → t=1):")
    tau_warm = 0.8
    refine_steps = 5
    print(f"   ODE steps: {refine_steps}")
    print(f"   Start time: τ={tau_warm}")
    
    # Create warm start latent
    z_tau = torch.randn(B, 64, 6, device=device)
    
    times_warm = []
    
    for _ in range(num_runs):
        start_time = time.time()
        result = generator.generate_with_warm_start(
            condition={'start_pos': start_pos, 'goal_pos': goal_pos},
            warm_start_latent=z_tau,
            t_start=tau_warm,
            steps=refine_steps,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_warm.append(time.time() - start_time)
    
    avg_time_warm = np.mean(times_warm)
    std_time_warm = np.std(times_warm)
    print(f"   Time: {avg_time_warm*1000:.2f} ± {std_time_warm*1000:.2f} ms")
    
    # Speedup
    speedup = avg_time_standard / avg_time_warm
    print(f"\n{'─' * 80}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time saved: {(avg_time_standard - avg_time_warm)*1000:.2f} ms per generation")
    print(f"Theoretical speedup: {gen_config.num_steps / refine_steps:.2f}x")
    print(f"{'─' * 80}")


if __name__ == "__main__":
    # Run examples
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "Warm Start Implementation Examples" + " " * 24 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Example 1: Simple API demonstration
    simple_warm_start_example()
    
    if HAS_L2_WARM_START_API:
        # Example 2: Full control loop
        warm_start_control_loop_example()
        
        # Example 3: Speedup benchmark
        benchmark_warm_start_speedup()
    else:
        print("\nNOTE: L2 warm-start examples (control loop, speed benchmark) are "
              "skipped because `TrajectoryGenerator` in this install does not "
              "expose the expected L2 warm-start methods. "
              "Upgrade the library to enable full L2 warm-start demos.")
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 30 + "Examples Complete!" + " " * 28 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
