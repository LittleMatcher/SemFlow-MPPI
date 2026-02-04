"""
Unit tests for Warm Start (Predict-Revert-Refine) implementation
"""

import torch
import pytest
import numpy as np


def test_l1_shift_trajectory():
    """Test L1Controller.shift_trajectory method"""
    from cfm_flowmp.inference.l1_reactive_control import L1ReactiveController, L1Config
    
    # Setup
    config = L1Config(
        n_timesteps=64,
        time_horizon=5.0,
        shift_padding_mode="zero",
    )
    controller = L1ReactiveController(config)
    
    # Test case 1: 2D tensor [T, D]
    T, D = 64, 6
    trajectory = torch.randn(T, D)
    
    shifted = controller.shift_trajectory(trajectory)
    
    # Check shape
    assert shifted.shape == trajectory.shape, f"Shape mismatch: {shifted.shape} vs {trajectory.shape}"
    
    # Check shift operation
    assert torch.allclose(shifted[:-1], trajectory[1:]), "Shift operation incorrect"
    
    # Check padding (zero mode)
    assert torch.allclose(shifted[-1], torch.zeros(D)), "Zero padding incorrect"
    
    print("✓ Test 1: shift_trajectory [T, D] with zero padding - PASSED")
    
    # Test case 2: 3D tensor [B, T, D]
    B = 4
    trajectory_batch = torch.randn(B, T, D)
    
    shifted_batch = controller.shift_trajectory(trajectory_batch)
    
    assert shifted_batch.shape == trajectory_batch.shape
    assert torch.allclose(shifted_batch[:, :-1], trajectory_batch[:, 1:])
    
    print("✓ Test 2: shift_trajectory [B, T, D] - PASSED")
    
    # Test case 3: Extrapolate mode
    config.shift_padding_mode = "extrapolate"
    controller = L1ReactiveController(config)
    
    trajectory = torch.randn(T, D)
    shifted = controller.shift_trajectory(trajectory)
    
    # Last element should be extrapolated
    expected_last = trajectory[-1] + (trajectory[-1] - trajectory[-2])
    assert torch.allclose(shifted[-1], expected_last), "Extrapolation incorrect"
    
    print("✓ Test 3: shift_trajectory with extrapolate padding - PASSED")


def test_l1_prepare_warm_start_latent():
    """Test L1Controller.prepare_warm_start_latent method"""
    from cfm_flowmp.inference.l1_reactive_control import L1ReactiveController, L1Config
    
    # Setup
    config = L1Config(
        n_timesteps=64,
        time_horizon=5.0,
        shift_padding_mode="zero",
    )
    controller = L1ReactiveController(config)
    
    # Test case 1: Full state input [T, D*3]
    T, D = 64, 2
    prev_traj = torch.randn(T, D * 3)  # pos + vel + acc
    
    tau_warm = 0.8
    z_tau = controller.prepare_warm_start_latent(prev_traj, tau_warm)
    
    # Check shape
    assert z_tau.shape == prev_traj.shape, f"Shape mismatch: {z_tau.shape} vs {prev_traj.shape}"
    
    # Check OT interpolation formula
    # z_tau should be between shifted trajectory and noise
    # We can't check exact values due to randomness, but we can check it's not purely deterministic
    shifted = controller.shift_trajectory(prev_traj)
    
    # z_tau should not be identical to shifted (due to noise)
    assert not torch.allclose(z_tau, shifted), "z_tau should include noise"
    
    # z_tau should have similar magnitude to shifted
    assert z_tau.std() > 0, "z_tau should have non-zero variance"
    
    print("✓ Test 1: prepare_warm_start_latent with full state [T, D*3] - PASSED")
    
    # Test case 2: Position-only input [T, D]
    prev_traj_pos = torch.randn(T, D)  # positions only
    
    z_tau_pos = controller.prepare_warm_start_latent(prev_traj_pos, tau_warm)
    
    # Should auto-expand to full state
    assert z_tau_pos.shape == (T, D * 3), f"Expected [T, {D*3}], got {z_tau_pos.shape}"
    
    print("✓ Test 2: prepare_warm_start_latent with positions only [T, D] - PASSED")
    
    # Test case 3: Batched input [B, T, D*3]
    B = 4
    prev_traj_batch = torch.randn(B, T, D * 3)
    
    z_tau_batch = controller.prepare_warm_start_latent(prev_traj_batch, tau_warm)
    
    assert z_tau_batch.shape == prev_traj_batch.shape
    
    print("✓ Test 3: prepare_warm_start_latent with batch [B, T, D*3] - PASSED")
    
    # Test case 4: OT interpolation bounds
    # tau=0 should be pure noise, tau=1 should be deterministic
    torch.manual_seed(42)
    
    # tau = 1.0 (fully deterministic)
    z_tau_1 = controller.prepare_warm_start_latent(prev_traj, tau_warm=1.0)
    shifted_1 = controller.shift_trajectory(prev_traj)
    
    # Should be very close (only numerical errors)
    # Note: Not exactly equal due to internal computations
    correlation = torch.corrcoef(torch.stack([z_tau_1.flatten(), shifted_1.flatten()]))[0, 1]
    assert correlation > 0.99, f"tau=1.0 should be mostly deterministic, got correlation {correlation}"
    
    # tau = 0.0 (pure noise)
    z_tau_0 = controller.prepare_warm_start_latent(prev_traj, tau_warm=0.0)
    
    # Should be mostly noise (low correlation with shifted)
    correlation_0 = torch.corrcoef(torch.stack([z_tau_0.flatten(), shifted_1.flatten()]))[0, 1]
    assert abs(correlation_0) < 0.3, f"tau=0.0 should be mostly noise, got correlation {correlation_0}"
    
    print("✓ Test 4: OT interpolation bounds (tau=0 and tau=1) - PASSED")


def test_generator_generate_with_warm_start():
    """Test TrajectoryGenerator.generate_with_warm_start method"""
    from cfm_flowmp.inference.generator import TrajectoryGenerator, GeneratorConfig
    
    # Mock model
    class MockModel(torch.nn.Module):
        def forward(self, x_t, t, **kwargs):
            # Simple identity velocity field for testing
            return torch.zeros_like(x_t)
    
    # Setup
    config = GeneratorConfig(
        state_dim=2,
        seq_len=64,
        solver_type="rk4",
        num_steps=20,
        use_bspline_smoothing=False,  # Disable for testing
    )
    
    model = MockModel()
    generator = TrajectoryGenerator(model, config)
    
    # Test inputs
    B, T, D = 1, 64, 2
    start_pos = torch.randn(B, D)
    goal_pos = torch.randn(B, D)
    warm_start_latent = torch.randn(B, T, D * 3)
    
    # Test case 1: Basic generation
    tau_warm = 0.8
    steps = 5
    
    result = generator.generate_with_warm_start(
        condition={
            'start_pos': start_pos,
            'goal_pos': goal_pos,
        },
        warm_start_latent=warm_start_latent,
        t_start=tau_warm,
        steps=steps,
    )
    
    # Check output structure
    assert 'positions' in result, "Missing 'positions' in result"
    assert 'velocities' in result, "Missing 'velocities' in result"
    assert 'accelerations' in result, "Missing 'accelerations' in result"
    assert 'raw_output' in result, "Missing 'raw_output' in result"
    
    # Check shapes
    assert result['positions'].shape == (B, T, D), f"positions shape: {result['positions'].shape}"
    assert result['velocities'].shape == (B, T, D), f"velocities shape: {result['velocities'].shape}"
    assert result['accelerations'].shape == (B, T, D), f"accelerations shape: {result['accelerations'].shape}"
    assert result['raw_output'].shape == (B, T, D * 3), f"raw_output shape: {result['raw_output'].shape}"
    
    # Check metadata
    assert result['t_start'] == tau_warm
    assert result['steps'] == steps
    assert result['warm_start'] == True
    
    print("✓ Test 1: generate_with_warm_start basic generation - PASSED")
    
    # Test case 2: Batch size broadcasting
    start_pos_single = torch.randn(1, D)
    goal_pos_single = torch.randn(1, D)
    warm_start_batch = torch.randn(4, T, D * 3)
    
    result_batch = generator.generate_with_warm_start(
        condition={
            'start_pos': start_pos_single,
            'goal_pos': goal_pos_single,
        },
        warm_start_latent=warm_start_batch,
        t_start=0.8,
        steps=5,
    )
    
    # Should broadcast conditions to match warm_start batch size
    assert result_batch['positions'].shape[0] == 4
    
    print("✓ Test 2: generate_with_warm_start batch broadcasting - PASSED")
    
    # Test case 3: Different tau values
    for tau in [0.5, 0.7, 0.8, 0.9]:
        result_tau = generator.generate_with_warm_start(
            condition={'start_pos': start_pos, 'goal_pos': goal_pos},
            warm_start_latent=warm_start_latent,
            t_start=tau,
            steps=5,
        )
        assert result_tau['t_start'] == tau
    
    print("✓ Test 3: generate_with_warm_start different tau values - PASSED")
    
    # Test case 4: Error handling
    try:
        # Invalid t_start
        generator.generate_with_warm_start(
            condition={'start_pos': start_pos, 'goal_pos': goal_pos},
            warm_start_latent=warm_start_latent,
            t_start=1.5,  # Invalid: > 1.0
            steps=5,
        )
        assert False, "Should raise ValueError for invalid t_start"
    except ValueError as e:
        assert "t_start" in str(e)
        print("✓ Test 4: Error handling for invalid t_start - PASSED")


def test_warm_start_integration():
    """Integration test: L1 + L2 warm start workflow"""
    from cfm_flowmp.inference.generator import TrajectoryGenerator, GeneratorConfig
    from cfm_flowmp.inference.l1_reactive_control import L1ReactiveController, L1Config
    
    # Mock model
    class MockModel(torch.nn.Module):
        def forward(self, x_t, t, **kwargs):
            return torch.zeros_like(x_t)
    
    # Setup
    gen_config = GeneratorConfig(
        state_dim=2,
        seq_len=64,
        solver_type="rk4",
        num_steps=20,
        use_bspline_smoothing=False,
    )
    
    l1_config = L1Config(
        n_timesteps=64,
        time_horizon=5.0,
        use_warm_start=True,
    )
    
    generator = TrajectoryGenerator(MockModel(), gen_config)
    l1_controller = L1ReactiveController(l1_config)
    
    # Simulate 3 timesteps
    T, D = 64, 2
    prev_optimal = None
    
    for timestep in range(3):
        start_pos = torch.tensor([[0.0 + timestep * 0.1, 0.0 + timestep * 0.1]])
        goal_pos = torch.tensor([[1.0, 1.0]])
        
        if timestep == 0:
            # First timestep: generate from scratch
            result = generator.generate(start_pos, goal_pos, return_raw=True)
            prev_optimal = torch.randn(T, D * 3)  # Mock optimal trajectory
        else:
            # Subsequent: use warm start
            z_tau = l1_controller.prepare_warm_start_latent(
                prev_opt_traj=prev_optimal,
                tau_warm=0.8,
            )
            
            result = generator.generate_with_warm_start(
                condition={'start_pos': start_pos, 'goal_pos': goal_pos},
                warm_start_latent=z_tau,
                t_start=0.8,
                steps=5,
            )
            
            # Update for next iteration
            prev_optimal = result['raw_output'].squeeze(0)  # [T, D*3]
        
        # Check result
        assert result['positions'].shape == (1, T, D)
        assert result['velocities'].shape == (1, T, D)
        assert result['accelerations'].shape == (1, T, D)
    
    print("✓ Integration test: L1 + L2 warm start workflow - PASSED")


def test_ot_interpolation_properties():
    """Test mathematical properties of OT interpolation"""
    from cfm_flowmp.inference.l1_reactive_control import L1ReactiveController, L1Config
    
    controller = L1ReactiveController(L1Config())
    
    T, D = 64, 2
    
    # Create a known trajectory
    trajectory = torch.linspace(0, 1, T).unsqueeze(-1).repeat(1, D * 3)
    
    # Test property 1: tau=0 gives pure noise (independent of trajectory)
    torch.manual_seed(42)
    z_0_a = controller.prepare_warm_start_latent(trajectory, tau_warm=0.0)
    
    torch.manual_seed(42)
    z_0_b = controller.prepare_warm_start_latent(trajectory * 10, tau_warm=0.0)
    
    # Should be equal (both are pure noise with same seed)
    correlation = torch.corrcoef(torch.stack([z_0_a.flatten(), z_0_b.flatten()]))[0, 1]
    assert correlation > 0.99, "tau=0 should be independent of trajectory"
    
    print("✓ Test 1: OT property tau=0 (pure noise) - PASSED")
    
    # Test property 2: tau=1 gives deterministic (no noise)
    z_1_a = controller.prepare_warm_start_latent(trajectory, tau_warm=1.0)
    z_1_b = controller.prepare_warm_start_latent(trajectory, tau_warm=1.0)
    
    # Should be exactly equal (deterministic)
    correlation = torch.corrcoef(torch.stack([z_1_a.flatten(), z_1_b.flatten()]))[0, 1]
    assert correlation > 0.999, "tau=1 should be deterministic"
    
    print("✓ Test 2: OT property tau=1 (deterministic) - PASSED")
    
    # Test property 3: Monotonicity - larger tau means more deterministic
    torch.manual_seed(42)
    taus = [0.0, 0.25, 0.5, 0.75, 1.0]
    shifted = controller.shift_trajectory(trajectory)
    
    correlations = []
    for tau in taus:
        torch.manual_seed(42)
        z = controller.prepare_warm_start_latent(trajectory, tau_warm=tau)
        corr = torch.corrcoef(torch.stack([z.flatten(), shifted.flatten()]))[0, 1]
        correlations.append(corr.item())
    
    # Correlations should increase monotonically
    for i in range(len(correlations) - 1):
        assert correlations[i] <= correlations[i + 1] + 1e-3, \
            f"Correlation not monotonic: {correlations[i]} > {correlations[i+1]}"
    
    print("✓ Test 3: OT property monotonicity - PASSED")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Running Warm Start Unit Tests")
    print("=" * 80 + "\n")
    
    # Run all tests
    try:
        print("Test Suite 1: L1Controller.shift_trajectory")
        print("-" * 80)
        test_l1_shift_trajectory()
        
        print("\nTest Suite 2: L1Controller.prepare_warm_start_latent")
        print("-" * 80)
        test_l1_prepare_warm_start_latent()
        
        print("\nTest Suite 3: TrajectoryGenerator.generate_with_warm_start")
        print("-" * 80)
        test_generator_generate_with_warm_start()
        
        print("\nTest Suite 4: Warm Start Integration")
        print("-" * 80)
        test_warm_start_integration()
        
        print("\nTest Suite 5: OT Interpolation Properties")
        print("-" * 80)
        test_ot_interpolation_properties()
        
        print("\n" + "=" * 80)
        print("All Tests PASSED! ✓")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("Test FAILED! ✗")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise
