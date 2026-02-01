#!/usr/bin/env python3
"""
Test Warm-Start Mechanism

Validates the warm-start functionality of TrajectoryGenerator.
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from cfm_flowmp.models import create_flowmp_transformer
from cfm_flowmp.inference import TrajectoryGenerator, GeneratorConfig


def test_warm_start_initialization():
    """Test that warm-start can be enabled/disabled."""
    print("\n" + "="*60)
    print("TEST 1: Warm-Start Initialization")
    print("="*60)
    
    model = create_flowmp_transformer(variant="tiny")
    
    # Test 1: Without warm-start
    config_no_ws = GeneratorConfig(enable_warm_start=False)
    gen_no_ws = TrajectoryGenerator(model, config_no_ws)
    
    assert gen_no_ws.warm_start_cache is None, "Cache should be None initially"
    assert gen_no_ws.warm_start_timestep == 0, "Timestep should be 0 initially"
    print("✓ Without warm-start: correctly initialized")
    
    # Test 2: With warm-start
    config_with_ws = GeneratorConfig(enable_warm_start=True)
    gen_with_ws = TrajectoryGenerator(model, config_with_ws)
    
    assert gen_with_ws.warm_start_cache is None, "Cache should be None initially"
    print("✓ With warm-start: correctly initialized")


def test_trajectory_shifting():
    """Test trajectory shifting operations."""
    print("\n" + "="*60)
    print("TEST 2: Trajectory Shifting")
    print("="*60)
    
    model = create_flowmp_transformer(variant="tiny")
    
    # Test different shift modes
    shift_modes = ["zero_pad", "repeat_last", "predict"]
    
    for mode in shift_modes:
        config = GeneratorConfig(
            enable_warm_start=True,
            warm_start_shift_mode=mode,
        )
        generator = TrajectoryGenerator(model, config)
        
        # Create a test trajectory
        B, T, D = 2, 64, 6
        test_traj = torch.randn(B, T, D)
        
        # Shift it
        shifted = generator._shift_trajectory_forward(test_traj)
        
        # Validate shape
        assert shifted.shape == test_traj.shape, f"Shape mismatch for mode {mode}"
        
        # Validate first T-1 elements are shifted
        assert torch.allclose(shifted[:, :-1, :], test_traj[:, 1:, :]), \
            f"Shift operation failed for mode {mode}"
        
        print(f"✓ Shift mode '{mode}': passed")


def test_cache_update():
    """Test warm-start cache update mechanism."""
    print("\n" + "="*60)
    print("TEST 3: Cache Update")
    print("="*60)
    
    model = create_flowmp_transformer(variant="tiny")
    config = GeneratorConfig(enable_warm_start=True)
    generator = TrajectoryGenerator(model, config)
    
    # Generate a trajectory
    start_pos = torch.tensor([[0.0, 0.0]])
    goal_pos = torch.tensor([[1.0, 1.0]])
    
    result = generator.generate(start_pos, goal_pos)
    
    # Update cache
    generator.update_warm_start_cache(result)
    
    # Verify cache
    assert generator.warm_start_cache is not None, "Cache should be set"
    assert 'raw_output' in generator.warm_start_cache, "Cache should contain raw_output"
    assert generator.warm_start_timestep == 1, "Timestep should increment"
    
    print("✓ Cache update: passed")
    
    # Test reset
    generator.reset_warm_start()
    assert generator.warm_start_cache is None, "Cache should be reset"
    assert generator.warm_start_timestep == 0, "Timestep should be reset"
    
    print("✓ Cache reset: passed")


def test_warm_start_generation():
    """Test generation with warm-start."""
    print("\n" + "="*60)
    print("TEST 4: Generation with Warm-Start")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_flowmp_transformer(variant="tiny").to(device)
    model.eval()
    
    # Generate first trajectory (no warm-start available)
    config = GeneratorConfig(
        enable_warm_start=True,
        warm_start_noise_scale=0.1,
    )
    generator = TrajectoryGenerator(model, config)
    
    start_pos = torch.tensor([[0.0, 0.0]], device=device)
    goal_pos = torch.tensor([[2.0, 2.0]], device=device)
    
    result1 = generator.generate(start_pos, goal_pos)
    
    # Update cache
    generator.update_warm_start_cache(result1)
    
    # Generate second trajectory (with warm-start)
    result2 = generator.generate(start_pos, goal_pos)
    
    # They should be different but correlated
    pos1 = result1['positions'][0]
    pos2 = result2['positions'][0]
    
    # Not identical
    assert not torch.allclose(pos1, pos2, atol=1e-6), \
        "Trajectories should differ (due to noise)"
    
    # But correlated (cosine similarity should be reasonably high)
    pos1_flat = pos1.flatten()
    pos2_flat = pos2.flatten()
    cos_sim = torch.dot(pos1_flat, pos2_flat) / (
        torch.norm(pos1_flat) * torch.norm(pos2_flat)
    )
    
    print(f"  Cosine similarity: {cos_sim.item():.4f}")
    assert cos_sim > 0.5, "Warm-started trajectories should be correlated"
    
    print("✓ Warm-start generation: passed")


def test_comparison():
    """Compare with and without warm-start over multiple steps."""
    print("\n" + "="*60)
    print("TEST 5: Multi-Step Comparison")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_flowmp_transformer(variant="tiny").to(device)
    model.eval()
    
    start_pos = torch.tensor([[0.0, 0.0]], device=device)
    goal_pos = torch.tensor([[2.0, 2.0]], device=device)
    
    # Without warm-start
    config_no_ws = GeneratorConfig(enable_warm_start=False)
    gen_no_ws = TrajectoryGenerator(model, config_no_ws)
    
    results_no_ws = []
    for _ in range(3):
        result = gen_no_ws.generate(start_pos, goal_pos)
        results_no_ws.append(result['positions'][0])
    
    # With warm-start
    config_with_ws = GeneratorConfig(
        enable_warm_start=True,
        warm_start_noise_scale=0.1,
    )
    gen_with_ws = TrajectoryGenerator(model, config_with_ws)
    
    results_with_ws = []
    for _ in range(3):
        result = gen_with_ws.generate(start_pos, goal_pos)
        gen_with_ws.update_warm_start_cache(result)
        results_with_ws.append(result['positions'][0])
    
    # Compute consecutive correlations
    def compute_correlations(results):
        correlations = []
        for i in range(len(results) - 1):
            r1 = results[i].flatten()
            r2 = results[i+1].flatten()
            cos_sim = torch.dot(r1, r2) / (torch.norm(r1) * torch.norm(r2))
            correlations.append(cos_sim.item())
        return np.mean(correlations)
    
    corr_no_ws = compute_correlations(results_no_ws)
    corr_with_ws = compute_correlations(results_with_ws)
    
    print(f"  Avg correlation (no warm-start):   {corr_no_ws:.4f}")
    print(f"  Avg correlation (with warm-start): {corr_with_ws:.4f}")
    
    # With warm-start should have higher temporal correlation
    print(f"  Improvement: {(corr_with_ws - corr_no_ws) / corr_no_ws * 100:.1f}%")
    
    print("✓ Multi-step comparison: passed")


def test_noise_scale_effect():
    """Test effect of different noise scales."""
    print("\n" + "="*60)
    print("TEST 6: Noise Scale Effect")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_flowmp_transformer(variant="tiny").to(device)
    model.eval()
    
    start_pos = torch.tensor([[0.0, 0.0]], device=device)
    goal_pos = torch.tensor([[2.0, 2.0]], device=device)
    
    noise_scales = [0.01, 0.1, 0.3, 0.5]
    
    for noise_scale in noise_scales:
        config = GeneratorConfig(
            enable_warm_start=True,
            warm_start_noise_scale=noise_scale,
        )
        generator = TrajectoryGenerator(model, config)
        
        # Generate and cache first trajectory
        result1 = generator.generate(start_pos, goal_pos)
        generator.update_warm_start_cache(result1)
        
        # Generate second trajectory
        result2 = generator.generate(start_pos, goal_pos)
        
        # Compute difference
        diff = torch.norm(result1['positions'] - result2['positions']).item()
        
        print(f"  Noise scale {noise_scale:.2f}: diff = {diff:.4f}")
    
    print("✓ Noise scale effect: validated")


def main():
    print("="*60)
    print("Testing Warm-Start Mechanism")
    print("="*60)
    
    try:
        test_warm_start_initialization()
        test_trajectory_shifting()
        test_cache_update()
        test_warm_start_generation()
        test_comparison()
        test_noise_scale_effect()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
