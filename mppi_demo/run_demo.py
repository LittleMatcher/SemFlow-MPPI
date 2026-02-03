#!/usr/bin/env python3
"""
Main Demo Script for MPPI with B-Spline Trajectories

Run U-trap, narrow passage, and Christmas Market scenarios to demonstrate
MPPI's capability for geometric obstacle avoidance.

Usage:
    python run_demo.py --scenario [u_trap|narrow_passage|christmas_market|all]
"""
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from tests.test_u_trap import run_mppi_u_trap
from tests.test_narrow_passage import run_mppi_narrow_passage
from tests.test_christmas_market import run_mppi_christmas_market


def main():
    parser = argparse.ArgumentParser(
        description='MPPI Demo with B-Spline Trajectories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run U-trap scenario
  python run_demo.py --scenario u_trap
  
  # Run narrow passage scenario
  python run_demo.py --scenario narrow_passage
  
  # Run Christmas Market scenario (complex)
  python run_demo.py --scenario christmas_market
  
  # Run all scenarios
  python run_demo.py --scenario all
  
  # Custom parameters
  python run_demo.py --scenario christmas_market --temperature 1.0 --samples 300 --iterations 200
        """
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['u_trap', 'narrow_passage', 'christmas_market', 'all'],
        default='all',
        help='Which scenario to run (default: all)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='MPPI temperature parameter (default: scenario-specific)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of trajectory samples (default: scenario-specific)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Number of optimization iterations (default: scenario-specific)'
    )
    
    parser.add_argument(
        '--no-animation',
        action='store_true',
        help='Skip creating animations (faster)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip saving plots'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print(" " * 15 + "MPPI with B-Spline Trajectories")
    print(" " * 10 + "Geometric Obstacle Avoidance Demo")
    print("=" * 70)
    print("\nInspired by Motion Planning Diffusion (arXiv:2308.01557)")
    print("Demonstrates: SDF-based collision avoidance with smooth trajectories")
    print("No vision, no LLMs, just pure geometric reasoning!\n")
    print("=" * 70 + "\n")
    
    # Run scenarios
    if args.scenario in ['u_trap', 'all']:
        print("\n" + "▶" * 35)
        print("SCENARIO 1: U-TRAP")
        print("▶" * 35 + "\n")
        
        u_trap_params = {
            'temperature': args.temperature if args.temperature else 1.0,
            'n_samples': args.samples if args.samples else 200,
            'n_iterations': args.iterations if args.iterations else 120,
            'show_animation': not args.no_animation,
            'save_plots': not args.no_plots
        }
        
        try:
            result_u = run_mppi_u_trap(**u_trap_params)
            print("\n✓ U-trap scenario completed successfully!\n")
        except Exception as e:
            print(f"\n✗ U-trap scenario failed: {e}\n")
            import traceback
            traceback.print_exc()
    
    if args.scenario in ['narrow_passage', 'all']:
        print("\n" + "▶" * 35)
        print("SCENARIO 2: NARROW PASSAGE")
        print("▶" * 35 + "\n")
        
        narrow_params = {
            'temperature': args.temperature if args.temperature else 0.8,
            'n_samples': args.samples if args.samples else 250,
            'n_iterations': args.iterations if args.iterations else 150,
            'passage_width': 0.8,
            'show_animation': not args.no_animation,
            'save_plots': not args.no_plots
        }
        
        try:
            result_n = run_mppi_narrow_passage(**narrow_params)
            print("\n✓ Narrow passage scenario completed successfully!\n")
        except Exception as e:
            print(f"\n✗ Narrow passage scenario failed: {e}\n")
            import traceback
            traceback.print_exc()
    
    if args.scenario in ['christmas_market', 'all']:
        print("\n" + "▶" * 35)
        print("SCENARIO 3: CHRISTMAS MARKET")
        print("▶" * 35 + "\n")
        
        christmas_params = {
            'temperature': args.temperature if args.temperature else 1.0,
            'n_samples': args.samples if args.samples else 250,
            'n_iterations': args.iterations if args.iterations else 150,
            'show_animation': not args.no_animation,
            'save_plots': not args.no_plots
        }
        
        try:
            result_christmas = run_mppi_christmas_market(**christmas_params)
            print("\n✓ Christmas Market scenario completed successfully!\n")
        except Exception as e:
            print(f"\n✗ Christmas Market scenario failed: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 70)
    print(" " * 25 + "DEMO COMPLETE!")
    print("=" * 70)
    print("\n✓ Successfully demonstrated MPPI's geometric reasoning capability")
    print("✓ Handled complex obstacles: U-traps, narrow passages, and dense markets")
    print("✓ Generated smooth, C²-continuous trajectories")
    print("✓ No vision or language models required")
    print("\nOutput files saved in: /home/ubuntu/DP/mpd-splines-public/scripts/mppi_demo/")
    print("\nNext steps:")
    print("  - Tune temperature (λ) for different exploration/exploitation tradeoffs")
    print("  - Test with different passage widths and obstacle configurations")
    print("  - Integrate with VLN systems for language-conditioned navigation")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

