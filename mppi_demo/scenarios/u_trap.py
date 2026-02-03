"""
U-Trap Scenario

This scenario demonstrates MPPI's ability to handle non-convex obstacles
where the robot must navigate around a U-shaped trap.
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from mppi_core.environment_2d import Environment2D


def create_u_trap_scenario():
    """Create U-trap environment"""
    # Environment bounds
    bounds = (-5, 5, -5, 5)
    env = Environment2D(bounds)
    
    # Create U-trap in the middle
    env.create_u_trap(
        center=(0.0, 0.0),
        width=3.0,
        height=4.0,
        thickness=0.3
    )
    
    # Start and goal positions
    start = np.array([-3.5, -3.5])
    goal = np.array([0.0, 3.0])  # Inside the U-trap!
    
    return env, start, goal, bounds
