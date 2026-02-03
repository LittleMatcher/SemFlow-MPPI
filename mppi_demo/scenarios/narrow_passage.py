"""
Narrow Passage Scenario

This is a classic stress test for motion planning algorithms.
The robot must navigate through a narrow gap between two walls.
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from mppi_core.environment_2d import Environment2D


def create_narrow_passage_scenario(passage_width=0.8):
    """Create narrow passage environment
    
    Args:
        passage_width: width of the passage (smaller = harder)
    """
    # Environment bounds
    bounds = (-5, 10, -5, 5)
    env = Environment2D(bounds)
    
    # Create narrow passage
    env.create_narrow_passage(
        start=(0, 0),
        end=(8, 0),
        passage_width=passage_width,
        wall_thickness=3.0
    )
    
    # Add some additional obstacles for complexity
    env.add_circle_obstacle(center=np.array([5.0, 2.5]), radius=0.8)
    env.add_circle_obstacle(center=np.array([5.0, -2.5]), radius=0.8)
    
    # Start and goal positions
    start = np.array([-3.0, 0.0])
    goal = np.array([9.0, 0.0])
    
    return env, start, goal, bounds
