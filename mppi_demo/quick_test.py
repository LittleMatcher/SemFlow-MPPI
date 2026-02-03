"""
Quick sanity check to verify all components work
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

print("Testing MPPI implementation...")
print("=" * 60)

# Test 1: Environment and SDF
print("\n1. Testing Environment and SDF...")
from mppi_core.environment_2d import Environment2D

env = Environment2D((-5, 5, -5, 5))
env.add_circle_obstacle(center=np.array([0, 0]), radius=1.0)

points = np.array([[2, 0], [0, 0], [-2, 0]])
sdf = env.compute_sdf(points)
print(f"   SDF values: {sdf}")
print(f"   ✓ SDF working (outside: {sdf[0]:.2f}, inside: {sdf[1]:.2f})")

# Test 2: B-Spline
print("\n2. Testing B-Spline Trajectory...")
from mppi_core.bspline_trajectory import BSplineTrajectory

bspline = BSplineTrajectory(degree=3, n_control_points=10, time_horizon=5.0)
control_points = np.random.randn(10, 2)
traj = bspline.evaluate(control_points, n_samples=50)
print(f"   Trajectory shape: {traj.shape}")
print(f"   ✓ B-spline evaluation working")

# Test 3: Dynamics
print("\n3. Testing Dynamics...")
from mppi_core.dynamics import DifferentialDriveRobot

robot = DifferentialDriveRobot(dt=0.1)
state = robot.create_state(0, 0, 0, 0, 0)
controls = np.zeros((10, 2))
states = robot.rollout(state, controls)
print(f"   States shape: {states.shape}")
print(f"   ✓ Dynamics rollout working")

# Test 4: Cost Functions
print("\n4. Testing Cost Functions...")
from mppi_core.cost_functions import CollisionCost, SmoothnessCost, GoalCost, CompositeCost

collision_cost = CollisionCost(env, robot_radius=0.2, weight=100.0)
smooth_cost = SmoothnessCost(penalize='acceleration', weight=1.0)
goal_cost = GoalCost(goal=np.array([3, 3]), weight=50.0)

cost_fn = CompositeCost([collision_cost, smooth_cost, goal_cost])

# Create dummy trajectory
dummy_traj = np.random.randn(5, 50, 2)
dummy_accel = np.random.randn(5, 50, 2) * 0.1

costs = cost_fn(positions=dummy_traj, accelerations=dummy_accel)
print(f"   Costs shape: {costs.shape}")
print(f"   Cost values: {costs[:3]}")
print(f"   ✓ Cost function working")

# Test 5: MPPI (minimal)
print("\n5. Testing MPPI (minimal run)...")
from mppi_core.mppi import MPPI_BSpline

mppi = MPPI_BSpline(
    cost_function=cost_fn,
    n_samples=20,  # Very few samples for quick test
    n_control_points=8,
    time_horizon=3.0,
    n_timesteps=20,
    temperature=1.0,
    noise_std=0.5,
    bounds=(-5, 5, -5, 5)
)

start = np.array([-2, -2])
goal = np.array([2, 2])

result = mppi.optimize(start, goal, n_iterations=5, verbose=False)
print(f"   Final trajectory shape: {result['trajectory'].shape}")
print(f"   Cost history: {result['cost_history']}")
print(f"   ✓ MPPI optimization working")

# Test 6: Visualization (no display)
print("\n6. Testing Visualization...")
from mppi_core.visualization import Visualizer
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

vis = Visualizer(env)
fig, ax = plt.subplots()
vis.plot_environment(ax)
vis.plot_trajectory(result['trajectory'], ax=ax)
plt.close(fig)
print(f"   ✓ Visualization working")

print("\n" + "=" * 60)
print("✓ All components working correctly!")
print("=" * 60)
print("\nYou can now run the full demos:")
print("  python test_u_trap.py")
print("  python test_narrow_passage.py")
print("  python run_demo.py --scenario both")

