"""
RRT* Path Planning Algorithm

RRT* (Rapidly-exploring Random Tree Star) is a sampling-based motion planning
algorithm that builds a tree structure to find feasible paths.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from mppi_core.environment_2d import Environment2D
import matplotlib.pyplot as plt


@dataclass
class Node:
    """Node in RRT* tree"""
    x: float
    y: float
    parent: Optional['Node'] = None
    cost: float = 0.0  # Cost from root to this node


class RRTStarPlanner:
    """
    RRT* Path Planner
    
    RRT* improves upon RRT by rewiring the tree to find shorter paths.
    """
    
    def __init__(
        self,
        env: Environment2D,
        resolution: float = 0.1,
        robot_radius: float = 0.0,
        step_size: float = 0.5,
        goal_sample_rate: float = 0.05,
        max_iterations: int = 5000,
        search_radius: float = 1.0,
        crowd_regions: Optional[List[Any]] = None,
        density_fn: Optional[Callable[[np.ndarray], float]] = None,
        density_bias_strength: float = 0.0,
        density_edge_weight: float = 0.0,
        density_edge_samples: int = 10,
        sample_candidates: int = 12,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize RRT* planner
        
        Args:
            env: 2D environment with obstacles
            resolution: Grid resolution for collision checking
            robot_radius: Robot radius for collision checking
            step_size: Maximum step size for tree expansion
            goal_sample_rate: Probability of sampling goal (0-1)
            max_iterations: Maximum number of iterations
            search_radius: Radius for rewiring in RRT*
        """
        self.env = env
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations
        self.search_radius = search_radius

        # Optional crowd-density awareness
        # - density_bias_strength: >0 biases random sampling toward low-density regions
        # - density_edge_weight: >0 adds a density penalty to edge costs (RRT* rewiring objective)
        self.crowd_regions = crowd_regions
        self.density_fn = density_fn
        self.density_bias_strength = float(density_bias_strength)
        self.density_edge_weight = float(density_edge_weight)
        self.density_edge_samples = int(max(2, density_edge_samples))
        self.sample_candidates = int(max(1, sample_candidates))
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Get environment bounds
        self.x_min, self.x_max = env.x_min, env.x_max
        self.y_min, self.y_max = env.y_min, env.y_max
        
        # Tree storage
        self.nodes: List[Node] = []
        self.goal_node: Optional[Node] = None
    
    def _is_valid(self, x: float, y: float) -> bool:
        """Check if point is in valid (collision-free) space"""
        point = np.array([x, y])
        sdf = self.env.compute_sdf(point.reshape(1, -1))[0]
        return sdf >= self.robot_radius

    def _density_multiplier(self, point: np.ndarray) -> float:
        """Return crowd-density multiplier at a point (>=1.0 typically).

        If no density source is provided, returns 1.0.
        """
        if self.density_fn is not None:
            try:
                return float(self.density_fn(point))
            except Exception:
                return 1.0

        if self.crowd_regions is not None:
            for region in self.crowd_regions:
                try:
                    if region.contains_point(point):
                        return float(region.get_density(point))
                except Exception:
                    continue

        return 1.0

    def _edge_cost(self, start: np.ndarray, end: np.ndarray) -> float:
        """Edge cost used by RRT* objective.

        Base: Euclidean distance.
        Optional: adds density penalty proportional to segment length and average density above 1.0.
        """
        dist = float(np.linalg.norm(end - start))
        if dist <= 0.0:
            return 0.0

        if self.density_edge_weight <= 0.0:
            return dist

        # Average density along the segment
        densities = []
        for i in range(self.density_edge_samples):
            alpha = i / (self.density_edge_samples - 1)
            p = start + alpha * (end - start)
            densities.append(self._density_multiplier(p))
        avg_density = float(np.mean(densities)) if densities else 1.0
        excess = max(0.0, avg_density - 1.0)

        return dist + self.density_edge_weight * dist * excess
    
    def _is_path_valid(self, start: np.ndarray, end: np.ndarray, n_checks: int = 20) -> bool:
        """Check if path segment is collision-free"""
        for i in range(n_checks + 1):
            alpha = i / n_checks
            point = start + alpha * (end - start)
            if not self._is_valid(point[0], point[1]):
                return False
        return True
    
    def _sample_random_point(self, goal: np.ndarray) -> np.ndarray:
        """Sample a random point (with bias towards goal)"""
        if self.rng.random() < self.goal_sample_rate:
            return goal

        # If no density bias requested, fall back to uniform sampling.
        has_density = (self.density_fn is not None) or (self.crowd_regions is not None)
        if (not has_density) or (self.density_bias_strength <= 0.0):
            x = self.rng.uniform(self.x_min, self.x_max)
            y = self.rng.uniform(self.y_min, self.y_max)
            return np.array([x, y])

        # Density-biased sampling:
        # sample several candidates and pick one with probability favoring low density.
        candidates = []
        weights = []
        for _ in range(self.sample_candidates):
            x = self.rng.uniform(self.x_min, self.x_max)
            y = self.rng.uniform(self.y_min, self.y_max)
            p = np.array([x, y])

            d = self._density_multiplier(p)
            # Higher density => lower weight
            w = float(np.exp(-self.density_bias_strength * max(0.0, d - 1.0)))
            candidates.append(p)
            weights.append(w)

        weights_arr = np.array(weights, dtype=float)
        if np.all(weights_arr <= 0) or not np.isfinite(weights_arr).all():
            return candidates[int(self.rng.integers(0, len(candidates)))]

        probs = weights_arr / np.sum(weights_arr)
        idx = int(self.rng.choice(len(candidates), p=probs))
        return candidates[idx]
    
    def _find_nearest_node(self, point: np.ndarray) -> Node:
        """Find nearest node in tree to given point"""
        min_dist = float('inf')
        nearest = None
        
        for node in self.nodes:
            dist = np.linalg.norm([node.x - point[0], node.y - point[1]])
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_node: Node, to_point: np.ndarray) -> Tuple[float, float]:
        """Steer from node towards point with step size limit"""
        direction = np.array([to_point[0] - from_node.x, to_point[1] - from_node.y])
        dist = np.linalg.norm(direction)
        
        if dist <= self.step_size:
            return to_point[0], to_point[1]
        else:
            direction = direction / dist * self.step_size
            new_x = from_node.x + direction[0]
            new_y = from_node.y + direction[1]
            return new_x, new_y
    
    def _calculate_cost(self, node: Node) -> float:
        """Calculate cost from root to node"""
        cost = 0.0
        current = node
        while current.parent is not None:
            start = np.array([current.parent.x, current.parent.y])
            end = np.array([current.x, current.y])
            cost += self._edge_cost(start, end)
            current = current.parent
        return float(cost)
    
    def _find_near_nodes(self, new_node: Node) -> List[Node]:
        """Find nodes within search radius for rewiring"""
        near_nodes = []
        for node in self.nodes:
            dist = np.sqrt((node.x - new_node.x)**2 + (node.y - new_node.y)**2)
            if dist <= self.search_radius:
                near_nodes.append(node)
        return near_nodes
    
    def _choose_parent(self, new_node: Node, near_nodes: List[Node]) -> Node:
        """Choose best parent from near nodes (RRT* feature)"""
        if not near_nodes:
            return new_node
        
        min_cost = float('inf')
        best_parent = None
        
        for near_node in near_nodes:
            # Check if path from near_node to new_node is valid
            start = np.array([near_node.x, near_node.y])
            end = np.array([new_node.x, new_node.y])
            
            if self._is_path_valid(start, end):
                # Calculate cost through this parent
                parent_cost = float(near_node.cost)
                edge_cost = self._edge_cost(start, end)
                total_cost = parent_cost + float(edge_cost)
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_parent = near_node
        
        if best_parent is not None:
            new_node.parent = best_parent
            new_node.cost = min_cost
        
        return new_node
    
    def _rewire(self, new_node: Node, near_nodes: List[Node]):
        """Rewire tree to improve path quality (RRT* feature)"""
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue
            
            # Check if path from new_node to near_node is valid
            start = np.array([new_node.x, new_node.y])
            end = np.array([near_node.x, near_node.y])
            
            if self._is_path_valid(start, end):
                # Calculate new cost through new_node
                new_cost = float(new_node.cost) + float(self._edge_cost(start, end))
                
                # If new cost is better, rewire
                if new_cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    # Update costs of children (simplified - in full implementation,
                    # we'd recursively update all descendants)
    
    def plan(self, start: np.ndarray, goal: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Plan path using RRT*
        
        Args:
            start: Start position (2,)
            goal: Goal position (2,)
        
        Returns:
            path: Path as array of waypoints (N, 2) or None if no path found
            info: Dictionary with planning information
        """
        # Reset tree
        self.nodes = []
        self.goal_node = None
        
        # Check start and goal validity
        if not self._is_valid(start[0], start[1]):
            return None, {'error': 'Start position is in collision'}
        
        if not self._is_valid(goal[0], goal[1]):
            return None, {'error': 'Goal position is in collision'}
        
        # Initialize tree with start node
        start_node = Node(x=start[0], y=start[1], parent=None, cost=0.0)
        self.nodes.append(start_node)
        
        goal_threshold = self.step_size * 1.5
        
        # RRT* main loop
        for iteration in range(self.max_iterations):
            # Sample random point
            random_point = self._sample_random_point(goal)
            
            # Find nearest node
            nearest_node = self._find_nearest_node(random_point)
            
            # Steer towards random point
            new_x, new_y = self._steer(nearest_node, random_point)
            
            # Check if path is valid
            start_pos = np.array([nearest_node.x, nearest_node.y])
            end_pos = np.array([new_x, new_y])
            
            if not self._is_path_valid(start_pos, end_pos):
                continue
            
            # Create new node
            new_node = Node(x=new_x, y=new_y)
            
            # Find near nodes for rewiring
            near_nodes = self._find_near_nodes(new_node)
            near_nodes.append(nearest_node)
            
            # Choose best parent
            new_node = self._choose_parent(new_node, near_nodes)

            # If no parent was assigned by _choose_parent, connect to nearest_node
            if new_node.parent is None:
                new_node.parent = nearest_node
                new_node.cost = float(nearest_node.cost) + float(self._edge_cost(start_pos, end_pos))
            
            # Add to tree
            self.nodes.append(new_node)
            
            # Rewire tree
            self._rewire(new_node, near_nodes)
            
            # Check if goal is reached
            dist_to_goal = np.sqrt((new_x - goal[0])**2 + (new_y - goal[1])**2)
            if dist_to_goal < goal_threshold:
                # Check if direct path to goal is valid
                goal_start = np.array([new_x, new_y])
                if self._is_path_valid(goal_start, goal):
                    goal_node = Node(x=goal[0], y=goal[1], parent=new_node)
                    goal_node.cost = float(new_node.cost) + float(self._edge_cost(goal_start, goal))
                    self.nodes.append(goal_node)
                    self.goal_node = goal_node
                    break
        
        # Reconstruct path
        if self.goal_node is None:
            return None, {
                'error': 'No path found',
                'iterations': iteration + 1,
                'nodes': len(self.nodes)
            }
        
        path = self._reconstruct_path(self.goal_node)
        path_length = self._calculate_path_length(path)
        path_cost = float(self.goal_node.cost)
        
        return path, {
            'path_length': path_length,
            'path_cost': path_cost,
            'iterations': iteration + 1,
            'nodes': len(self.nodes),
            'goal_node': self.goal_node
        }
    
    def _reconstruct_path(self, goal_node: Node) -> np.ndarray:
        """Reconstruct path from goal to start"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append([current.x, current.y])
            current = current.parent
        
        path.reverse()
        return np.array(path)
    
    def _calculate_path_length(self, path: np.ndarray) -> float:
        """Calculate total path length"""
        if len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i+1] - path[i])
        return length
    
    def visualize_tree(self, ax, show_path: bool = True, path_color: str = 'blue'):
        """Visualize RRT* tree"""
        # Draw all edges
        for node in self.nodes:
            if node.parent is not None:
                ax.plot([node.parent.x, node.x], [node.parent.y, node.y],
                       color='gray', linewidth=0.5, alpha=0.3, zorder=1)
        
        # Draw path if goal is reached
        if show_path and self.goal_node is not None:
            path = self._reconstruct_path(self.goal_node)
            ax.plot(path[:, 0], path[:, 1],
                   color=path_color, linewidth=3, label='RRT* Path', zorder=5)
        
        # Draw nodes
        if self.nodes:
            node_x = [node.x for node in self.nodes]
            node_y = [node.y for node in self.nodes]
            ax.scatter(node_x, node_y, color='lightblue', s=10, alpha=0.5, zorder=2)
        
        # Draw start and goal
        if self.nodes:
            start_node = self.nodes[0]
            ax.scatter(start_node.x, start_node.y, color='green', s=100,
                      marker='o', edgecolor='black', linewidth=2, zorder=10,
                      label='Start')
        
        if self.goal_node is not None:
            ax.scatter(self.goal_node.x, self.goal_node.y, color='red', s=100,
                      marker='*', edgecolor='black', linewidth=2, zorder=10,
                      label='Goal')


if __name__ == "__main__":
    # Simple test
    from mppi_core.environment_2d import Environment2D, Rectangle, Circle
    
    # Create simple test environment
    bounds = (-5, 5, -5, 5)
    env = Environment2D(bounds=bounds)
    env.add_rectangle_obstacle(-2, -1, -1, 1)
    env.add_circle_obstacle(center=np.array([2, 2]), radius=0.8)
    
    # Create planner
    planner = RRTStarPlanner(env, resolution=0.1, robot_radius=0.2, 
                            step_size=0.5, max_iterations=2000)
    
    # Plan path
    start = np.array([-4.0, -4.0])
    goal = np.array([4.0, 4.0])
    
    path, info = planner.plan(start, goal)
    
    if path is not None:
        print(f"Path found! Length: {info['path_length']:.2f}")
        print(f"Iterations: {info['iterations']}")
        print(f"Nodes: {info['nodes']}")
        print(f"Path has {len(path)} waypoints")
        
        # Visualize
        from mppi_core.visualization import Visualizer
        vis = Visualizer(env, figsize=(10, 10))
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        vis.plot_environment(ax)
        planner.visualize_tree(ax)
        ax.set_title('RRT* Path Planning')
        ax.legend()
        plt.savefig('rrt_star_test.png', dpi=150, bbox_inches='tight')
        print("Saved: rrt_star_test.png")
    else:
        print(f"No path found: {info.get('error', 'Unknown error')}")

