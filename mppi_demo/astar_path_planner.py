"""
A* Path Planning Algorithm

A classic heuristic search algorithm for finding optimal paths on a grid.
Used as a baseline comparison for MPPI.
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import heapq
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from mppi_core.environment_2d import Environment2D


@dataclass
class Node:
    """A node in the A* search graph"""
    x: int
    y: int
    g: float = 0.0  # Cost from start to this node
    h: float = 0.0  # Heuristic cost from this node to goal
    parent: Optional['Node'] = None
    
    @property
    def f(self) -> float:
        """Total cost: f = g + h"""
        return self.g + self.h
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.f < other.f or (self.f == other.f and self.g < other.g)
    
    def __eq__(self, other):
        """For checking if nodes are the same"""
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        """For use in sets"""
        return hash((self.x, self.y))


class AStarPlanner:
    """
    A* path planning algorithm
    
    Finds optimal path from start to goal on a grid representation
    of the environment, avoiding obstacles.
    """
    
    def __init__(self, env: Environment2D, 
                 resolution: float = 0.2,
                 robot_radius: float = 0.3):
        """
        Args:
            env: 2D environment with obstacles
            resolution: Grid resolution (smaller = finer grid, more accurate but slower)
            robot_radius: Robot radius for collision checking
        """
        self.env = env
        self.resolution = resolution
        self.robot_radius = robot_radius
        
        # Calculate grid dimensions
        self.x_min, self.x_max = env.x_min, env.x_max
        self.y_min, self.y_max = env.y_min, env.y_max
        
        # Grid size
        self.width = int(np.ceil((self.x_max - self.x_min) / resolution))
        self.height = int(np.ceil((self.y_max - self.y_min) / resolution))
        
        # Create obstacle grid (True = obstacle, False = free)
        self.obstacle_grid = self._create_obstacle_grid()
    
    def _create_obstacle_grid(self) -> np.ndarray:
        """Create a grid representation of obstacles
        
        Returns:
            obstacle_grid: shape (height, width), True where obstacles are
        """
        grid = np.zeros((self.height, self.width), dtype=bool)
        
        # Create coordinate arrays
        x_coords = np.linspace(self.x_min, self.x_max, self.width)
        y_coords = np.linspace(self.y_min, self.y_max, self.height)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Flatten for SDF computation
        points = np.stack([X.flatten(), Y.flatten()], axis=-1)
        
        # Compute SDF for all points
        sdf = self.env.compute_sdf(points)
        sdf = sdf.reshape(self.height, self.width)
        
        # Mark obstacles: sdf < robot_radius means collision
        grid = sdf < self.robot_radius
        
        return grid
    
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices
        
        Args:
            x, y: World coordinates
        Returns:
            grid_x, grid_y: Grid indices
        """
        grid_x = int(np.floor((x - self.x_min) / self.resolution))
        grid_y = int(np.floor((y - self.y_min) / self.resolution))
        
        # Clamp to grid bounds
        grid_x = np.clip(grid_x, 0, self.width - 1)
        grid_y = np.clip(grid_y, 0, self.height - 1)
        
        return grid_x, grid_y
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates
        
        Args:
            grid_x, grid_y: Grid indices
        Returns:
            x, y: World coordinates (grid cell center)
        """
        x = self.x_min + (grid_x + 0.5) * self.resolution
        y = self.y_min + (grid_y + 0.5) * self.resolution
        return x, y
    
    def _is_valid(self, grid_x: int, grid_y: int) -> bool:
        """Check if a grid cell is valid (within bounds and not an obstacle)
        
        Args:
            grid_x, grid_y: Grid indices
        Returns:
            True if valid (free space)
        """
        if grid_x < 0 or grid_x >= self.width:
            return False
        if grid_y < 0 or grid_y >= self.height:
            return False
        return not self.obstacle_grid[grid_y, grid_x]
    
    def _heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Euclidean distance heuristic
        
        Args:
            x1, y1: Grid coordinates of current node
            x2, y2: Grid coordinates of goal node
        Returns:
            Heuristic distance
        """
        dx = x2 - x1
        dy = y2 - y1
        return np.sqrt(dx*dx + dy*dy) * self.resolution
    
    def _get_neighbors(self, node: Node) -> List[Node]:
        """Get valid neighboring nodes (8-connected grid)
        
        Args:
            node: Current node
        Returns:
            List of valid neighbor nodes
        """
        neighbors = []
        
        # 8-connected grid
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_x = node.x + dx
                new_y = node.y + dy
                
                if self._is_valid(new_x, new_y):
                    neighbor = Node(x=new_x, y=new_y)
                    neighbors.append(neighbor)
        
        return neighbors
    
    def _get_cost(self, node1: Node, node2: Node) -> float:
        """Get cost to move from node1 to node2
        
        Args:
            node1, node2: Nodes
        Returns:
            Movement cost (distance)
        """
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        
        # Euclidean distance
        dist = np.sqrt(dx*dx + dy*dy) * self.resolution
        
        # Diagonal moves cost more (optional, can use sqrt(2) * resolution)
        if dx != 0 and dy != 0:
            return dist  # Already accounts for diagonal
        else:
            return dist
    
    def plan(self, start: np.ndarray, goal: np.ndarray) -> Tuple[Optional[np.ndarray], dict]:
        """
        Plan path from start to goal using A*
        
        Args:
            start: Start position (2,)
            goal: Goal position (2,)
        Returns:
            path: Array of shape (N, 2) with waypoints, or None if no path found
            info: Dictionary with search information
        """
        # Convert to grid coordinates
        start_grid_x, start_grid_y = self._world_to_grid(start[0], start[1])
        goal_grid_x, goal_grid_y = self._world_to_grid(goal[0], goal[1])
        
        # Check if start and goal are valid
        if not self._is_valid(start_grid_x, start_grid_y):
            return None, {'error': 'Start position is in obstacle'}
        if not self._is_valid(goal_grid_x, goal_grid_y):
            return None, {'error': 'Goal position is in obstacle'}
        
        # Initialize start node
        start_node = Node(x=start_grid_x, y=start_grid_y, g=0.0)
        start_node.h = self._heuristic(start_grid_x, start_grid_y, goal_grid_x, goal_grid_y)
        
        # Initialize data structures
        open_set = []  # Priority queue (min-heap)
        heapq.heappush(open_set, start_node)
        closed_set = set()
        node_map = {(start_node.x, start_node.y): start_node}
        
        # Search statistics
        nodes_expanded = 0
        max_open_size = 1
        
        # Main search loop
        while open_set:
            # Get node with lowest f value
            current = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if current.x == goal_grid_x and current.y == goal_grid_y:
                # Reconstruct path
                path = self._reconstruct_path(current)
                info = {
                    'nodes_expanded': nodes_expanded,
                    'path_length': self._compute_path_length(path),
                    'max_open_size': max_open_size,
                    'success': True
                }
                return path, info
            
            # Mark as visited
            closed_set.add((current.x, current.y))
            nodes_expanded += 1
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                neighbor_key = (neighbor.x, neighbor.y)
                
                # Skip if already visited
                if neighbor_key in closed_set:
                    continue
                
                # Calculate tentative g cost
                tentative_g = current.g + self._get_cost(current, neighbor)
                
                # Check if we found a better path to this neighbor
                if neighbor_key in node_map:
                    existing_node = node_map[neighbor_key]
                    if tentative_g >= existing_node.g:
                        continue  # This path is not better
                    # Update existing node
                    existing_node.g = tentative_g
                    existing_node.parent = current
                    existing_node.h = self._heuristic(neighbor.x, neighbor.y, goal_grid_x, goal_grid_y)
                    # Note: We should rebuild the heap, but for simplicity we'll push again
                    # (In production, use a better priority queue that supports updates)
                    heapq.heappush(open_set, existing_node)
                else:
                    # New node
                    neighbor.g = tentative_g
                    neighbor.h = self._heuristic(neighbor.x, neighbor.y, goal_grid_x, goal_grid_y)
                    neighbor.parent = current
                    node_map[neighbor_key] = neighbor
                    heapq.heappush(open_set, neighbor)
            
            # Track max open set size
            max_open_size = max(max_open_size, len(open_set))
        
        # No path found
        return None, {
            'nodes_expanded': nodes_expanded,
            'success': False,
            'error': 'No path found'
        }
    
    def _reconstruct_path(self, goal_node: Node) -> np.ndarray:
        """Reconstruct path from goal node to start
        
        Args:
            goal_node: Goal node with parent chain
        Returns:
            path: Array of shape (N, 2) with waypoints in world coordinates
        """
        path = []
        current = goal_node
        
        while current is not None:
            x, y = self._grid_to_world(current.x, current.y)
            path.append([x, y])
            current = current.parent
        
        # Reverse to get path from start to goal
        path.reverse()
        return np.array(path)
    
    def _compute_path_length(self, path: np.ndarray) -> float:
        """Compute total path length
        
        Args:
            path: Array of shape (N, 2)
        Returns:
            Total path length
        """
        if len(path) < 2:
            return 0.0
        
        diffs = np.diff(path, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return np.sum(segment_lengths)


if __name__ == "__main__":
    # Simple test
    from mppi_core.environment_2d import Environment2D, Rectangle, Circle
    
    # Create simple environment
    bounds = (-5, 5, -5, 5)
    env = Environment2D(bounds)
    env.add_rectangle_obstacle(-2, 2, -2, 2)
    
    # Create planner
    planner = AStarPlanner(env, resolution=0.1, robot_radius=0.3)
    
    # Plan path
    start = np.array([-4.0, -4.0])
    goal = np.array([4.0, 4.0])
    
    path, info = planner.plan(start, goal)
    
    if path is not None:
        print(f"Path found! Length: {info['path_length']:.2f}")
        print(f"Nodes expanded: {info['nodes_expanded']}")
        print(f"Path has {len(path)} waypoints")
    else:
        print(f"No path found: {info.get('error', 'Unknown error')}")

