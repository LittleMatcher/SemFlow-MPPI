"""
ODE Solvers for Flow Matching Inference

Implements numerical integration methods for solving the ODE:
    dx/dt = v_θ(x_t, t, c)

from t=0 to t=1 to generate trajectories.

Available solvers:
- EulerSolver: Simple first-order method
- RK4Solver: Classic 4th order Runge-Kutta (recommended)
- AdaptiveRK45Solver: Adaptive step size RK45 method
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class SolverConfig:
    """Configuration for ODE solvers."""
    
    # Number of integration steps (for uniform stepping)
    num_steps: int = 20
    
    # Time range
    t_start: float = 0.0
    t_end: float = 1.0
    
    # For adaptive solvers
    atol: float = 1e-5
    rtol: float = 1e-5
    max_steps: int = 1000
    
    # Whether to return intermediate states
    return_trajectory: bool = False
    
    # Custom time schedule (overrides num_steps if provided)
    # Following "Unified Generation-Refinement Planning" approach
    time_schedule: Optional[List[float]] = None
    
    # Predefined schedules
    use_8step_schedule: bool = False  # Use aggressive 8-step schedule


# Predefined time schedules based on "Unified Generation-Refinement Planning"
SCHEDULE_8STEP = [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
SCHEDULE_UNIFORM_10 = [i/10 for i in range(11)]
SCHEDULE_UNIFORM_20 = [i/20 for i in range(21)]


class EulerSolver:
    """
    Euler method (first-order) ODE solver.
    
    Simple but less accurate. Useful for debugging and fast inference.
    Supports custom time schedules for non-uniform stepping.
    
    Update rule:
        x_{n+1} = x_n + dt * v(x_n, t_n)
    """
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
    
    def _get_time_schedule(self, num_steps: int = None) -> List[float]:
        """Get the time schedule for ODE integration."""
        if self.config.time_schedule is not None:
            return self.config.time_schedule
        elif self.config.use_8step_schedule:
            return SCHEDULE_8STEP
        else:
            n = num_steps or self.config.num_steps
            return [self.config.t_start + i * (self.config.t_end - self.config.t_start) / n 
                    for i in range(n + 1)]
    
    def solve(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        num_steps: int = None,
        time_schedule: List[float] = None,
    ) -> torch.Tensor:
        """
        Solve ODE using Euler method.
        
        Args:
            velocity_fn: Function v(x, t) returning velocity at state x and time t
            x_0: Initial state [B, T, D]
            num_steps: Number of integration steps (overrides config)
            time_schedule: Custom time schedule (overrides config)
            
        Returns:
            Final state x_1 at t=1
        """
        # Get time schedule
        if time_schedule is not None:
            schedule = time_schedule
        else:
            schedule = self._get_time_schedule(num_steps)
        
        x = x_0.clone()
        B = x_0.shape[0]
        
        trajectory = [x.clone()] if self.config.return_trajectory else None
        
        # Integrate through the time schedule
        for i in range(len(schedule) - 1):
            t_curr = schedule[i]
            t_next = schedule[i + 1]
            dt = t_next - t_curr
            
            t = torch.full((B,), t_curr, device=x_0.device, dtype=x_0.dtype)
            
            # Compute velocity and update
            v = velocity_fn(x, t)
            x = x + dt * v
            
            if trajectory is not None:
                trajectory.append(x.clone())
        
        if self.config.return_trajectory:
            return torch.stack(trajectory, dim=1)  # [B, num_steps+1, T, D]
        
        return x


class RK4Solver:
    """
    Classic 4th-order Runge-Kutta ODE solver.
    
    Provides good balance between accuracy and computational cost.
    This is the recommended solver for FlowMP inference.
    
    Supports both uniform stepping and custom time schedules as described
    in "Unified Generation-Refinement Planning" (e.g., 8-step schedule).
    
    Update rule (RK4):
        k1 = v(x_n, t_n)
        k2 = v(x_n + dt/2 * k1, t_n + dt/2)
        k3 = v(x_n + dt/2 * k2, t_n + dt/2)
        k4 = v(x_n + dt * k3, t_n + dt)
        x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
    
    def _get_time_schedule(self, num_steps: int = None) -> List[float]:
        """Get the time schedule for ODE integration."""
        # Priority: custom schedule > 8-step flag > uniform stepping
        if self.config.time_schedule is not None:
            return self.config.time_schedule
        elif self.config.use_8step_schedule:
            return SCHEDULE_8STEP
        else:
            n = num_steps or self.config.num_steps
            return [self.config.t_start + i * (self.config.t_end - self.config.t_start) / n 
                    for i in range(n + 1)]
    
    def solve(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        num_steps: int = None,
        time_schedule: List[float] = None,
    ) -> torch.Tensor:
        """
        Solve ODE using RK4 method.
        
        Args:
            velocity_fn: Function v(x, t) returning velocity at state x and time t
            x_0: Initial state [B, T, D] or [B, D]
            num_steps: Number of integration steps (overrides config, ignored if time_schedule provided)
            time_schedule: Custom time schedule (overrides config)
            
        Returns:
            Final state x_1 at t=1, or trajectory if return_trajectory=True
        """
        # Get time schedule
        if time_schedule is not None:
            schedule = time_schedule
        else:
            schedule = self._get_time_schedule(num_steps)
        
        x = x_0.clone()
        B = x_0.shape[0]
        
        trajectory = [x.clone()] if self.config.return_trajectory else None
        
        # Integrate through the time schedule
        for i in range(len(schedule) - 1):
            t_curr = schedule[i]
            t_next = schedule[i + 1]
            dt = t_next - t_curr
            
            # Current time as tensor
            t = torch.full((B,), t_curr, device=x_0.device, dtype=x_0.dtype)
            
            # RK4 stages
            k1 = velocity_fn(x, t)
            k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = velocity_fn(x + dt * k3, t + dt)
            
            # Update state
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            if trajectory is not None:
                trajectory.append(x.clone())
        
        if self.config.return_trajectory:
            return torch.stack(trajectory, dim=1)  # [B, num_steps+1, ...]
        
        return x
    
    def solve_with_intermediates(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        num_steps: int = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Solve ODE and return intermediate states and velocities.
        
        Useful for debugging and visualization.
        
        Returns:
            Tuple of (final_state, states_list, velocities_list)
        """
        num_steps = num_steps or self.config.num_steps
        dt = (self.config.t_end - self.config.t_start) / num_steps
        
        x = x_0.clone()
        B = x_0.shape[0]
        t = torch.full((B,), self.config.t_start, device=x_0.device, dtype=x_0.dtype)
        
        states = [x.clone()]
        velocities = []
        times = [t.clone()]
        
        for step in range(num_steps):
            k1 = velocity_fn(x, t)
            k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = velocity_fn(x + dt * k3, t + dt)
            
            # Store average velocity for this step
            v_avg = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            velocities.append(v_avg.clone())
            
            x = x + dt * v_avg
            t = t + dt
            
            states.append(x.clone())
            times.append(t.clone())
        
        return x, states, velocities, times


class AdaptiveRK45Solver:
    """
    Adaptive step-size RK45 (Dormand-Prince) solver.
    
    Automatically adjusts step size for accuracy/speed tradeoff.
    More accurate but potentially slower than fixed-step RK4.
    
    Uses embedded RK4/RK5 pair for error estimation.
    """
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
        
        # Dormand-Prince coefficients
        self.a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
        ]
        
        # 5th order weights
        self.b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        
        # 4th order weights
        self.b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        
        # Time fractions
        self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    
    def solve(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve ODE using adaptive RK45.
        
        Args:
            velocity_fn: Function v(x, t) returning velocity
            x_0: Initial state
            
        Returns:
            Final state at t=1
        """
        x = x_0.clone()
        B = x_0.shape[0]
        t = torch.full((B,), self.config.t_start, device=x_0.device, dtype=x_0.dtype)
        
        # Initial step size
        dt = (self.config.t_end - self.config.t_start) / self.config.num_steps
        
        trajectory = [x.clone()] if self.config.return_trajectory else None
        
        step_count = 0
        while t[0] < self.config.t_end and step_count < self.config.max_steps:
            # Don't overshoot
            dt = min(dt, self.config.t_end - t[0].item())
            
            # Compute RK stages
            k = []
            k.append(velocity_fn(x, t))
            
            for i in range(1, 7):
                x_stage = x.clone()
                for j in range(i):
                    x_stage = x_stage + dt * self.a[i][j] * k[j]
                t_stage = t + self.c[i] * dt
                k.append(velocity_fn(x_stage, t_stage))
            
            # Compute 4th and 5th order solutions
            x5 = x.clone()
            x4 = x.clone()
            for i in range(7):
                x5 = x5 + dt * self.b5[i] * k[i]
                x4 = x4 + dt * self.b4[i] * k[i]
            
            # Error estimation
            error = (x5 - x4).abs().max()
            
            # Tolerance check
            tol = self.config.atol + self.config.rtol * x.abs().max()
            
            if error < tol:
                # Accept step
                x = x5
                t = t + dt
                step_count += 1
                
                if trajectory is not None:
                    trajectory.append(x.clone())
            
            # Adjust step size
            if error > 0:
                dt = 0.9 * dt * (tol / error) ** 0.2
            else:
                dt = 2 * dt
            
            dt = max(dt, 1e-6)  # Minimum step size
        
        if self.config.return_trajectory:
            return torch.stack(trajectory, dim=1)
        
        return x


class MidpointSolver:
    """
    Midpoint method (2nd order) ODE solver.
    
    Better than Euler, simpler than RK4.
    
    Update rule:
        k1 = v(x_n, t_n)
        x_{n+1} = x_n + dt * v(x_n + dt/2 * k1, t_n + dt/2)
    """
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
    
    def solve(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        num_steps: int = None,
    ) -> torch.Tensor:
        """Solve ODE using midpoint method."""
        num_steps = num_steps or self.config.num_steps
        dt = (self.config.t_end - self.config.t_start) / num_steps
        
        x = x_0.clone()
        B = x_0.shape[0]
        t = torch.full((B,), self.config.t_start, device=x_0.device, dtype=x_0.dtype)
        
        trajectory = [x.clone()] if self.config.return_trajectory else None
        
        for step in range(num_steps):
            k1 = velocity_fn(x, t)
            k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            
            x = x + dt * k2
            t = t + dt
            
            if trajectory is not None:
                trajectory.append(x.clone())
        
        if self.config.return_trajectory:
            return torch.stack(trajectory, dim=1)
        
        return x


def create_solver(
    solver_type: str = "rk4",
    config: SolverConfig = None,
):
    """
    Factory function to create ODE solvers.
    
    Args:
        solver_type: Type of solver ("euler", "midpoint", "rk4", "rk45")
        config: Solver configuration
        
    Returns:
        ODE solver instance
    """
    solvers = {
        "euler": EulerSolver,
        "midpoint": MidpointSolver,
        "rk4": RK4Solver,
        "rk45": AdaptiveRK45Solver,
    }
    
    if solver_type not in solvers:
        raise ValueError(f"Unknown solver type: {solver_type}. Choose from {list(solvers.keys())}")
    
    return solvers[solver_type](config)
