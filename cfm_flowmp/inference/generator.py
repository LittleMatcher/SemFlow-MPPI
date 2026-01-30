"""
Trajectory Generator for FlowMP

Complete trajectory generation pipeline:
1. Sample initial noise
2. Solve ODE from t=0 to t=1
3. Post-process with B-spline smoothing for physical consistency
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .ode_solver import RK4Solver, SolverConfig, create_solver


@dataclass
class GeneratorConfig:
    """Configuration for trajectory generator."""
    
    # Solver settings
    solver_type: str = "rk4"
    num_steps: int = 20  # Used for uniform stepping
    
    # Time schedule (as per "Unified Generation-Refinement Planning")
    # Non-uniform schedule: large steps early, small steps near t=1
    use_8step_schedule: bool = True  # Use aggressive 8-step schedule by default
    custom_time_schedule: list = None  # Override with custom schedule
    
    # Custom time schedule (overrides num_steps if provided)
    # Example: [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0] for 8-step schedule
    # This non-uniform schedule uses larger steps early and smaller steps near t=1
    # for better fine-grained control in the refinement phase
    time_schedule: Optional[List[float]] = None
    
    # State dimensions
    state_dim: int = 2
    seq_len: int = 64
    
    # Smoothing (B-spline fitting for physical consistency)
    use_bspline_smoothing: bool = True
    bspline_degree: int = 3
    bspline_num_control_points: int = 20
    
    # Sampling
    num_samples: int = 1  # Number of trajectories to generate per condition


# 8-step schedule from "Unified Generation-Refinement Planning"
# Front-loaded: large steps early (exploration), small steps late (refinement)
DEFAULT_8STEP_SCHEDULE = [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]


class BSplineSmoother:
    """
    B-spline smoothing for trajectory post-processing.
    
    Fits a B-spline to the generated trajectory to ensure:
    - Smoothness (continuous derivatives)
    - Physical consistency
    - Reduced noise from ODE integration errors
    """
    
    def __init__(
        self,
        degree: int = 3,
        num_control_points: int = 20,
    ):
        """
        Args:
            degree: B-spline degree (3 = cubic)
            num_control_points: Number of control points for fitting
        """
        self.degree = degree
        self.num_control_points = num_control_points
    
    def smooth(
        self,
        trajectory: torch.Tensor,
        return_derivatives: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Smooth trajectory using B-spline fitting.
        
        Args:
            trajectory: Position trajectory [B, T, D]
            return_derivatives: Whether to compute velocity and acceleration
            
        Returns:
            Dictionary with smoothed 'positions', 'velocities', 'accelerations'
        """
        B, T, D = trajectory.shape
        device = trajectory.device
        dtype = trajectory.dtype
        
        # Convert to numpy for scipy operations
        traj_np = trajectory.detach().cpu().numpy()
        
        try:
            from scipy.interpolate import splprep, splev
            
            smoothed_positions = []
            smoothed_velocities = []
            smoothed_accelerations = []
            
            # Parameter values for the trajectory points
            t_eval = np.linspace(0, 1, T)
            
            for b in range(B):
                traj_b = traj_np[b]  # [T, D]
                
                # Transpose for splprep: expects [D, T]
                traj_b_t = traj_b.T
                
                # Fit B-spline
                # s=0 for interpolation, s>0 for smoothing
                smoothing_factor = max(0, T - np.sqrt(2 * T))  # Adaptive smoothing
                
                try:
                    tck, u = splprep(
                        traj_b_t,
                        k=self.degree,
                        s=smoothing_factor,
                    )
                    
                    # Evaluate at original parameter values
                    pos = np.array(splev(t_eval, tck)).T  # [T, D]
                    
                    if return_derivatives:
                        # First derivative (velocity)
                        vel = np.array(splev(t_eval, tck, der=1)).T
                        vel = vel / (T - 1)  # Scale by time step
                        
                        # Second derivative (acceleration)
                        acc = np.array(splev(t_eval, tck, der=2)).T
                        acc = acc / ((T - 1) ** 2)
                        
                        smoothed_velocities.append(vel)
                        smoothed_accelerations.append(acc)
                    
                    smoothed_positions.append(pos)
                    
                except Exception as e:
                    # If spline fitting fails, use original trajectory
                    smoothed_positions.append(traj_b)
                    if return_derivatives:
                        # Compute numerical derivatives
                        vel = np.gradient(traj_b, axis=0)
                        acc = np.gradient(vel, axis=0)
                        smoothed_velocities.append(vel)
                        smoothed_accelerations.append(acc)
            
            # Convert back to torch
            result = {
                'positions': torch.tensor(
                    np.stack(smoothed_positions, axis=0),
                    device=device, dtype=dtype
                )
            }
            
            if return_derivatives:
                result['velocities'] = torch.tensor(
                    np.stack(smoothed_velocities, axis=0),
                    device=device, dtype=dtype
                )
                result['accelerations'] = torch.tensor(
                    np.stack(smoothed_accelerations, axis=0),
                    device=device, dtype=dtype
                )
            
            return result
            
        except ImportError:
            # Fallback to simple moving average if scipy not available
            return self._smooth_moving_average(trajectory, return_derivatives)
    
    def _smooth_moving_average(
        self,
        trajectory: torch.Tensor,
        return_derivatives: bool = True,
        window_size: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Simple moving average smoothing fallback.
        """
        B, T, D = trajectory.shape
        
        # Pad and apply moving average
        pad = window_size // 2
        padded = torch.nn.functional.pad(
            trajectory.permute(0, 2, 1),  # [B, D, T]
            (pad, pad),
            mode='replicate'
        )
        
        # Moving average via conv1d
        kernel = torch.ones(1, 1, window_size, device=trajectory.device) / window_size
        
        smoothed = []
        for d in range(D):
            smoothed_d = torch.nn.functional.conv1d(
                padded[:, d:d+1, :],
                kernel,
                padding=0
            )
            smoothed.append(smoothed_d)
        
        positions = torch.cat(smoothed, dim=1).permute(0, 2, 1)  # [B, T, D]
        
        result = {'positions': positions}
        
        if return_derivatives:
            # Numerical derivatives
            velocities = torch.gradient(positions, dim=1)[0]
            accelerations = torch.gradient(velocities, dim=1)[0]
            result['velocities'] = velocities
            result['accelerations'] = accelerations
        
        return result


class TrajectoryGenerator:
    """
    Complete trajectory generation pipeline for FlowMP.
    
    Generates trajectories by:
    1. Sampling initial noise from N(0, I)
    2. Solving the ODE dx/dt = v_θ(x, t, c) from t=0 to t=1
    3. Optionally smoothing with B-splines for physical consistency
    
    Usage:
        generator = TrajectoryGenerator(model, config)
        trajectories = generator.generate(
            start_pos, goal_pos, start_vel
        )
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: GeneratorConfig = None,
    ):
        """
        Initialize trajectory generator.
        
        Args:
            model: Trained FlowMP transformer model
            config: Generator configuration
        """
        self.model = model
        self.config = config or GeneratorConfig()
        
        # Determine time schedule
        if self.config.custom_time_schedule is not None:
            time_schedule = self.config.custom_time_schedule
        elif self.config.use_8step_schedule:
            time_schedule = DEFAULT_8STEP_SCHEDULE
        else:
            time_schedule = None  # Use uniform stepping
        
        # Create ODE solver with time schedule
        solver_config = SolverConfig(
            num_steps=self.config.num_steps,
            time_schedule=self.config.time_schedule,
            return_trajectory=False,
            time_schedule=time_schedule,
            use_8step_schedule=self.config.use_8step_schedule,
        )
        self.solver = create_solver(self.config.solver_type, solver_config)
        self.time_schedule = time_schedule
        
        # Create B-spline smoother for physical consistency
        # As per spec: "output smoothing via B-spline to eliminate numerical drift"
        if self.config.use_bspline_smoothing:
            self.smoother = BSplineSmoother(
                degree=self.config.bspline_degree,
                num_control_points=self.config.bspline_num_control_points,
            )
        else:
            self.smoother = None
    
    def _create_velocity_fn(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        start_vel: Optional[torch.Tensor] = None,
    ):
        """
        Create a velocity function for the ODE solver.
        
        The velocity function wraps the model and handles conditioning.
        """
        def velocity_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """
            Compute velocity at state x_t and time t.
            
            Args:
                x_t: Current state [B, T, 6] (pos, vel, acc)
                t: Current time [B]
                
            Returns:
                Velocity field [B, T, 6]
            """
            with torch.no_grad():
                output = self.model(
                    x_t=x_t,
                    t=t,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    start_vel=start_vel,
                )
            return output
        
        return velocity_fn
    
    @torch.no_grad()
    def generate(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        start_vel: Optional[torch.Tensor] = None,
        num_samples: int = None,
        return_raw: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate trajectories for given conditions.
        
        Args:
            start_pos: Starting positions [B, D]
            goal_pos: Goal positions [B, D]
            start_vel: Starting velocities [B, D] (optional)
            num_samples: Number of samples per condition
            return_raw: Whether to also return raw (unsmoothed) trajectory
            
        Returns:
            Dictionary containing:
                - 'positions': Generated position trajectories [B, T, D]
                - 'velocities': Generated velocity trajectories [B, T, D]
                - 'accelerations': Generated acceleration trajectories [B, T, D]
                - 'raw_output': Raw model output before smoothing (if return_raw=True)
        """
        self.model.eval()
        
        B = start_pos.shape[0]
        D = self.config.state_dim
        T = self.config.seq_len
        device = start_pos.device
        dtype = start_pos.dtype
        
        num_samples = num_samples or self.config.num_samples
        
        # Handle multiple samples per condition
        if num_samples > 1:
            start_pos = start_pos.repeat(num_samples, 1)
            goal_pos = goal_pos.repeat(num_samples, 1)
            if start_vel is not None:
                start_vel = start_vel.repeat(num_samples, 1)
            B = B * num_samples
        
        # Sample initial noise x_0 ~ N(0, I)
        # State has 6 channels: pos(2) + vel(2) + acc(2)
        x_0 = torch.randn(B, T, D * 3, device=device, dtype=dtype)
        
        # Create velocity function
        velocity_fn = self._create_velocity_fn(start_pos, goal_pos, start_vel)
        
        # Solve ODE
        x_1 = self.solver.solve(velocity_fn, x_0)
        
        # Extract components
        positions_raw = x_1[..., :D]
        velocities_raw = x_1[..., D:D*2]
        accelerations_raw = x_1[..., D*2:D*3]
        
        result = {}
        
        # Apply smoothing if enabled
        if self.smoother is not None:
            smoothed = self.smoother.smooth(positions_raw, return_derivatives=True)
            result['positions'] = smoothed['positions']
            result['velocities'] = smoothed['velocities']
            result['accelerations'] = smoothed['accelerations']
        else:
            result['positions'] = positions_raw
            result['velocities'] = velocities_raw
            result['accelerations'] = accelerations_raw
        
        if return_raw:
            result['raw_positions'] = positions_raw
            result['raw_velocities'] = velocities_raw
            result['raw_accelerations'] = accelerations_raw
            result['raw_output'] = x_1
        
        return result
    
    @torch.no_grad()
    def generate_with_guidance(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        start_vel: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        obstacle_fn: Optional[callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate trajectories with classifier-free guidance.
        
        Allows steering the generation towards desired properties
        like obstacle avoidance.
        
        Args:
            start_pos: Starting positions [B, D]
            goal_pos: Goal positions [B, D]
            start_vel: Starting velocities [B, D]
            guidance_scale: Scale for guidance (1.0 = no guidance)
            obstacle_fn: Function that returns gradient for obstacle avoidance
            
        Returns:
            Generated trajectories
        """
        # Note: Full CFG requires model trained with condition dropout
        # This is a simplified version with optional obstacle guidance
        
        self.model.eval()
        
        B = start_pos.shape[0]
        D = self.config.state_dim
        T = self.config.seq_len
        device = start_pos.device
        dtype = start_pos.dtype
        
        x_0 = torch.randn(B, T, D * 3, device=device, dtype=dtype)
        
        def guided_velocity_fn(x_t, t):
            # Conditional velocity
            v_cond = self.model(
                x_t=x_t,
                t=t,
                start_pos=start_pos,
                goal_pos=goal_pos,
                start_vel=start_vel,
            )
            
            # Add obstacle avoidance gradient if provided
            if obstacle_fn is not None and guidance_scale != 1.0:
                x_t_clone = x_t.clone().requires_grad_(True)
                obstacle_cost = obstacle_fn(x_t_clone[..., :D])
                
                if obstacle_cost.requires_grad:
                    grad = torch.autograd.grad(
                        obstacle_cost.sum(),
                        x_t_clone,
                        create_graph=False
                    )[0]
                    
                    # Apply guidance
                    v_cond = v_cond - guidance_scale * grad
            
            return v_cond
        
        x_1 = self.solver.solve(guided_velocity_fn, x_0)
        
        # Extract and optionally smooth
        positions_raw = x_1[..., :D]
        
        if self.smoother is not None:
            smoothed = self.smoother.smooth(positions_raw, return_derivatives=True)
            return smoothed
        
        return {
            'positions': positions_raw,
            'velocities': x_1[..., D:D*2],
            'accelerations': x_1[..., D*2:D*3],
        }
    
    @torch.no_grad()
    def generate_batch(
        self,
        conditions: List[Dict[str, torch.Tensor]],
        batch_size: int = 32,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate trajectories for multiple conditions in batches.
        
        Args:
            conditions: List of condition dictionaries
            batch_size: Maximum batch size for inference
            
        Returns:
            List of trajectory dictionaries
        """
        results = []
        
        for i in range(0, len(conditions), batch_size):
            batch_conds = conditions[i:i+batch_size]
            
            # Stack conditions
            start_pos = torch.stack([c['start_pos'] for c in batch_conds])
            goal_pos = torch.stack([c['goal_pos'] for c in batch_conds])
            
            start_vel = None
            if 'start_vel' in batch_conds[0]:
                start_vel = torch.stack([c['start_vel'] for c in batch_conds])
            
            # Generate
            batch_result = self.generate(start_pos, goal_pos, start_vel)
            
            # Split results
            for j in range(len(batch_conds)):
                result = {
                    'positions': batch_result['positions'][j],
                    'velocities': batch_result['velocities'][j],
                    'accelerations': batch_result['accelerations'][j],
                }
                results.append(result)
        
        return results


def create_8step_schedule() -> List[float]:
    """
    Create the 8-step non-uniform time schedule as specified in the implementation strategy.
    
    This schedule uses larger steps early (coarse generation) and smaller steps
    near t=1 (fine-grained refinement) to preserve more details in the final phase.
    
    Returns:
        List of time values: [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
    """
    return [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]


def compute_trajectory_metrics(
    generated: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute metrics for generated trajectories.
    
    Args:
        generated: Generated trajectory dictionary
        target: Ground truth trajectory dictionary (optional)
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    positions = generated['positions']
    velocities = generated['velocities']
    accelerations = generated['accelerations']
    
    # Smoothness metrics (lower is better)
    # Jerk: rate of change of acceleration
    if positions.dim() == 3:
        jerk = torch.diff(accelerations, dim=1)
        metrics['avg_jerk'] = jerk.norm(dim=-1).mean().item()
        
        # Curvature variation
        vel_norm = velocities.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        curvature = (velocities[..., 0] * accelerations[..., 1] - 
                    velocities[..., 1] * accelerations[..., 0]) / (vel_norm.squeeze(-1) ** 3)
        metrics['curvature_var'] = curvature.var(dim=1).mean().item()
    
    # If target provided, compute errors
    if target is not None:
        # Position error
        pos_error = (generated['positions'] - target['positions']).norm(dim=-1)
        metrics['pos_mse'] = pos_error.pow(2).mean().item()
        metrics['pos_mae'] = pos_error.mean().item()
        
        # Goal reaching error (final position)
        goal_error = (generated['positions'][:, -1] - target['positions'][:, -1]).norm(dim=-1)
        metrics['goal_error'] = goal_error.mean().item()
        
        # Velocity error
        if 'velocities' in target:
            vel_error = (generated['velocities'] - target['velocities']).norm(dim=-1)
            metrics['vel_mse'] = vel_error.pow(2).mean().item()
    
    return metrics
