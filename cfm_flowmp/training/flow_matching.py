"""
Flow Matching Training Logic

Implements the core flow matching algorithm for trajectory generation:
1. Interpolation path construction (Eq. 6, 8, 10 from FlowMP)
2. Target field computation
3. Flow matching loss

Reference: FlowMP paper equations for conditional probability paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching training."""
    
    # State dimensions
    state_dim: int = 2  # Position dimension (x, y)
    
    # Loss weights
    lambda_vel: float = 1.0      # Weight for velocity field loss
    lambda_acc: float = 1.0      # Weight for acceleration field loss
    lambda_jerk: float = 1.0     # Weight for jerk field loss
    
    # Interpolation parameters
    sigma_min: float = 1e-4      # Minimum noise scale at t=1
    
    # Optional: different sigma for each state component
    sigma_pos: float = 1e-4      # Noise scale for position
    sigma_vel: float = 1e-4      # Noise scale for velocity
    sigma_acc: float = 1e-4      # Noise scale for acceleration
    
    # Time sampling
    t_min: float = 0.0           # Minimum flow time
    t_max: float = 1.0           # Maximum flow time
    

class FlowInterpolator:
    """
    Constructs interpolation paths for flow matching.
    
    Implements the conditional probability path p_t(x|x_1) used in
    Conditional Flow Matching. At t=0, the distribution is N(0, I),
    and at t=1, the distribution concentrates at the target x_1.
    
    The interpolation follows:
        x_t = t * x_1 + (1 - t) * epsilon
        
    where epsilon ~ N(0, I) and x_1 is the target trajectory.
    
    For FlowMP, we have three coupled interpolations:
        q_t = interpolate(q_0, q_1, t)     # position
        q_dot_t = interpolate(q_dot_0, q_dot_1, t)    # velocity
        q_ddot_t = interpolate(q_ddot_0, q_ddot_1, t)  # acceleration
    """
    
    def __init__(self, config: FlowMatchingConfig = None):
        """
        Args:
            config: Flow matching configuration
        """
        self.config = config or FlowMatchingConfig()
    
    def sample_time(
        self, 
        batch_size: int, 
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample flow time t uniformly from [t_min, t_max].
        
        Args:
            batch_size: Number of samples
            device: Target device
            dtype: Target dtype
            
        Returns:
            Time values of shape [batch_size]
        """
        t = torch.rand(batch_size, device=device, dtype=dtype)
        t = self.config.t_min + t * (self.config.t_max - self.config.t_min)
        return t
    
    def sample_noise(
        self,
        target_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample Gaussian noise epsilon ~ N(0, I).
        
        Args:
            target_shape: Shape of noise tensor
            device: Target device
            dtype: Target dtype
            
        Returns:
            Noise tensor of given shape
        """
        return torch.randn(target_shape, device=device, dtype=dtype)
    
    def interpolate_simple(
        self,
        x_0: torch.Tensor,  # Noise
        x_1: torch.Tensor,  # Target
        t: torch.Tensor,    # Time
    ) -> torch.Tensor:
        """
        Simple linear interpolation: x_t = t * x_1 + (1 - t) * x_0
        
        This is the standard OT (Optimal Transport) path used in
        rectified flow / flow matching.
        
        Args:
            x_0: Initial state (noise), shape [B, ...]
            x_1: Target state, shape [B, ...]
            t: Time values, shape [B] or [B, 1, ...]
            
        Returns:
            Interpolated state x_t
        """
        # Expand t for broadcasting
        while t.dim() < x_0.dim():
            t = t.unsqueeze(-1)
        
        return t * x_1 + (1 - t) * x_0
    
    def compute_target_velocity(
        self,
        x_0: torch.Tensor,  # Noise (or x_t)
        x_1: torch.Tensor,  # Target
        t: torch.Tensor,    # Time
        from_interpolated: bool = True,
    ) -> torch.Tensor:
        """
        Compute the target velocity field.
        
        For the OT path x_t = t * x_1 + (1-t) * x_0:
            dx_t/dt = x_1 - x_0
        
        Or equivalently, expressed in terms of x_t:
            v_target = (x_1 - x_t) / (1 - t)
        
        Args:
            x_0: Initial state (noise) or interpolated state
            x_1: Target state
            t: Time values
            from_interpolated: If True, x_0 is actually x_t (interpolated)
            
        Returns:
            Target velocity field
        """
        # Expand t for broadcasting
        while t.dim() < x_1.dim():
            t = t.unsqueeze(-1)
        
        if from_interpolated:
            # x_0 is actually x_t
            x_t = x_0
            # v = (x_1 - x_t) / (1 - t)
            # Add small epsilon to avoid division by zero at t=1
            eps = 1e-6
            v_target = (x_1 - x_t) / (1 - t + eps)
        else:
            # Simple form: v = x_1 - x_0
            v_target = x_1 - x_0
        
        return v_target
    
    def interpolate_trajectory(
        self,
        q_1: torch.Tensor,       # Target position [B, T, 2]
        q_dot_1: torch.Tensor,   # Target velocity [B, T, 2]
        q_ddot_1: torch.Tensor,  # Target acceleration [B, T, 2]
        t: torch.Tensor,         # Flow time [B]
        epsilon_q: torch.Tensor = None,      # Position noise
        epsilon_q_dot: torch.Tensor = None,  # Velocity noise
        epsilon_q_ddot: torch.Tensor = None, # Acceleration noise
    ) -> Dict[str, torch.Tensor]:
        """
        Construct interpolated trajectory state at flow time t.
        
        Implements the FlowMP interpolation for position, velocity,
        and acceleration simultaneously (Eq. 6, 8, 10).
        
        The interpolation follows the optimal transport path:
            x_t = t * x_1 + (1 - t) * epsilon
        
        And the target velocity field is:
            u_target = (x_1 - x_t) / (1 - t)
        
        Note: For t close to 1, we use numerical stabilization.
        
        Args:
            q_1: Target position trajectory [B, T, state_dim]
            q_dot_1: Target velocity trajectory [B, T, state_dim]
            q_ddot_1: Target acceleration trajectory [B, T, state_dim]
            t: Flow time values [B]
            epsilon_*: Optional pre-sampled noise tensors
            
        Returns:
            Dictionary containing:
                - 'q_t': Interpolated position [B, T, state_dim]
                - 'q_dot_t': Interpolated velocity [B, T, state_dim]
                - 'q_ddot_t': Interpolated acceleration [B, T, state_dim]
                - 'u_target': Position velocity field target [B, T, state_dim]
                - 'v_target': Velocity acceleration field target [B, T, state_dim]
                - 'w_target': Acceleration jerk field target [B, T, state_dim]
                - 'x_t': Concatenated state [B, T, state_dim * 3]
                - 'target': Concatenated target field [B, T, state_dim * 3]
        """
        B, T, D = q_1.shape
        device = q_1.device
        dtype = q_1.dtype
        
        # Sample noise if not provided
        if epsilon_q is None:
            epsilon_q = self.sample_noise((B, T, D), device, dtype)
        if epsilon_q_dot is None:
            epsilon_q_dot = self.sample_noise((B, T, D), device, dtype)
        if epsilon_q_ddot is None:
            epsilon_q_ddot = self.sample_noise((B, T, D), device, dtype)
        
        # Expand t for broadcasting: [B] -> [B, 1, 1]
        t_expanded = t[:, None, None]
        
        # ============ Interpolate States (Eq. 6, 8, 10) ============
        # q_t = t * q_1 + (1 - t) * epsilon_q
        q_t = t_expanded * q_1 + (1 - t_expanded) * epsilon_q
        
        # q_dot_t = t * q_dot_1 + (1 - t) * epsilon_q_dot
        q_dot_t = t_expanded * q_dot_1 + (1 - t_expanded) * epsilon_q_dot
        
        # q_ddot_t = t * q_ddot_1 + (1 - t) * epsilon_q_ddot
        q_ddot_t = t_expanded * q_ddot_1 + (1 - t_expanded) * epsilon_q_ddot
        
        # ============ Compute Target Fields ============
<<<<<<< Current (Your changes)
        # According to the implementation strategy, we use the form:
        # u_target = (q_1 - q_t) / (1 - t)
        # v_target = (q_dot_1 - q_dot_t) / (1 - t)
        # w_target = (q_ddot_1 - q_ddot_t) / (1 - t)
        # This form is more numerically stable and matches the ODE formulation
        
        # Add small epsilon to avoid division by zero at t=1
        eps = 1e-6
        u_target = (q_1 - q_t) / (1 - t_expanded + eps)
        v_target = (q_dot_1 - q_dot_t) / (1 - t_expanded + eps)
        w_target = (q_ddot_1 - q_ddot_t) / (1 - t_expanded + eps)
=======
        # According to FlowMP Algorithm 1:
        # u_target = (q_1 - q_t) / (1 - t)
        # v_target = (q_dot_1 - q_dot_t) / (1 - t)
        # w_target = (q_ddot_1 - q_ddot_t) / (1 - t)
        #
        # Note: Since q_t = t * q_1 + (1-t) * epsilon,
        #       (q_1 - q_t) / (1-t) = (q_1 - t*q_1 - (1-t)*epsilon) / (1-t)
        #                          = ((1-t)*q_1 - (1-t)*epsilon) / (1-t)
        #                          = q_1 - epsilon
        # Both forms are mathematically equivalent, but we use the explicit
        # form (x_1 - x_t) / (1-t) for numerical consistency with the paper.
        
        # Small epsilon to avoid division by zero when t is close to 1
        eps = 1e-5
        one_minus_t = (1 - t_expanded).clamp(min=eps)
        
        u_target = (q_1 - q_t) / one_minus_t
        v_target = (q_dot_1 - q_dot_t) / one_minus_t
        w_target = (q_ddot_1 - q_ddot_t) / one_minus_t
>>>>>>> Incoming (Background Agent changes)
        
        # ============ Concatenate for Network Input/Output ============
        # Input state: [pos, vel, acc] -> [B, T, 6]
        x_t = torch.cat([q_t, q_dot_t, q_ddot_t], dim=-1)
        
        # Target field: [u, v, w] -> [B, T, 6]
        target = torch.cat([u_target, v_target, w_target], dim=-1)
        
        return {
            'q_t': q_t,
            'q_dot_t': q_dot_t,
            'q_ddot_t': q_ddot_t,
            'u_target': u_target,
            'v_target': v_target,
            'w_target': w_target,
            'x_t': x_t,
            'target': target,
            'epsilon_q': epsilon_q,
            'epsilon_q_dot': epsilon_q_dot,
            'epsilon_q_ddot': epsilon_q_ddot,
            't': t,
        }


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching Loss for FlowMP.
    
    Computes the weighted MSE loss between predicted and target vector fields:
    
    L = ||u_pred - u_target||^2 
        + λ_acc * ||v_pred - v_target||^2 
        + λ_jerk * ||w_pred - w_target||^2
    
    where:
        - u: velocity field (for position)
        - v: acceleration field (for velocity)
        - w: jerk field (for acceleration)
    """
    
    def __init__(self, config: FlowMatchingConfig = None):
        """
        Args:
            config: Flow matching configuration with loss weights
        """
        super().__init__()
        self.config = config or FlowMatchingConfig()
        self.interpolator = FlowInterpolator(config)
    
    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean',
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss.
        
        Args:
            model_output: Predicted vector field [B, T, 6]
            target: Target vector field [B, T, 6]
            reduction: Loss reduction method ('mean', 'sum', 'none')
            
        Returns:
            Dictionary containing:
                - 'loss': Total weighted loss
                - 'loss_vel': Velocity field loss (u)
                - 'loss_acc': Acceleration field loss (v)
                - 'loss_jerk': Jerk field loss (w)
        """
        D = self.config.state_dim
        
        # Extract field components
        u_pred = model_output[..., :D]
        v_pred = model_output[..., D:D*2]
        w_pred = model_output[..., D*2:D*3]
        
        u_target = target[..., :D]
        v_target = target[..., D:D*2]
        w_target = target[..., D*2:D*3]
        
        # Compute MSE for each field
        loss_vel = F.mse_loss(u_pred, u_target, reduction=reduction)
        loss_acc = F.mse_loss(v_pred, v_target, reduction=reduction)
        loss_jerk = F.mse_loss(w_pred, w_target, reduction=reduction)
        
        # Weighted sum
        total_loss = (
            self.config.lambda_vel * loss_vel +
            self.config.lambda_acc * loss_acc +
            self.config.lambda_jerk * loss_jerk
        )
        
        return {
            'loss': total_loss,
            'loss_vel': loss_vel,
            'loss_acc': loss_acc,
            'loss_jerk': loss_jerk,
        }
    
    def compute_training_loss(
        self,
        model: nn.Module,
        q_1: torch.Tensor,
        q_dot_1: torch.Tensor,
        q_ddot_1: torch.Tensor,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        start_vel: torch.Tensor = None,
        t: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Complete training loss computation including interpolation.
        
        This method handles the full training pipeline:
        1. Sample flow time t
        2. Sample noise
        3. Construct interpolated states
        4. Compute target fields
        5. Run model forward pass
        6. Compute loss
        
        Args:
            model: FlowMP transformer model
            q_1: Target position trajectory [B, T, D]
            q_dot_1: Target velocity trajectory [B, T, D]
            q_ddot_1: Target acceleration trajectory [B, T, D]
            start_pos: Starting position [B, D]
            goal_pos: Goal position [B, D]
            start_vel: Starting velocity [B, D] (optional)
            t: Pre-sampled time (optional, will sample if None)
            
        Returns:
            Loss dictionary
        """
        B = q_1.shape[0]
        device = q_1.device
        dtype = q_1.dtype
        
        # Sample flow time if not provided
        if t is None:
            t = self.interpolator.sample_time(B, device, dtype)
        
        # Construct interpolated trajectory and targets
        interp_result = self.interpolator.interpolate_trajectory(
            q_1=q_1,
            q_dot_1=q_dot_1,
            q_ddot_1=q_ddot_1,
            t=t,
        )
        
        x_t = interp_result['x_t']
        target = interp_result['target']
        
        # Model forward pass
        model_output = model(
            x_t=x_t,
            t=t,
            start_pos=start_pos,
            goal_pos=goal_pos,
            start_vel=start_vel,
        )
        
        # Compute loss
        loss_dict = self.forward(model_output, target)
        
        # Add interpolation info for debugging
        loss_dict['t'] = t.mean()
        
        return loss_dict


class VelocityConsistencyLoss(nn.Module):
    """
    Optional auxiliary loss for physical consistency.
    
    Encourages the velocity field to be consistent with
    the time derivative of the position field.
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        q_pred: torch.Tensor,
        q_dot_pred: torch.Tensor,
        dt: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute velocity consistency loss.
        
        Args:
            q_pred: Predicted positions [B, T, D]
            q_dot_pred: Predicted velocities [B, T, D]
            dt: Time step between trajectory points
            
        Returns:
            Consistency loss scalar
        """
        # Finite difference velocity: dq/dt ≈ (q[t+1] - q[t]) / dt
        q_diff = (q_pred[:, 1:, :] - q_pred[:, :-1, :]) / dt
        
        # Compare with predicted velocity (excluding last step)
        q_dot_mid = (q_dot_pred[:, 1:, :] + q_dot_pred[:, :-1, :]) / 2
        
        loss = F.mse_loss(q_diff, q_dot_mid)
        return self.weight * loss
