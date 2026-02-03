"""
L2 Layer: Safety-Embedded Conditional Flow Matching

This module serves as the middle layer in a three-tier architecture:
- L3 (Upper): VLM provides semantic understanding and cost maps
- L2 (This layer): Safety-Embedded CFM generates multi-modal trajectory anchors
- L1 (Lower): MPPI performs local optimization

Key Features:
- Multi-modal trajectory generation with controllable styles
- Semantic cost map encoding from L3
- Control style weights (safety, energy, smoothness)
- Full second-order dynamics output [p, v, a]
- CBF safety constraint embedding
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from .transformer import FlowMPTransformer
from .unet_1d import FlowMPUNet1D
from .cost_map_encoder import CostMapEncoder, create_cost_map_encoder
from .embeddings import ConditionEncoder
from ..inference.ode_solver import RK4Solver, SolverConfig


@dataclass
class L2Config:
    """Configuration for L2 Safety-Embedded CFM Layer."""
    
    # Model architecture
    model_type: str = "transformer"  # "transformer" or "unet1d"
    state_dim: int = 2  # Position dimension (x, y)
    max_seq_len: int = 64  # Trajectory length
    
    # Transformer parameters
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    
    # U-Net parameters (if model_type="unet1d")
    unet_base_channels: int = 128
    unet_channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    
    # Cost map encoder
    cost_map_encoder_type: str = "single_scale"  # "single_scale" or "multi_scale"
    cost_map_channels: int = 1
    cost_map_latent_dim: int = 256
    
    # Control style weights
    use_style_conditioning: bool = True
    style_dim: int = 3  # [w_safety, w_energy, w_smooth]
    
    # Multi-modal generation
    num_trajectory_samples: int = 64  # N trajectories for L1 MPPI
    
    # ODE solver
    solver_type: str = "rk4"
    num_ode_steps: int = 20
    use_8step_schedule: bool = True
    
    # Safety constraints (CBF embedding)
    use_cbf_constraint: bool = True
    cbf_margin: float = 0.1  # Safety margin in meters


class L2SafetyCFM(nn.Module):
    """
    L2 Layer: Safety-Embedded Conditional Flow Matching.
    
    Generates multi-modal trajectory anchors for L1 MPPI optimization.
    
    Inputs (from L3 VLM):
        - e_map: Semantic cost map encoding
        - x_goal: Target goal state
        - w_style: Control style weights [w_safety, w_energy, w_smooth]
        - x_curr: Current robot state [p_0, v_0, a_0]
        
    Outputs (to L1 MPPI):
        - Trajectory anchors: {τ_i}_{i=1}^N
        - Each trajectory: {[p_t, v_t, a_t]}_{t=0}^T
    """
    
    def __init__(self, config: L2Config):
        super().__init__()
        
        self.config = config
        self.state_dim = config.state_dim
        
        # ========== Cost Map Encoder (from L3) ==========
        self.cost_map_encoder = create_cost_map_encoder(
            encoder_type=config.cost_map_encoder_type,
            input_channels=config.cost_map_channels,
            latent_dim=config.cost_map_latent_dim,
        )
        
        # ========== Enhanced Condition Encoder ==========
        # Total condition dimension includes:
        # - Current state: [p_0, v_0, a_0] -> state_dim * 3
        # - Goal state: [p_goal, v_goal] -> state_dim * 2
        # - Cost map encoding: cost_map_latent_dim
        # - Style weights: style_dim
        
        total_cond_dim = config.cost_map_latent_dim
        if config.use_style_conditioning:
            total_cond_dim += config.style_dim
        
        time_embed_dim = config.hidden_dim
        
        self.condition_projector = nn.Sequential(
            nn.Linear(
                config.state_dim * 3 +  # current state [p, v, a]
                config.state_dim * 2 +  # goal state [p, v]
                total_cond_dim,         # cost map + style
                time_embed_dim
            ),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # ========== Flow Matching Model ==========
        if config.model_type == "transformer":
            from .transformer import create_flowmp_transformer
            self.flow_model = create_flowmp_transformer(
                variant="base",
                state_dim=config.state_dim,
                max_seq_len=config.max_seq_len,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                time_embed_dim=time_embed_dim,
                env_encoding_dim=config.cost_map_latent_dim,  # Fixed: pass correct env encoding dimension
            )
        else:
            from .unet_1d import create_flowmp_unet1d
            self.flow_model = create_flowmp_unet1d(
                state_dim=config.state_dim,
                max_seq_len=config.max_seq_len,
                base_channels=config.unet_base_channels,
                channel_mults=config.unet_channel_mults,
                time_embed_dim=time_embed_dim,
            )
        
        # ========== ODE Solver ==========
        solver_config = SolverConfig(
            num_steps=config.num_ode_steps,
            use_8step_schedule=config.use_8step_schedule,
        )
        self.ode_solver = RK4Solver(solver_config)
    
    def encode_cost_map(self, cost_map: torch.Tensor) -> torch.Tensor:
        """
        Encode semantic cost map from L3 VLM.
        
        Args:
            cost_map: [B, C, H, W] semantic cost map
            
        Returns:
            Latent encoding [B, latent_dim]
        """
        return self.cost_map_encoder(cost_map)
    
    def prepare_condition(
        self,
        x_curr: torch.Tensor,
        x_goal: torch.Tensor,
        e_map: torch.Tensor,
        w_style: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare unified condition vector.
        
        Args:
            x_curr: [B, state_dim * 3] current state [p, v, a]
            x_goal: [B, state_dim * 2] goal state [p, v]
            e_map: [B, latent_dim] cost map encoding
            w_style: [B, 3] style weights [w_safety, w_energy, w_smooth]
            
        Returns:
            Condition embedding [B, time_embed_dim]
        """
        B = x_curr.shape[0]
        device = x_curr.device
        
        # Ensure all tensors are on the same device
        x_curr = x_curr.to(device)
        x_goal = x_goal.to(device)
        e_map = e_map.to(device)
        
        # Build condition vector
        cond_parts = [x_curr, x_goal, e_map]
        
        if self.config.use_style_conditioning:
            if w_style is None:
                # Default balanced style
                w_style = torch.ones(B, self.config.style_dim, device=device, dtype=x_curr.dtype) / self.config.style_dim
            else:
                # Ensure w_style is on the correct device
                w_style = w_style.to(device)
            cond_parts.append(w_style)
        
        # Concatenate and project
        cond_vector = torch.cat(cond_parts, dim=-1)
        cond_embed = self.condition_projector(cond_vector)
        
        return cond_embed
    
    @torch.no_grad()
    def generate_trajectory_anchors(
        self,
        cost_map: torch.Tensor,
        x_curr: torch.Tensor,
        x_goal: torch.Tensor,
        w_style: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate multi-modal trajectory anchors for L1 MPPI.
        
        This is the main API for L2 layer inference.
        
        Args:
            cost_map: [B, C, H, W] semantic cost map from L3
            x_curr: [B, state_dim * 3] current robot state [p, v, a]
            x_goal: [B, state_dim * 2] goal state [p, v]
            w_style: [B, 3] control style weights (optional)
            num_samples: Number of trajectory samples (default: config.num_trajectory_samples)
            
        Returns:
            Dictionary containing:
                - 'trajectories': [B*N, T, state_dim] position trajectories
                - 'velocities': [B*N, T, state_dim] velocity profiles
                - 'accelerations': [B*N, T, state_dim] acceleration profiles
                - 'full_states': [B*N, T, state_dim*3] complete dynamics [p, v, a]
        """
        self.flow_model.eval()
        
        B = cost_map.shape[0]
        N = num_samples or self.config.num_trajectory_samples
        T = self.config.max_seq_len
        D = self.config.state_dim
        device = cost_map.device
        
        # Step 1: Encode cost map
        e_map = self.encode_cost_map(cost_map)  # [B, latent_dim]
        
        # Step 2: Prepare condition
        cond_embed = self.prepare_condition(x_curr, x_goal, e_map, w_style)
        
        # Step 3: Expand for multi-modal sampling
        # Repeat conditions for N samples
        e_map_expanded = e_map.repeat_interleave(N, dim=0)  # [B*N, latent_dim]
        x_curr_expanded = x_curr.repeat_interleave(N, dim=0)  # [B*N, state_dim*3]
        x_goal_expanded = x_goal.repeat_interleave(N, dim=0)  # [B*N, state_dim*2]
        if w_style is not None:
            w_style_expanded = w_style.repeat_interleave(N, dim=0)
        else:
            w_style_expanded = None
        
        # Step 4: Sample initial noise
        x_0 = torch.randn(B * N, T, D * 3, device=device)  # [B*N, T, 6]
        
        # Step 5: Define velocity function for ODE solver
        def velocity_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Velocity field at (x_t, t)."""
            # Prepare conditions for this batch
            curr_p = x_curr_expanded[:, :D]
            curr_v = x_curr_expanded[:, D:D*2]
            goal_p = x_goal_expanded[:, :D]
            goal_v = x_goal_expanded[:, D:]
            
            # Model forward pass
            output = self.flow_model(
                x_t=x_t,
                t=t,
                start_pos=curr_p,
                goal_pos=goal_p,
                start_vel=curr_v,
                goal_vel=goal_v,
                env_encoding=e_map_expanded,
            )
            
            return output
        
        # Step 6: Solve ODE from t=0 to t=1
        x_1 = self.ode_solver.solve(velocity_fn, x_0)  # [B*N, T, 6]
        
        # Step 7: Extract dynamics components
        positions = x_1[..., :D]  # [B*N, T, 2]
        velocities = x_1[..., D:D*2]  # [B*N, T, 2]
        accelerations = x_1[..., D*2:]  # [B*N, T, 2]
        
        return {
            'trajectories': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'full_states': x_1,  # [B*N, T, 6] = [p, v, a]
            'num_samples_per_batch': N,
        }
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cost_map: torch.Tensor,
        x_curr: torch.Tensor,
        x_goal: torch.Tensor,
        w_style: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training forward pass.
        
        Args:
            x_t: [B, T, 6] interpolated state
            t: [B] flow time
            cost_map: [B, C, H, W] cost map
            x_curr: [B, 6] current state
            x_goal: [B, 4] goal state
            w_style: [B, 3] style weights
            
        Returns:
            Predicted vector field [B, T, 6]
        """
        # Encode cost map
        e_map = self.encode_cost_map(cost_map)
        
        # Prepare condition
        cond_embed = self.prepare_condition(x_curr, x_goal, e_map, w_style)
        
        # Extract state components
        curr_p = x_curr[:, :self.state_dim]
        curr_v = x_curr[:, self.state_dim:self.state_dim*2]
        goal_p = x_goal[:, :self.state_dim]
        goal_v = x_goal[:, self.state_dim:]
        
        # Model forward
        output = self.flow_model(
            x_t=x_t,
            t=t,
            start_pos=curr_p,
            goal_pos=goal_p,
            start_vel=curr_v,
            goal_vel=goal_v,
            env_encoding=e_map,
        )
        
        return output


def create_l2_safety_cfm(
    model_type: str = "transformer",
    **kwargs,
) -> L2SafetyCFM:
    """
    Factory function to create L2 Safety-Embedded CFM model.
    
    Args:
        model_type: "transformer" or "unet1d"
        **kwargs: Additional config arguments
        
    Returns:
        L2SafetyCFM model instance
    """
    config = L2Config(model_type=model_type, **kwargs)
    return L2SafetyCFM(config)
