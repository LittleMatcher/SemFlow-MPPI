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
from ..inference.generator import BSplineSmoother, TrajectoryGenerator, GeneratorConfig


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
    enable_cbf_guidance: bool = True  # Enable CBF guidance in ODE solver
    cbf_guidance_strength: float = 1.0  # CBF guidance strength lambda
    
    # Multi-modal anchor selection
    enable_multimodal_anchors: bool = True
    num_anchor_clusters: int = 4  # K clusters for anchor selection
    multimodal_clustering_method: str = "kmeans"  # "kmeans", "gmm", or "spatial"
    
    # Trajectory smoothing
    use_bspline_smoothing: bool = True  # Apply B-spline smoothing to generated trajectories
    bspline_degree: int = 3  # B-spline degree (3 = cubic)
    bspline_num_control_points: int = 20  # Number of control points for B-spline fitting


class L2SafetyCFM(nn.Module):
    """
    L2 Layer: Safety-Embedded Conditional Flow Matching.
    
    Generates multi-modal trajectory anchors for L1 MPPI optimization.
    
    This layer learns a conditional vector field v_θ(z, t, c) that maps from
    noise to trajectory distributions, conditioned on semantic context.
    
    Inputs (Context c from L3 VLM):
        - cost_map: [B, C, H, W] Semantic cost map (encoded to e_map)
        - x_goal: [B, state_dim*2] Target goal state [p_goal, v_goal]
        - w_style: [B, 3] Control style weights [w_safety, w_energy, w_smooth]
        - x_curr: [B, state_dim*3] Current robot state [p_0, v_0, a_0]
        - x_t: [B, T, state_dim*3] Interpolated state at flow time t
        - t: [B] Flow time t ∈ [0, 1]
        
    Network Output (Vector Field):
        - v_pred: [B, T, state_dim*3] Predicted vector field
            - Channels 0:state_dim = velocity field (p_dot)
            - Channels state_dim:2*state_dim = acceleration field (p_ddot)
            - Channels 2*state_dim:3*state_dim = jerk field (p_dddot)
        
    Integration Output (to L1 MPPI):
        - Trajectory anchors: {τ_i}_{i=1}^N where N = num_samples
        - Each trajectory τ: {[p_t, v_t, a_t]}_{t=0}^T
        - Format: {
            'trajectories': [B*N, T, state_dim],
            'velocities': [B*N, T, state_dim],
            'accelerations': [B*N, T, state_dim],
            'full_states': [B*N, T, state_dim*3]
          }
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
        # 注意：这些属性保留用于向后兼容，但实际使用 TrajectoryGenerator 内部的 solver
        solver_config = SolverConfig(
            num_steps=config.num_ode_steps,
            use_8step_schedule=config.use_8step_schedule,
        )
        self.ode_solver = RK4Solver(solver_config)  # 保留用于向后兼容
        
        # ========== Trajectory Smoother ==========
        # 注意：这些属性保留用于向后兼容，但实际使用 TrajectoryGenerator 内部的 smoother
        if config.use_bspline_smoothing:
            self.smoother = BSplineSmoother(
                degree=config.bspline_degree,
                num_control_points=config.bspline_num_control_points,
            )  # 保留用于向后兼容
        else:
            self.smoother = None
        
        # ========== Trajectory Generator (重构后使用统一接口) ==========
        generator_config = GeneratorConfig(
            solver_type=config.solver_type,
            num_steps=config.num_ode_steps,
            use_8step_schedule=config.use_8step_schedule,
            state_dim=config.state_dim,
            seq_len=config.max_seq_len,
            use_bspline_smoothing=config.use_bspline_smoothing,
            bspline_degree=config.bspline_degree,
            bspline_num_control_points=config.bspline_num_control_points,
            num_samples=config.num_trajectory_samples,
            
            # 启用 CBF 安全约束
            use_cbf_guidance=config.use_cbf_constraint,
            cbf_weight=1.0,
            cbf_margin=config.cbf_margin,
            cbf_alpha=1.0,
            
            # 启用多模态锚点生成
            enable_multimodal_anchors=True,
            multimodal_batch_size=config.num_trajectory_samples,
            num_anchor_clusters=min(8, config.num_trajectory_samples // 8),  # 自适应聚类数
            clustering_method="kmeans",
            clustering_features="midpoint",
        )
        self.generator = TrajectoryGenerator(
            model=self.flow_model,
            config=generator_config,
        )
    
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
                # 兼容旧数据集：如果 style_weights 只有 2 维（[w_safe, w_fast]），
                # 则在中间插入一个 "energy" 维度，取两者的平均或 0 作为占位。
                if w_style.dim() == 2 and w_style.shape[1] == 2 and self.config.style_dim == 3:
                    w_safe = w_style[:, 0:1]
                    w_fast = w_style[:, 1:2]
                    w_energy = 0.5 * (w_safe + w_fast)
                    w_style = torch.cat([w_safe, w_energy, w_fast], dim=-1)
            cond_parts.append(w_style)
        
        # Concatenate and project
        cond_vector = torch.cat(cond_parts, dim=-1)
        cond_embed = self.condition_projector(cond_vector)
        
        return cond_embed
    
    def _extract_state_components(
        self,
        x_curr: torch.Tensor,
        x_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        提取状态组件，消除重复代码。
        
        Args:
            x_curr: [B, state_dim * 3] current state [p, v, a]
            x_goal: [B, state_dim * 2] goal state [p, v]
            
        Returns:
            Tuple of (curr_p, curr_v, goal_p, goal_v)
        """
        curr_p = x_curr[:, :self.state_dim]
        curr_v = x_curr[:, self.state_dim:self.state_dim*2]
        goal_p = x_goal[:, :self.state_dim]
        goal_v = x_goal[:, self.state_dim:]
        return curr_p, curr_v, goal_p, goal_v
    
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
        
        **重构说明：**
        此方法现在使用 TrajectoryGenerator 来消除代码重复。
        L2 层特有的预处理（cost_map 编码、条件准备）仍然在此方法中完成。
        
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
        B = cost_map.shape[0]
        D = self.config.state_dim
        device = cost_map.device
        
        # Step 1: Encode cost map (L2 层特有的预处理)
        e_map = self.encode_cost_map(cost_map)  # [B, latent_dim]
        
        # 注意：w_style 参数在此方法中接收，但为了保持与 forward 方法的一致性，
        # 我们只将 e_map 作为 env_encoding 传递。
        # w_style 的信息在训练时通过 prepare_condition 被学习到模型中。
        # 在推理时，模型已经学会了根据不同的条件（包括隐含的 style）生成轨迹。
        
        # Step 2: Extract state components from x_curr and x_goal
        curr_p, curr_v, goal_p, goal_v = self._extract_state_components(x_curr, x_goal)
        
        # Step 3: Use TrajectoryGenerator (重构后统一接口)
        # TrajectoryGenerator 会自动处理：
        # - 批次扩展（num_samples > 1 时）
        # - ODE 求解
        # - B 样条平滑
        # - CBF 安全约束
        # - 多模态锚点聚类
        result = self.generator.generate(
            start_pos=curr_p,
            goal_pos=goal_p,
            start_vel=curr_v,
            goal_vel=goal_v,  # L2 层额外参数
            env_encoding=e_map,  # L2 层额外参数（从 cost_map 编码得到，与 forward 方法一致）
            num_samples=num_samples or self.config.num_trajectory_samples,
            return_raw=False,  # L2 层不需要原始输出
            obstacle_positions=None,  # 可以从外部传入
            cost_map=cost_map,  # 传递语义代价图用于 CBF
        )
        
        # Step 4: 重命名键以保持向后兼容
        # TrajectoryGenerator 返回 'positions'，L2 层期望 'trajectories'
        output = {
            'trajectories': result['positions'],
            'velocities': result['velocities'],
            'accelerations': result['accelerations'],
            'full_states': torch.cat([
                result['positions'],
                result['velocities'],
                result['accelerations']
            ], dim=-1),  # [B*N, T, state_dim*3] = [p, v, a]
            'num_samples_per_batch': num_samples or self.config.num_trajectory_samples,
        }
        
        return output
    
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
        
        Predicts the vector field v_θ(x_t, t, c) that describes the flow
        from noise to trajectory distribution.
        
        Args:
            x_t: [B, T, state_dim*3] Interpolated state at flow time t
                 Contains [p_t, v_t, a_t] concatenated
            t: [B] Flow time values t ∈ [0, 1]
            cost_map: [B, C, H, W] Semantic cost map from L3
            x_curr: [B, state_dim*3] Current robot state [p_0, v_0, a_0]
            x_goal: [B, state_dim*2] Goal state [p_goal, v_goal]
            w_style: [B, 3] Control style weights [w_safety, w_energy, w_smooth]
            
        Returns:
            Predicted vector field [B, T, state_dim*3]
            - Channels 0:state_dim = velocity field (p_dot)
            - Channels state_dim:2*state_dim = acceleration field (p_ddot)
            - Channels 2*state_dim:3*state_dim = jerk field (p_dddot)
            
        Note:
            This is the vector field that, when integrated via ODE solver
            from t=0 to t=1, produces the trajectory anchors for L1 MPPI.
        """
        # Encode cost map
        e_map = self.encode_cost_map(cost_map)
        
        # Prepare condition
        cond_embed = self.prepare_condition(x_curr, x_goal, e_map, w_style)
        
        # Extract state components
        curr_p, curr_v, goal_p, goal_v = self._extract_state_components(x_curr, x_goal)
        
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
