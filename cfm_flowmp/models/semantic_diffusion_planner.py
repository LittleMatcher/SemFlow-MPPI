"""
Semantic-Conditioned Diffusion Planner for L2.

This module implements a practical L2 design aligned with generative predictive control:
- Inputs: robot state, local geometry map, semantic cost map, and goal conditions.
- Backbone: 1D U-Net diffusion denoiser over trajectory tokens.
- Outputs: multi-modal trajectory proposals with scores/weights/tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cost_map_encoder import create_cost_map_encoder
from .unet_1d import create_flowmp_unet1d


@dataclass
class SemanticDiffusionConfig:
    # Trajectory/action tokens
    horizon: int = 64
    action_dim: int = 2  # waypoint or action dimension per step

    # Robot state
    robot_state_dim: int = 6  # [px, py, vx, vy, ax, ay]

    # Goal conditioning
    goal_embed_dim: int = 128
    goal_room_vocab_size: int = 64
    goal_image_feat_dim: int = 256
    goal_text_feat_dim: int = 256

    # Map encoders
    map_channels: int = 1
    map_latent_dim: int = 256
    map_encoder_type: str = "single_scale"

    # Diffusion model
    hidden_dim: int = 256
    unet_base_channels: int = 128
    unet_channel_mults: tuple = (1, 2, 4, 8)
    diffusion_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class GoalConditionEncoder(nn.Module):
    """Encode heterogeneous goal conditions into one latent embedding."""

    def __init__(self, config: SemanticDiffusionConfig):
        super().__init__()
        self.config = config

        self.room_embedding = nn.Embedding(config.goal_room_vocab_size, config.goal_embed_dim)

        self.goal_state_proj = nn.Sequential(
            nn.Linear(2, config.goal_embed_dim),
            nn.SiLU(),
            nn.Linear(config.goal_embed_dim, config.goal_embed_dim),
        )

        self.goal_image_proj = nn.Sequential(
            nn.Linear(config.goal_image_feat_dim, config.goal_embed_dim),
            nn.SiLU(),
            nn.Linear(config.goal_embed_dim, config.goal_embed_dim),
        )

        self.goal_text_proj = nn.Sequential(
            nn.Linear(config.goal_text_feat_dim, config.goal_embed_dim),
            nn.SiLU(),
            nn.Linear(config.goal_embed_dim, config.goal_embed_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(config.goal_embed_dim * 4, config.goal_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(config.goal_embed_dim * 2, config.goal_embed_dim),
        )

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        goal_state: Optional[torch.Tensor] = None,
        goal_room_id: Optional[torch.Tensor] = None,
        goal_image_feat: Optional[torch.Tensor] = None,
        goal_text_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        zeros = torch.zeros(batch_size, self.config.goal_embed_dim, device=device, dtype=dtype)

        if goal_state is not None:
            gs = self.goal_state_proj(goal_state)
        else:
            gs = zeros

        if goal_room_id is not None:
            room = self.room_embedding(goal_room_id.long())
            room = room.to(dtype=dtype)
        else:
            room = zeros

        if goal_image_feat is not None:
            img = self.goal_image_proj(goal_image_feat)
        else:
            img = zeros

        if goal_text_feat is not None:
            txt = self.goal_text_proj(goal_text_feat)
        else:
            txt = zeros

        fused = torch.cat([gs, room, img, txt], dim=-1)
        return self.fusion(fused)


class SemanticConditionedDiffusionPlanner(nn.Module):
    """
    L2 semantic-conditioned diffusion trajectory planner.

    Training:
      - Minimize epsilon-prediction objective over noisy trajectory tokens.

    Inference:
      - Sample K trajectories from conditional diffusion model.
      - Return proposal_scores / mode_weights / semantic_tags for L1.
    """

    def __init__(self, config: SemanticDiffusionConfig):
        super().__init__()
        self.config = config

        self.local_map_encoder = create_cost_map_encoder(
            encoder_type=config.map_encoder_type,
            input_channels=config.map_channels,
            latent_dim=config.map_latent_dim,
        )
        self.semantic_map_encoder = create_cost_map_encoder(
            encoder_type=config.map_encoder_type,
            input_channels=config.map_channels,
            latent_dim=config.map_latent_dim,
        )
        self.goal_encoder = GoalConditionEncoder(config)

        self.robot_state_proj = nn.Sequential(
            nn.Linear(config.robot_state_dim, config.goal_embed_dim),
            nn.SiLU(),
            nn.Linear(config.goal_embed_dim, config.goal_embed_dim),
        )

        self.cond_fusion = nn.Sequential(
            nn.Linear(config.map_latent_dim * 2 + config.goal_embed_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        self.denoiser = create_flowmp_unet1d(
            state_dim=config.action_dim,
            input_channels=config.action_dim,
            output_channels=config.action_dim,
            max_seq_len=config.horizon,
            base_channels=config.unet_base_channels,
            channel_mults=config.unet_channel_mults,
            time_embed_dim=config.hidden_dim,
            env_encoding_dim=config.hidden_dim,
        )

        betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def _prepare_condition(
        self,
        robot_state: torch.Tensor,
        local_map: torch.Tensor,
        semantic_cost_map: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        goal_room_id: Optional[torch.Tensor] = None,
        goal_image_feat: Optional[torch.Tensor] = None,
        goal_text_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz = robot_state.shape[0]
        device = robot_state.device
        dtype = robot_state.dtype

        local_latent = self.local_map_encoder(local_map)
        semantic_latent = self.semantic_map_encoder(semantic_cost_map)
        robot_latent = self.robot_state_proj(robot_state)
        goal_latent = self.goal_encoder(
            batch_size=bsz,
            device=device,
            dtype=dtype,
            goal_state=goal_state,
            goal_room_id=goal_room_id,
            goal_image_feat=goal_image_feat,
            goal_text_feat=goal_text_feat,
        )

        fused = self.cond_fusion(torch.cat([local_latent, semantic_latent, robot_latent, goal_latent], dim=-1))

        start_pos = robot_state[:, : self.config.action_dim]
        if goal_state is not None:
            goal_pos = goal_state[:, : self.config.action_dim]
        else:
            goal_pos = torch.zeros_like(start_pos)

        start_vel = robot_state[:, self.config.action_dim : self.config.action_dim * 2]
        if start_vel.shape[-1] != self.config.action_dim:
            start_vel = torch.zeros_like(start_pos)

        return {
            "start_pos": start_pos,
            "goal_pos": goal_pos,
            "start_vel": start_vel,
            "env_encoding": fused,
        }

    def _extract(self, vec: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = vec.gather(0, t.long())
        while out.dim() < x.dim():
            out = out.unsqueeze(-1)
        return out

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = torch.sqrt(self._extract(self.alpha_bars, t, x0))
        sqrt_one_minus_ab = torch.sqrt(1.0 - self._extract(self.alpha_bars, t, x0))
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise

    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        robot_state: torch.Tensor,
        local_map: torch.Tensor,
        semantic_cost_map: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        goal_room_id: Optional[torch.Tensor] = None,
        goal_image_feat: Optional[torch.Tensor] = None,
        goal_text_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self._prepare_condition(
            robot_state=robot_state,
            local_map=local_map,
            semantic_cost_map=semantic_cost_map,
            goal_state=goal_state,
            goal_room_id=goal_room_id,
            goal_image_feat=goal_image_feat,
            goal_text_feat=goal_text_feat,
        )

        return self.denoiser(
            x_t=x_t,
            t=t.float() / max(1, self.config.diffusion_steps - 1),
            start_pos=cond["start_pos"],
            goal_pos=cond["goal_pos"],
            start_vel=cond["start_vel"],
            env_encoding=cond["env_encoding"],
        )

    def diffusion_loss(
        self,
        x0: torch.Tensor,
        robot_state: torch.Tensor,
        local_map: torch.Tensor,
        semantic_cost_map: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        goal_room_id: Optional[torch.Tensor] = None,
        goal_image_feat: Optional[torch.Tensor] = None,
        goal_text_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = x0.shape[0]
        t = torch.randint(0, self.config.diffusion_steps, (bsz,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.predict_noise(
            x_t=x_t,
            t=t,
            robot_state=robot_state,
            local_map=local_map,
            semantic_cost_map=semantic_cost_map,
            goal_state=goal_state,
            goal_room_id=goal_room_id,
            goal_image_feat=goal_image_feat,
            goal_text_feat=goal_text_feat,
        )
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(
        self,
        robot_state: torch.Tensor,
        local_map: torch.Tensor,
        semantic_cost_map: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        goal_room_id: Optional[torch.Tensor] = None,
        goal_image_feat: Optional[torch.Tensor] = None,
        goal_text_feat: Optional[torch.Tensor] = None,
        num_samples: int = 16,
    ) -> Dict[str, Any]:
        bsz = robot_state.shape[0]
        h = self.config.horizon
        d = self.config.action_dim

        robot_state = robot_state.repeat_interleave(num_samples, dim=0)
        local_map = local_map.repeat_interleave(num_samples, dim=0)
        semantic_cost_map = semantic_cost_map.repeat_interleave(num_samples, dim=0)

        if goal_state is not None:
            goal_state = goal_state.repeat_interleave(num_samples, dim=0)
        if goal_room_id is not None:
            goal_room_id = goal_room_id.repeat_interleave(num_samples, dim=0)
        if goal_image_feat is not None:
            goal_image_feat = goal_image_feat.repeat_interleave(num_samples, dim=0)
        if goal_text_feat is not None:
            goal_text_feat = goal_text_feat.repeat_interleave(num_samples, dim=0)

        x_t = torch.randn(bsz * num_samples, h, d, device=robot_state.device, dtype=robot_state.dtype)

        for step in reversed(range(self.config.diffusion_steps)):
            t = torch.full((x_t.shape[0],), step, device=x_t.device, dtype=torch.long)
            eps = self.predict_noise(
                x_t=x_t,
                t=t,
                robot_state=robot_state,
                local_map=local_map,
                semantic_cost_map=semantic_cost_map,
                goal_state=goal_state,
                goal_room_id=goal_room_id,
                goal_image_feat=goal_image_feat,
                goal_text_feat=goal_text_feat,
            )

            alpha_t = self._extract(self.alphas, t, x_t)
            alpha_bar_t = self._extract(self.alpha_bars, t, x_t)
            beta_t = self._extract(self.betas, t, x_t)

            mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps)

            if step > 0:
                z = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * z
            else:
                x_t = mean

        trajectories = x_t

        # Proposal scoring: goal-reaching + smoothness heuristic
        if goal_state is not None:
            goal_pos = goal_state[:, :d]
            goal_err = torch.norm(trajectories[:, -1, :] - goal_pos, dim=-1)
        else:
            goal_err = torch.norm(trajectories[:, -1, :], dim=-1)

        accel = trajectories[:, 2:, :] - 2 * trajectories[:, 1:-1, :] + trajectories[:, :-2, :]
        smooth_cost = accel.norm(dim=-1).mean(dim=-1) if accel.shape[1] > 0 else torch.zeros_like(goal_err)

        proposal_scores = 1.0 / (1.0 + goal_err + 0.5 * smooth_cost)
        mode_weights = proposal_scores / proposal_scores.sum().clamp(min=1e-6)

        semantic_tags = self._infer_semantic_tags(trajectories)

        return {
            "trajectories": trajectories,
            "proposal_scores": proposal_scores,
            "mode_weights": mode_weights,
            "semantic_tags": semantic_tags,
            "num_samples_per_batch": num_samples,
        }

    def _infer_semantic_tags(self, trajectories: torch.Tensor) -> List[str]:
        tags: List[str] = []
        for traj in trajectories:
            delta = traj[-1] - traj[0]
            lateral = delta[1].item() if traj.shape[-1] >= 2 else 0.0
            if lateral > 0.10:
                side_tag = "right_detour"
            elif lateral < -0.10:
                side_tag = "left_detour"
            else:
                side_tag = "center_passage"

            if traj.shape[0] >= 3:
                second = traj[2:] - 2.0 * traj[1:-1] + traj[:-2]
                curvature = second.norm(dim=-1).mean().item()
            else:
                curvature = 0.0
            motion_tag = "obstacle_avoidance" if curvature > 0.03 else "direct_to_goal"
            tags.append(f"{side_tag}|{motion_tag}")
        return tags


def create_semantic_diffusion_planner(**kwargs) -> SemanticConditionedDiffusionPlanner:
    config = SemanticDiffusionConfig(**kwargs)
    return SemanticConditionedDiffusionPlanner(config)
