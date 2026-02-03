"""
FlowMP 1D U-Net Architecture

A 1D U-Net conditional vector field prediction network for
trajectory generation using Conditional Flow Matching.

Input:
    - x_t: [B, T, 6] (pos, vel, acc)
    - t: [B]
    - start/goal conditions

Output:
    - [B, T, 6] (u, v, w fields)
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import (
    GaussianFourierProjection,
    SinusoidalPositionalEncoding,
    ConditionEncoder,
)



def _get_num_groups(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    if channels % groups != 0:
        groups = 1
    return groups


class ResBlock1D(nn.Module):
    """Conditional residual block with FiLM-style modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        dropout: float = 0.0,
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        norm1_groups = _get_num_groups(in_channels, num_groups)
        norm2_groups = _get_num_groups(out_channels, num_groups)

        self.norm1 = nn.GroupNorm(num_groups=norm1_groups, num_channels=in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=norm2_groups, num_channels=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2),
        )

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
            cond: [B, cond_dim]
        Returns:
            [B, C_out, T]
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class Downsample1D(nn.Module):
    """Downsample by 2 using strided conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.down = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Upsample1D(nn.Module):
    """Upsample by 2 using transposed conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != target_len:
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x


class FlowMPUNet1D(nn.Module):
    """
    FlowMP 1D U-Net for conditional vector field prediction.

    Uses time + condition embeddings to modulate residual blocks.
    """

    def __init__(
        self,
        state_dim: int = 2,
        input_channels: int = 6,
        output_channels: int = 6,
        max_seq_len: int = 64,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 1,
        dropout: float = 0.0,
        time_embed_type: str = "fourier",
        time_embed_dim: int = 256,
        fourier_scale: float = 30.0,
        include_start_velocity: bool = True,
        include_goal_velocity: bool = False,
        env_encoding_dim: int = 0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.max_seq_len = max_seq_len

        # ==================== Embeddings ====================
        if time_embed_type == "fourier":
            self.time_embed = GaussianFourierProjection(
                embed_dim=time_embed_dim,
                scale=fourier_scale,
            )
        else:
            self.time_embed = SinusoidalPositionalEncoding(embed_dim=time_embed_dim)

        self.cond_encoder = ConditionEncoder(
            state_dim=state_dim,
            embed_dim=time_embed_dim,
            include_start_velocity=include_start_velocity,
            include_goal_velocity=include_goal_velocity,
            env_encoding_dim=env_encoding_dim,
        )

        self.cond_combine = nn.Sequential(
            nn.Linear(time_embed_dim * 2, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        cond_dim = time_embed_dim

        # ==================== U-Net ====================
        self.input_proj = nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1)

        # Down path
        down_channels = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = base_channels
        for i, ch in enumerate(down_channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock1D(in_ch, ch, cond_dim, dropout=dropout))
                in_ch = ch
            self.down_blocks.append(blocks)

            if i != len(down_channels) - 1:
                self.downsamples.append(Downsample1D(in_ch))
            else:
                self.downsamples.append(nn.Identity())

        # Middle
        self.mid_block1 = ResBlock1D(in_ch, in_ch, cond_dim, dropout=dropout)
        self.mid_block2 = ResBlock1D(in_ch, in_ch, cond_dim, dropout=dropout)

        # Up path
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in reversed(range(len(down_channels) - 1)):
            out_ch = down_channels[i]
            self.upsamples.append(Upsample1D(in_ch, out_ch))
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                # First block takes concatenated skip (out_ch + out_ch)
                # Subsequent blocks take just out_ch
                in_block_ch = out_ch + out_ch if j == 0 else out_ch
                blocks.append(ResBlock1D(in_block_ch, out_ch, cond_dim, dropout=dropout))
            self.up_blocks.append(blocks)
            in_ch = out_ch

        # Output
        self.out_norm = nn.GroupNorm(
            num_groups=_get_num_groups(base_channels),
            num_channels=base_channels,
        )
        self.out_conv = nn.Conv1d(base_channels, output_channels, kernel_size=3, padding=1)

        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self.config = {
            "state_dim": state_dim,
            "input_channels": input_channels,
            "output_channels": output_channels,
            "max_seq_len": max_seq_len,
            "base_channels": base_channels,
            "channel_mults": channel_mults,
            "num_res_blocks": num_res_blocks,
            "time_embed_dim": time_embed_dim,
        }

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        start_vel: Optional[torch.Tensor] = None,
        goal_vel: Optional[torch.Tensor] = None,
        env_encoding: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t: [B, T, C]
            t: [B]
            start_pos: [B, D]
            goal_pos: [B, D]
        Returns:
            [B, T, output_channels]
        """
        if t.dim() == 2:
            t = t.squeeze(-1)

        time_emb = self.time_embed(t)
        cond_emb = self.cond_encoder(
            start_pos=start_pos,
            goal_pos=goal_pos,
            start_vel=start_vel,
            goal_vel=goal_vel,
            env_encoding=env_encoding,
        )
        cond = self.cond_combine(torch.cat([time_emb, cond_emb], dim=-1))

        # [B, T, C] -> [B, C, T]
        h = x_t.transpose(1, 2)
        h = self.input_proj(h)

        skips = []
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = block(h, cond)
            skips.append(h)
            h = down(h)

        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)

        # Remove the last skip (deepest layer) as it's not used in upsampling
        # The upsampling path has len(down_channels) - 1 blocks, so we need to remove one skip
        if len(skips) > len(self.up_blocks):
            skips.pop()  # Remove the deepest skip (corresponds to the layer before middle)

        for blocks, up in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            h = up(h, target_len=skip.shape[-1])
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, cond)

        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)

        return h.transpose(1, 2)

    def get_velocity_field(self, output: torch.Tensor) -> torch.Tensor:
        return output[..., :self.state_dim]

    def get_acceleration_field(self, output: torch.Tensor) -> torch.Tensor:
        return output[..., self.state_dim:self.state_dim * 2]

    def get_jerk_field(self, output: torch.Tensor) -> torch.Tensor:
        return output[..., self.state_dim * 2:self.state_dim * 3]


def create_flowmp_unet1d(
    base_channels: int = 128,
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
    **kwargs,
) -> FlowMPUNet1D:
    """
    Factory function to create 1D U-Net model.
    """
    return FlowMPUNet1D(
        base_channels=base_channels,
        channel_mults=channel_mults,
        **kwargs,
    )
