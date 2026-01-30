"""
Embedding and Conditioning Modules for FlowMP

Contains:
- GaussianFourierProjection: Maps scalar time t to high-dimensional embedding
- SinusoidalPositionalEncoding: Alternative time/position encoding
- ConditionEncoder: Encodes start/goal conditions
- AdaLN: Adaptive Layer Normalization
- FiLM: Feature-wise Linear Modulation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier Projection for time embedding.
    
    Maps scalar t ∈ [0, 1] to a high-dimensional embedding using
    random Fourier features with learned or fixed frequencies.
    
    Reference: "Fourier Features Let Networks Learn High Frequency Functions"
    https://arxiv.org/abs/2006.10739
    """
    
    def __init__(
        self, 
        embed_dim: int = 256, 
        scale: float = 30.0,
        learnable: bool = False
    ):
        """
        Args:
            embed_dim: Output embedding dimension (will be embed_dim * 2 due to sin/cos)
            scale: Standard deviation for random frequencies
            learnable: Whether to make frequencies learnable
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        
        # Random frequencies sampled from N(0, scale^2)
        W = torch.randn(embed_dim) * scale
        if learnable:
            self.W = nn.Parameter(W)
        else:
            self.register_buffer('W', W)
        
        # Output projection to desired dimension
        self.output_proj = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values of shape [B] or [B, 1], values in [0, 1]
            
        Returns:
            Time embeddings of shape [B, embed_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        
        # Compute Fourier features: [B, embed_dim]
        t_proj = t * self.W[None, :] * 2 * math.pi  # [B, embed_dim]
        
        # Concatenate sin and cos: [B, embed_dim * 2]
        embedding = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        
        # Project to output dimension
        return self.output_proj(embedding)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time embedding.
    
    Standard transformer-style positional encoding adapted for
    continuous time values.
    """
    
    def __init__(self, embed_dim: int = 256, max_period: float = 10000.0):
        """
        Args:
            embed_dim: Output embedding dimension
            max_period: Maximum period for the sinusoidal functions
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        
        # Precompute frequency bands
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values of shape [B] or [B, 1], values in [0, 1]
            
        Returns:
            Time embeddings of shape [B, embed_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        
        # Scale time to larger range for better frequency coverage
        t = t * 1000.0  # [B, 1]
        
        # Compute embeddings
        args = t * self.freqs[None, :]  # [B, half_dim]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, embed_dim]
        
        return self.output_proj(embedding)


class ConditionEncoder(nn.Module):
    """
    Encodes task conditions (start point, goal point, optional environment).
    
    Condition format:
    - Start state: (q_start, q_dot_start) ∈ R^4 (position + velocity)
    - Goal state: q_goal ∈ R^2 (position only)
    - Optional: Environment encoding from CNN
    """
    
    def __init__(
        self,
        state_dim: int = 2,  # Position dimension (x, y)
        embed_dim: int = 256,
        include_start_velocity: bool = True,
        include_goal_velocity: bool = False,
        env_encoding_dim: int = 0,  # 0 means no environment encoding
    ):
        """
        Args:
            state_dim: Dimension of position (typically 2 for 2D)
            embed_dim: Output embedding dimension
            include_start_velocity: Whether start condition includes velocity
            include_goal_velocity: Whether goal condition includes velocity
            env_encoding_dim: Dimension of environment encoding (0 to disable)
        """
        super().__init__()
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.include_start_velocity = include_start_velocity
        self.include_goal_velocity = include_goal_velocity
        
        # Calculate input dimension
        start_dim = state_dim * (2 if include_start_velocity else 1)  # pos + vel
        goal_dim = state_dim * (2 if include_goal_velocity else 1)
        total_cond_dim = start_dim + goal_dim + env_encoding_dim
        
        # MLP encoder for conditions
        self.encoder = nn.Sequential(
            nn.Linear(total_cond_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.env_encoding_dim = env_encoding_dim
    
    def forward(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        start_vel: torch.Tensor = None,
        goal_vel: torch.Tensor = None,
        env_encoding: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            start_pos: [B, state_dim] starting position
            goal_pos: [B, state_dim] goal position
            start_vel: [B, state_dim] starting velocity (optional)
            goal_vel: [B, state_dim] goal velocity (optional)
            env_encoding: [B, env_encoding_dim] environment encoding (optional)
            
        Returns:
            Condition embedding of shape [B, embed_dim]
        """
        # Build condition vector
        cond_parts = [start_pos]
        
        if self.include_start_velocity:
            if start_vel is None:
                start_vel = torch.zeros_like(start_pos)
            cond_parts.append(start_vel)
        
        cond_parts.append(goal_pos)
        
        if self.include_goal_velocity:
            if goal_vel is None:
                goal_vel = torch.zeros_like(goal_pos)
            cond_parts.append(goal_vel)
        
        if self.env_encoding_dim > 0:
            if env_encoding is None:
                env_encoding = torch.zeros(
                    start_pos.shape[0], self.env_encoding_dim,
                    device=start_pos.device, dtype=start_pos.dtype
                )
            cond_parts.append(env_encoding)
        
        # Concatenate and encode
        cond = torch.cat(cond_parts, dim=-1)
        return self.encoder(cond)


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN).
    
    Modulates layer normalization parameters based on conditioning signal.
    Used in DiT (Diffusion Transformers) and flow matching models.
    
    AdaLN(h, c) = γ(c) * LayerNorm(h) + β(c)
    
    Reference: "Scalable Diffusion Models with Transformers" (DiT)
    https://arxiv.org/abs/2212.09748
    """
    
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        eps: float = 1e-6,
    ):
        """
        Args:
            hidden_dim: Dimension of features to normalize
            cond_dim: Dimension of conditioning signal
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        # LayerNorm without learnable parameters (we use adaptive ones)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
        
        # MLP to predict scale (γ) and shift (β) from condition
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )
        
        # Initialize to identity transformation
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, T, hidden_dim] or [B, hidden_dim]
            cond: Conditioning signal [B, cond_dim]
            
        Returns:
            Normalized and modulated features, same shape as x
        """
        # Predict modulation parameters
        modulation = self.adaLN_modulation(cond)  # [B, hidden_dim * 2]
        scale, shift = modulation.chunk(2, dim=-1)  # [B, hidden_dim] each
        
        # Add dimensions for broadcasting if needed
        if x.dim() == 3:
            scale = scale.unsqueeze(1)  # [B, 1, hidden_dim]
            shift = shift.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Apply adaptive layer norm
        x_norm = self.norm(x)
        return x_norm * (1 + scale) + shift


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero: Adaptive Layer Normalization with zero initialization.
    
    Used in DiT for the final modulation before residual connection.
    Includes an additional gate parameter initialized to zero.
    
    AdaLN-Zero(h, c) = gate(c) * (γ(c) * LayerNorm(h) + β(c))
    """
    
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
        
        # MLP to predict scale (γ), shift (β), and gate (α)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 3),
        )
        
        # Initialize to zero for stable training
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple:
        """
        Args:
            x: Input features [B, T, hidden_dim] or [B, hidden_dim]
            cond: Conditioning signal [B, cond_dim]
            
        Returns:
            Tuple of (normalized features, gate) for use in residual
        """
        modulation = self.adaLN_modulation(cond)  # [B, hidden_dim * 3]
        scale, shift, gate = modulation.chunk(3, dim=-1)
        
        if x.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
            gate = gate.unsqueeze(1)
        
        x_norm = self.norm(x)
        return x_norm * (1 + scale) + shift, gate


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    
    Alternative to AdaLN that directly modulates features without normalization.
    
    FiLM(h, c) = γ(c) * h + β(c)
    
    Reference: "FiLM: Visual Reasoning with a General Conditioning Layer"
    https://arxiv.org/abs/1709.07871
    """
    
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
    ):
        """
        Args:
            hidden_dim: Dimension of features to modulate
            cond_dim: Dimension of conditioning signal
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # MLP to predict scale (γ) and shift (β)
        self.film_generator = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )
        
        # Initialize close to identity
        nn.init.zeros_(self.film_generator[-1].bias)
        nn.init.xavier_uniform_(self.film_generator[-1].weight, gain=0.02)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, T, hidden_dim] or [B, hidden_dim]
            cond: Conditioning signal [B, cond_dim]
            
        Returns:
            Modulated features, same shape as x
        """
        modulation = self.film_generator(cond)
        scale, shift = modulation.chunk(2, dim=-1)
        
        if x.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        return x * (1 + scale) + shift


class CombinedEmbedding(nn.Module):
    """
    Combines time embedding and condition embedding into a single
    conditioning signal for use in AdaLN layers.
    """
    
    def __init__(
        self,
        time_embed_dim: int = 256,
        cond_embed_dim: int = 256,
        output_dim: int = 256,
        time_embedding_type: str = "fourier",  # "fourier" or "sinusoidal"
        fourier_scale: float = 30.0,
    ):
        super().__init__()
        
        # Time embedding
        if time_embedding_type == "fourier":
            self.time_embed = GaussianFourierProjection(
                embed_dim=time_embed_dim,
                scale=fourier_scale,
            )
        else:
            self.time_embed = SinusoidalPositionalEncoding(
                embed_dim=time_embed_dim,
            )
        
        # Combined projection
        self.combined_proj = nn.Sequential(
            nn.Linear(time_embed_dim + cond_embed_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(
        self, 
        t: torch.Tensor, 
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            t: Time values [B] or [B, 1]
            cond: Condition embedding [B, cond_embed_dim]
            
        Returns:
            Combined embedding [B, output_dim]
        """
        time_emb = self.time_embed(t)  # [B, time_embed_dim]
        combined = torch.cat([time_emb, cond], dim=-1)
        return self.combined_proj(combined)
