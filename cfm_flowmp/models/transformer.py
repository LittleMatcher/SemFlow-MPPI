"""
FlowMP Transformer Architecture

A Transformer-based conditional vector field prediction network for
trajectory generation using Conditional Flow Matching.

Architecture:
- Input Projection: Maps trajectory state to hidden dimension
- Transformer Blocks with AdaLN conditioning
- Multi-Head Self-Attention for temporal dependencies  
- Output Head: Predicts velocity, acceleration, and jerk fields
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple

from .embeddings import (
    GaussianFourierProjection,
    SinusoidalPositionalEncoding,
    ConditionEncoder,
    AdaLN,
    AdaLNZero,
    CombinedEmbedding,
)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with optional flash attention.
    
    Captures temporal dependencies between trajectory time steps.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout  # Store dropout rate for flash attention
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # QKV projection
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=qkv_bias)
        
        # Output projection
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [B, T, hidden_dim]
            attn_mask: Optional attention mask [B, T, T] or [T, T]
            
        Returns:
            Output features [B, T, hidden_dim]
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Try to use Flash Attention if available (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's efficient attention
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # Manual attention computation
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
            
            if attn_mask is not None:
                attn = attn + attn_mask
            
            attn = F.softmax(attn, dim=-1)
            attn = self.proj_drop(attn)
            attn_out = attn @ v  # [B, num_heads, T, head_dim]
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T, C)
        out = self.proj(attn_out)
        
        return out


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) with GeLU activation.
    
    Standard transformer FFN with expansion factor.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = hidden_dim * expansion_factor
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """
    Cross-Attention module for condition injection.
    
    Allows the trajectory tokens to attend to condition tokens,
    enabling flexible condition injection as an alternative to AdaLN.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query from trajectory, Key/Value from condition
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,           # [B, T, D] trajectory tokens
        cond_tokens: torch.Tensor, # [B, N, D] condition tokens
    ) -> torch.Tensor:
        B, T, D = x.shape
        N = cond_tokens.shape[1]
        
        # Project queries from trajectory
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        # Project keys and values from condition
        kv = self.kv_proj(cond_tokens).reshape(B, N, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2).contiguous(), kv[:, :, 1].transpose(1, 2).contiguous()
        
        # Cross-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block with AdaLN conditioning.
    
    Structure (following Mizuta et al. and DiT):
    1. AdaLN → Multi-Head Self-Attention → Residual
    2. (Optional) Cross-Attention with condition tokens
    3. AdaLN → Feed-Forward Network → Residual
    
    The time embedding and condition are injected via AdaLN.
    Optionally, condition can also be injected via Cross-Attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int = 8,
        expansion_factor: int = 4,
        dropout: float = 0.0,
        use_adaln_zero: bool = True,
        use_cross_attention: bool = False,
    ):
        """
        Args:
            hidden_dim: Transformer hidden dimension
            cond_dim: Conditioning signal dimension
            num_heads: Number of attention heads
            expansion_factor: FFN expansion factor
            dropout: Dropout rate
            use_adaln_zero: Use AdaLN-Zero (recommended for stable training)
            use_cross_attention: Use cross-attention for condition injection
        """
        super().__init__()
        self.use_adaln_zero = use_adaln_zero
        self.use_cross_attention = use_cross_attention
        
        # Self-Attention block
        self.attn = MultiHeadSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Optional Cross-Attention for condition
        if use_cross_attention:
            self.cross_attn = CrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward block with GeLU activation
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        
        # AdaLN for conditioning
        if use_adaln_zero:
            self.adaln1 = AdaLNZero(hidden_dim, cond_dim)
            self.adaln2 = AdaLNZero(hidden_dim, cond_dim)
        else:
            self.adaln1 = AdaLN(hidden_dim, cond_dim)
            self.adaln2 = AdaLN(hidden_dim, cond_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cond_tokens: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [B, T, hidden_dim]
            cond: Combined time + condition embedding [B, cond_dim] for AdaLN
            cond_tokens: Condition tokens [B, N, hidden_dim] for cross-attention
            attn_mask: Optional attention mask
            
        Returns:
            Output features [B, T, hidden_dim]
        """
        # Self-Attention block with AdaLN
        if self.use_adaln_zero:
            x_norm, gate1 = self.adaln1(x, cond)
            x = x + gate1 * self.attn(x_norm, attn_mask)
        else:
            x = x + self.attn(self.adaln1(x, cond), attn_mask)
        
        # Cross-Attention with condition tokens (optional)
        if self.use_cross_attention and cond_tokens is not None:
            x = x + self.cross_attn(self.cross_attn_norm(x), cond_tokens)
        
        # FFN block with AdaLN
        if self.use_adaln_zero:
            x_norm, gate2 = self.adaln2(x, cond)
            x = x + gate2 * self.ffn(x_norm)
        else:
            x = x + self.ffn(self.adaln2(x, cond))
        
        return x


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding for trajectory time steps.
    """
    
    def __init__(self, max_seq_len: int, hidden_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        B, T, _ = x.shape
        return x + self.pos_embed[:, :T, :]


class SinusoidalSeqPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence positions.
    """
    
    def __init__(self, max_seq_len: int, hidden_dim: int):
        super().__init__()
        
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        
        pe = torch.zeros(1, max_seq_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x + self.pe[:, :T, :]


class FlowMPTransformer(nn.Module):
    """
    FlowMP Transformer: Conditional Vector Field Prediction Network
    
    A Transformer-based architecture for predicting the vector field
    in Conditional Flow Matching for trajectory generation.
    
    Based on:
    - FlowMP official implementation (https://github.com/mkhangg/flow_mp)
    - "Unified Generation-Refinement Planning" architecture guidelines
    - DiT-style AdaLN conditioning
    
    Input:
        - x_t: Current interpolated state [B, T, 6] containing (x, y, vx, vy, ax, ay)
        - t: Flow time ∈ [0, 1]
        - c: Task condition (start/goal)
    
    Output:
        - Predicted vector field [B, T, 6] containing (u, v, w) for position,
          velocity, and acceleration respectively
    
    Conditioning Options:
        1. AdaLN: Time and condition modulate layer norm parameters
        2. Cross-Attention: Condition tokens attend via cross-attention
        3. Token Prepending: Condition as prefix tokens (concatenated)
    """
    
    def __init__(
        self,
        # State dimensions
        state_dim: int = 2,  # Position dimension (x, y)
        input_channels: int = 6,  # pos(2) + vel(2) + acc(2)
        output_channels: int = 6,  # velocity_field(2) + acc_field(2) + jerk_field(2)
        
        # Sequence parameters
        max_seq_len: int = 64,  # Maximum trajectory length
        
        # Transformer parameters (following Mizuta et al.)
        hidden_dim: int = 256,
        num_layers: int = 8,  # Default 8 layers as per spec
        num_heads: int = 8,   # Multi-Head Self-Attention
        expansion_factor: int = 4,
        dropout: float = 0.1,
        
        # Time embedding parameters
        time_embed_type: str = "fourier",  # "fourier" or "sinusoidal"
        time_embed_dim: int = 256,
        fourier_scale: float = 30.0,
        
        # Condition parameters
        include_start_velocity: bool = True,
        include_goal_velocity: bool = False,
        env_encoding_dim: int = 0,
        
        # Condition injection method
        condition_type: str = "adaln",  # "adaln", "cross_attention", "token"
        
        # Position encoding
        pos_encoding_type: str = "sinusoidal",  # "learned" or "sinusoidal"
        
        # Training options
        use_adaln_zero: bool = True,
    ):
        """
        Initialize FlowMP Transformer.
        
        Args:
            state_dim: Dimension of position space (typically 2 for 2D)
            input_channels: Number of input channels (pos + vel + acc)
            output_channels: Number of output channels (u + v + w fields)
            max_seq_len: Maximum sequence length
            hidden_dim: Transformer hidden dimension (D=256 recommended)
            num_layers: Number of transformer layers (L=8 recommended)
            num_heads: Number of attention heads (4-8 recommended)
            expansion_factor: FFN expansion factor
            dropout: Dropout rate
            time_embed_type: Type of time embedding ("fourier" or "sinusoidal")
            time_embed_dim: Dimension of time embedding
            fourier_scale: Scale for Gaussian Fourier projection
            include_start_velocity: Include velocity in start condition
            include_goal_velocity: Include velocity in goal condition
            env_encoding_dim: Dimension of environment encoding
            condition_type: How to inject conditions ("adaln", "cross_attention", "token")
            pos_encoding_type: Type of positional encoding
            use_adaln_zero: Use AdaLN-Zero for stable training
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.condition_type = condition_type
        
        # ==================== Trajectory Tokenizer ====================
        # Project input state to hidden dimension D (Linear Layer as per spec)
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.GELU(),  # GeLU activation as per spec
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # ==================== Positional Encoding ====================
        # Sinusoidal positional encoding as per spec
        if pos_encoding_type == "learned":
            self.pos_encoding = LearnedPositionalEncoding(max_seq_len, hidden_dim)
        else:
            self.pos_encoding = SinusoidalSeqPositionalEncoding(max_seq_len, hidden_dim)
        
        # ==================== Time Embedding ====================
        # Sinusoidal or Gaussian Fourier projection for flow time t ∈ [0,1]
        if time_embed_type == "fourier":
            self.time_embed = GaussianFourierProjection(
                embed_dim=time_embed_dim,
                scale=fourier_scale,
            )
        else:
            self.time_embed = SinusoidalPositionalEncoding(embed_dim=time_embed_dim)
        
        # ==================== Condition Encoder ====================
        self.cond_encoder = ConditionEncoder(
            state_dim=state_dim,
            embed_dim=time_embed_dim,
            include_start_velocity=include_start_velocity,
            include_goal_velocity=include_goal_velocity,
            env_encoding_dim=env_encoding_dim,
        )
        
        # ==================== Condition Injection ====================
        cond_dim = time_embed_dim
        use_cross_attention = (condition_type == "cross_attention")
        
        if condition_type == "token":
            # Condition as extra tokens (prepended to sequence)
            # Start token + Goal token = 2 condition tokens
            self.cond_token_proj = nn.Linear(time_embed_dim, hidden_dim)
            self.num_cond_tokens = 2
            
            # Still need cond_combine for AdaLN normalization
            self.cond_combine = nn.Sequential(
                nn.Linear(time_embed_dim * 2, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:
            # AdaLN or Cross-Attention: combine time and condition
            self.cond_combine = nn.Sequential(
                nn.Linear(time_embed_dim * 2, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
            )
            
            # For adaln mode: project time embedding to hidden_dim for token addition
            if condition_type == "adaln":
                self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        
        # For cross-attention: project condition to tokens
        if use_cross_attention:
            self.cond_token_proj = nn.Sequential(
                nn.Linear(time_embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # ==================== Transformer Encoder Blocks ====================
        # Stack L=8 layers as per "Unified Generation-Refinement Planning"
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                dropout=dropout,
                use_adaln_zero=use_adaln_zero,
                use_cross_attention=use_cross_attention,
            )
            for _ in range(num_layers)
        ])
        
        # ==================== Output Head (Unembedding) ====================
        # Linear layer mapping D back to original state dimension 6
        self.final_adaln = AdaLN(hidden_dim, cond_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),  # GeLU activation
            nn.Linear(hidden_dim, output_channels),
        )
        
        # Initialize output layer to zero for stable training (residual connection style)
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
        
        # Store config
        self.config = {
            'state_dim': state_dim,
            'input_channels': input_channels,
            'output_channels': output_channels,
            'max_seq_len': max_seq_len,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'condition_type': condition_type,
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
        Forward pass of FlowMP Transformer.
        
        Implements the forward pass following FlowMP architecture:
        1. Tokenize trajectory state via Linear projection
        2. Add positional encoding
        3. Inject time embedding (added to tokens as per spec)
        4. Process through Transformer Encoder blocks with condition
        5. Output vector field via unembedding layer
        
        Args:
            x_t: Interpolated state at flow time t, shape [B, T, input_channels]
                 Contains (x, y, vx, vy, ax, ay) concatenated
            t: Flow time values, shape [B] or [B, 1], values in [0, 1]
            start_pos: Starting position, shape [B, state_dim]
            goal_pos: Goal position, shape [B, state_dim]
            start_vel: Starting velocity (optional), shape [B, state_dim]
            goal_vel: Goal velocity (optional), shape [B, state_dim]
            env_encoding: Environment encoding (optional), shape [B, env_encoding_dim]
            attn_mask: Optional attention mask
            
        Returns:
            Predicted vector field, shape [B, T, output_channels]
            - Channels 0:2 = velocity field (u) for position (ṗ prediction)
            - Channels 2:4 = acceleration field (v) for velocity (p̈ prediction)
            - Channels 4:6 = jerk field (w) for acceleration (p⃛ prediction)
        """
        B, T, C = x_t.shape
        
        # ==================== Trajectory Tokenizer ====================
        # Project input state to hidden dimension D
        h = self.input_proj(x_t)  # [B, T, hidden_dim]
        
        # ==================== Add Positional Encoding ====================
        h = self.pos_encoding(h)  # [B, T, hidden_dim]
        
        # ==================== Time Embedding ====================
        # Map flow time t ∈ [0,1] to vector via sinusoidal/Fourier encoding
        if t.dim() == 2:
            t = t.squeeze(-1)  # [B]
        time_emb = self.time_embed(t)  # [B, time_embed_dim]
        
        # ==================== Condition Embedding ====================
        cond_emb = self.cond_encoder(
            start_pos=start_pos,
            goal_pos=goal_pos,
            start_vel=start_vel,
            goal_vel=goal_vel,
            env_encoding=env_encoding,
        )  # [B, time_embed_dim]
        
        # ==================== Condition Injection ====================
        cond_tokens = None  # For cross-attention
        
        if self.condition_type == "token":
            # Prepend condition as extra tokens
            # Create start and goal condition tokens
            cond_token = self.cond_token_proj(cond_emb)  # [B, hidden_dim]
            time_token = self.cond_token_proj(time_emb)  # [B, hidden_dim]
            
            # Stack as [B, 2, hidden_dim] and prepend to trajectory
            prefix_tokens = torch.stack([time_token, cond_token], dim=1)
            h = torch.cat([prefix_tokens, h], dim=1)  # [B, T+2, hidden_dim]
            
            # Combined condition for AdaLN (still needed for normalization)
            combined_cond = self.cond_combine(
                torch.cat([time_emb, cond_emb], dim=-1)
            )
            
        elif self.condition_type == "cross_attention":
            # Condition as tokens for cross-attention
            cond_tokens = self.cond_token_proj(cond_emb).unsqueeze(1)  # [B, 1, hidden_dim]
            time_tokens = self.cond_token_proj(time_emb).unsqueeze(1)  # [B, 1, hidden_dim]
            cond_tokens = torch.cat([time_tokens, cond_tokens], dim=1)  # [B, 2, hidden_dim]
            
            # Combined condition for AdaLN
            combined_cond = self.cond_combine(
                torch.cat([time_emb, cond_emb], dim=-1)
            )
            
        else:  # "adaln" (default)
            # Time embedding added to token embedding as per spec
            # Then combined with condition for AdaLN modulation
            time_emb_proj = self.time_proj(time_emb)  # Project to hidden_dim
            h = h + time_emb_proj.unsqueeze(1)  # Add time embedding to all tokens
            
            combined_cond = self.cond_combine(
                torch.cat([time_emb, cond_emb], dim=-1)
            )  # [B, cond_dim]
        
        # ==================== Transformer Encoder Blocks ====================
        for block in self.blocks:
            h = block(h, combined_cond, cond_tokens, attn_mask)
        
        # ==================== Remove prefix tokens if added ====================
        if self.condition_type == "token":
            h = h[:, 2:, :]  # Remove the 2 prefix tokens
        
        # ==================== Output Head (Unembedding) ====================
        h = self.final_adaln(h, combined_cond)
        output = self.output_head(h)  # [B, T, output_channels]
        
        return output
    
    def get_velocity_field(self, output: torch.Tensor) -> torch.Tensor:
        """Extract position velocity field from output."""
        return output[..., :self.state_dim]
    
    def get_acceleration_field(self, output: torch.Tensor) -> torch.Tensor:
        """Extract velocity acceleration field from output."""
        return output[..., self.state_dim:self.state_dim*2]
    
    def get_jerk_field(self, output: torch.Tensor) -> torch.Tensor:
        """Extract acceleration jerk field from output."""
        return output[..., self.state_dim*2:self.state_dim*3]
    
    @torch.no_grad()
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FlowMPTransformerSmall(FlowMPTransformer):
    """Small variant of FlowMP Transformer (~3M parameters)."""
    
    def __init__(self, **kwargs):
        defaults = {
            'hidden_dim': 128,
            'num_layers': 4,
            'num_heads': 4,
            'time_embed_dim': 128,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class FlowMPTransformerBase(FlowMPTransformer):
    """
    Base variant of FlowMP Transformer (~15M parameters).
    
    Follows the recommended configuration from Mizuta et al.:
    - L=8 Transformer Encoder Layers
    - D=256 hidden dimension
    - 8 attention heads
    """
    
    def __init__(self, **kwargs):
        defaults = {
            'hidden_dim': 256,
            'num_layers': 8,  # 8 layers as per spec
            'num_heads': 8,
            'time_embed_dim': 256,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class FlowMPTransformerLarge(FlowMPTransformer):
    """Large variant of FlowMP Transformer (~50M parameters)."""
    
    def __init__(self, **kwargs):
        defaults = {
            'hidden_dim': 512,
            'num_layers': 12,
            'num_heads': 16,
            'time_embed_dim': 512,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


def create_flowmp_transformer(
    variant: str = "base",
    **kwargs
) -> FlowMPTransformer:
    """
    Factory function to create FlowMP Transformer variants.
    
    Args:
        variant: Model size variant ("small", "base", "large")
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        FlowMPTransformer instance
    """
    variants = {
        "small": FlowMPTransformerSmall,
        "base": FlowMPTransformerBase,
        "large": FlowMPTransformerLarge,
    }
    
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")
    
    return variants[variant](**kwargs)
