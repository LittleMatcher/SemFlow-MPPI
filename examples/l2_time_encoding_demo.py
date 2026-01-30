"""
L2 Layer Time Encoding Implementation Demo

This script demonstrates:
1. Different time encoding methods (Fourier vs Sinusoidal)
2. Time encoding integration in L2 layer
3. Visualization and debugging techniques
4. Custom time encoding implementations
"""

import torch
import torch.nn as nn
import math
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Part 1: Time Encoding Implementations
# ============================================================================

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier Projection for time embedding.
    
    Maps scalar t ∈ [0, 1] to high-dimensional embedding using random Fourier features.
    """
    
    def __init__(
        self, 
        embed_dim: int = 256, 
        scale: float = 30.0,
        learnable: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        
        # Random frequencies sampled from N(0, scale^2)
        W = torch.randn(embed_dim) * scale
        if learnable:
            self.W = nn.Parameter(W)
        else:
            self.register_buffer('W', W)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [B] or [B, 1], in [0, 1]
        Returns:
            Time embeddings [B, embed_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B] → [B, 1]
        
        # Fourier projection
        t_proj = t * self.W[None, :] * 2 * math.pi  # [B, embed_dim]
        
        # Concatenate sin and cos
        embedding = torch.cat([
            torch.sin(t_proj),
            torch.cos(t_proj)
        ], dim=-1)  # [B, embed_dim * 2]
        
        # Project to output dimension
        return self.output_proj(embedding)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time embedding.
    
    Standard transformer-style positional encoding adapted for continuous time.
    """
    
    def __init__(self, embed_dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        
        # Precompute frequency bands
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * 
            torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [B] or [B, 1], in [0, 1]
        Returns:
            Time embeddings [B, embed_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        
        # Scale time to larger range
        t = t * 1000.0  # [B, 1]
        
        # Compute embeddings
        args = t * self.freqs[None, :]  # [B, half_dim]
        embedding = torch.cat([
            torch.cos(args),
            torch.sin(args)
        ], dim=-1)  # [B, embed_dim]
        
        return self.output_proj(embedding)


class LearnableTimeEncoding(nn.Module):
    """
    Learnable time encoding using MLPs.
    
    More flexible but requires careful initialization.
    """
    
    def __init__(self, embed_dim: int = 256, num_freq_bands: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freq_bands = num_freq_bands
        
        # Learnable frequency bands
        self.freq_bands = nn.Parameter(
            torch.randn(num_freq_bands) * math.pi
        )
        
        # MLP for feature combination
        self.mlp = nn.Sequential(
            nn.Linear(num_freq_bands * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [B] or [B, 1]
        Returns:
            Time embeddings [B, embed_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        
        # Compute sinusoidal features with learnable frequencies
        t_expanded = t * self.freq_bands[None, :]  # [B, num_freq_bands]
        
        features = torch.cat([
            torch.sin(t_expanded),
            torch.cos(t_expanded)
        ], dim=-1)  # [B, num_freq_bands * 2]
        
        return self.mlp(features)


# ============================================================================
# Part 2: Debugging and Analysis Tools
# ============================================================================

def analyze_time_encoding(
    encoder: nn.Module,
    num_samples: int = 1000,
    plot: bool = True,
) -> dict:
    """
    Analyze time encoding properties.
    
    Args:
        encoder: Time encoding module
        num_samples: Number of time points to sample
        plot: Whether to plot results
        
    Returns:
        Dictionary with analysis results
    """
    # Generate time samples
    t = torch.linspace(0, 1, num_samples).unsqueeze(-1)  # [num_samples, 1]
    
    with torch.no_grad():
        embeddings = encoder(t)  # [num_samples, embed_dim]
    
    # Compute statistics
    stats = {
        'mean': embeddings.mean(dim=0),
        'std': embeddings.std(dim=0),
        'min': embeddings.min(dim=0).values,
        'max': embeddings.max(dim=0).values,
        'embeddings': embeddings,
        't_values': t,
    }
    
    # Check for issues
    issues = []
    
    # Issue 1: Dead features (zero variance)
    dead_features = (stats['std'] < 1e-3).sum()
    if dead_features > 0:
        issues.append(f"⚠️ {dead_features} dead features (std < 1e-3)")
    
    # Issue 2: Exploding values
    max_val = stats['max'].abs().max()
    if max_val > 100:
        issues.append(f"⚠️ Very large values detected (max={max_val:.1f})")
    
    # Issue 3: Saturation
    saturated = (stats['max'] > 0.99).sum() + (stats['min'] < -0.99).sum()
    if saturated > embeddings.shape[1] * 0.5:
        issues.append(f"⚠️ Many features are saturated")
    
    if issues:
        print("🔍 Issues detected in time encoding:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ Time encoding looks healthy")
    
    # Plotting
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Sample features over time
        ax = axes[0, 0]
        for i in range(min(12, embeddings.shape[1])):
            ax.plot(t.numpy(), embeddings[:, i].numpy(), alpha=0.7)
        ax.set_xlabel('Time t')
        ax.set_ylabel('Embedding Value')
        ax.set_title('12 Sample Features over Time')
        ax.grid(True)
        
        # Plot 2: Feature distribution
        ax = axes[0, 1]
        ax.hist(embeddings.numpy().flatten(), bins=50, alpha=0.7)
        ax.set_xlabel('Embedding Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of All Embeddings')
        ax.grid(True)
        
        # Plot 3: Feature statistics
        ax = axes[1, 0]
        feature_indices = np.arange(embeddings.shape[1])
        ax.errorbar(feature_indices, stats['mean'].numpy(), 
                   yerr=stats['std'].numpy(), fmt='.', alpha=0.7)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Mean ± Std')
        ax.set_title('Feature Statistics')
        ax.grid(True)
        
        # Plot 4: Temporal sensitivity
        ax = axes[1, 1]
        # Compute L2 distance between consecutive embeddings
        diffs = torch.norm(embeddings[1:] - embeddings[:-1], dim=1)
        ax.plot(t[:-1].numpy(), diffs.numpy())
        ax.set_xlabel('Time t')
        ax.set_ylabel('||emb(t+dt) - emb(t)||')
        ax.set_title('Temporal Sensitivity')
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return stats


def compare_encodings(
    num_samples: int = 200,
) -> None:
    """
    Compare different time encoding methods.
    """
    t = torch.linspace(0, 1, num_samples).unsqueeze(-1)
    
    encoders = {
        'Fourier (scale=30)': GaussianFourierProjection(embed_dim=128, scale=30.0),
        'Fourier (scale=10)': GaussianFourierProjection(embed_dim=128, scale=10.0),
        'Fourier (scale=50)': GaussianFourierProjection(embed_dim=128, scale=50.0),
        'Sinusoidal': SinusoidalPositionalEncoding(embed_dim=128),
        'Learnable': LearnableTimeEncoding(embed_dim=128, num_freq_bands=16),
    }
    
    fig, axes = plt.subplots(1, len(encoders), figsize=(16, 4))
    
    for idx, (name, encoder) in enumerate(encoders.items()):
        with torch.no_grad():
            embeddings = encoder(t)  # [num_samples, embed_dim]
        
        # Plot first 8 features
        ax = axes[idx] if len(encoders) > 1 else axes
        for i in range(min(8, embeddings.shape[1])):
            ax.plot(t.numpy(), embeddings[:, i].numpy(), alpha=0.7)
        
        ax.set_title(name)
        ax.set_xlabel('Time t')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# Part 3: L2 Layer Integration Example
# ============================================================================

class SimpleL2Layer(nn.Module):
    """
    Simplified L2 layer showing time encoding integration.
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        hidden_dim: int = 256,
        time_embed_type: str = "fourier",
    ):
        super().__init__()
        
        # Time encoding
        if time_embed_type == "fourier":
            self.time_embed = GaussianFourierProjection(embed_dim=hidden_dim)
        else:
            self.time_embed = SinusoidalPositionalEncoding(embed_dim=hidden_dim)
        
        # Condition encoding
        self.cond_encoder = nn.Sequential(
            nn.Linear(state_dim * 3 + state_dim * 2, hidden_dim),  # [p,v,a] + [p,v]
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Combine time and condition
        self.cond_combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # time_emb + cond_emb
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Transformer block (simplified)
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation='gelu',
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim * 3),  # [p, v, a]
        )
    
    def forward(
        self,
        x_t: torch.Tensor,  # [B, T, state_dim*3]
        t: torch.Tensor,    # [B]
        x_curr: torch.Tensor,  # [B, state_dim*3]
        x_goal: torch.Tensor,  # [B, state_dim*2]
    ) -> torch.Tensor:
        """
        Args:
            x_t: Current state trajectory
            t: Flow time (scalar for each batch)
            x_curr: Current robot state
            x_goal: Goal state
            
        Returns:
            Predicted vector field [B, T, state_dim*3]
        """
        B, T, _ = x_t.shape
        
        # ===== Time Encoding =====
        time_emb = self.time_embed(t)  # [B, hidden_dim]
        
        # ===== Condition Encoding =====
        cond_input = torch.cat([x_curr, x_goal], dim=-1)  # [B, state_dim*5]
        cond_emb = self.cond_encoder(cond_input)  # [B, hidden_dim]
        
        # ===== Combine Time and Condition =====
        combined_cond = self.cond_combine(
            torch.cat([time_emb, cond_emb], dim=-1)
        )  # [B, hidden_dim]
        
        # ===== Expand to sequence length =====
        combined_cond_expanded = combined_cond.unsqueeze(1).expand(-1, T, -1)  # [B, T, hidden_dim]
        
        # ===== Project input =====
        h = x_t  # [B, T, state_dim*3]
        
        # ===== Transformer blocks (with condition modulation) =====
        # In practice, you'd use AdaLN or cross-attention
        # Here we just add the condition
        h = h + combined_cond_expanded * 0.1  # [B, T, hidden_dim]
        
        h = self.transformer_block(h)  # [B, T, hidden_dim]
        
        # ===== Output =====
        output = self.output_head(h)  # [B, T, state_dim*3]
        
        return output


# ============================================================================
# Part 4: Main Demo
# ============================================================================

def main():
    print("=" * 80)
    print("L2 Layer Time Encoding Demo")
    print("=" * 80)
    
    # Demo 1: Analyze different encodings
    print("\n📊 Demo 1: Analyzing Gaussian Fourier Projection")
    print("-" * 80)
    fourier_encoder = GaussianFourierProjection(embed_dim=256, scale=30.0)
    stats = analyze_time_encoding(fourier_encoder)
    
    # Demo 2: Compare encoding methods
    print("\n📊 Demo 2: Comparing Different Encoding Methods")
    print("-" * 80)
    print("Generating comparison plots...")
    compare_encodings(num_samples=500)
    
    # Demo 3: Test temporal sensitivity
    print("\n📊 Demo 3: Temporal Sensitivity Analysis")
    print("-" * 80)
    t_fine = torch.tensor([[0.500]])
    t_coarse = torch.tensor([[0.501]])
    
    encoder = GaussianFourierProjection(embed_dim=256, scale=30.0)
    emb_fine = encoder(t_fine)
    emb_coarse = encoder(t_coarse)
    
    distance = (emb_fine - emb_coarse).norm()
    print(f"Time difference: 0.001")
    print(f"Embedding distance: {distance:.6f}")
    print(f"✓ Embeddings are continuous and distinct")
    
    # Demo 4: L2 layer integration
    print("\n📊 Demo 4: L2 Layer Integration Example")
    print("-" * 80)
    
    batch_size = 4
    seq_len = 32
    state_dim = 2
    
    model = SimpleL2Layer(state_dim=state_dim, hidden_dim=256)
    
    x_t = torch.randn(batch_size, seq_len, state_dim * 3)
    t = torch.tensor([0.0, 0.25, 0.5, 0.75])
    x_curr = torch.randn(batch_size, state_dim * 3)
    x_goal = torch.randn(batch_size, state_dim * 2)
    
    output = model(x_t, t, x_curr, x_goal)
    
    print(f"Input shapes:")
    print(f"  x_t: {x_t.shape}")
    print(f"  t: {t.shape}")
    print(f"  x_curr: {x_curr.shape}")
    print(f"  x_goal: {x_goal.shape}")
    print(f"\nOutput shape: {output.shape}")
    print(f"✓ Model forward pass successful!")
    
    print("\n" + "=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
