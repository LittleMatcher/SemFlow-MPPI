# CFM FlowMP: Conditional Flow Matching for Trajectory Planning

A PyTorch implementation of the FlowMP architecture for learning trajectory generation using **Conditional Flow Matching (CFM)**. This framework enables learning smooth, physically consistent trajectories from expert demonstrations.

## Overview

This implementation follows the FlowMP paper's approach for trajectory planning using Conditional Flow Matching:

1. **Network Architecture**: Transformer-based conditional vector field prediction with AdaLN (Adaptive Layer Normalization) conditioning
2. **Training**: Flow matching with interpolated states and velocity field regression
3. **Inference**: RK4 ODE integration from noise to generate trajectories
4. **Post-processing**: B-spline smoothing for physical consistency

### Key Features

- **Transformer Architecture** with Gaussian Fourier time embedding and AdaLN conditioning
- **Joint prediction** of velocity, acceleration, and jerk fields
- **Multiple ODE solvers**: Euler, Midpoint, RK4, and adaptive RK45
- **B-spline smoothing** for physical consistency
- **Mixed precision training** with gradient accumulation
- **EMA (Exponential Moving Average)** weight averaging

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd cfm-flowmp

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
cfm_flowmp/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── embeddings.py      # Time & condition embeddings, AdaLN
│   └── transformer.py     # FlowMP Transformer architecture
├── training/
│   ├── __init__.py
│   ├── flow_matching.py   # Flow interpolation & loss
│   └── trainer.py         # Training loop & utilities
├── inference/
│   ├── __init__.py
│   ├── ode_solver.py      # RK4 and other ODE solvers
│   └── generator.py       # Trajectory generation pipeline
├── data/
│   ├── __init__.py
│   └── dataset.py         # Dataset classes
└── utils/
    ├── __init__.py
    ├── visualization.py   # Plotting utilities
    └── metrics.py         # Evaluation metrics

train.py                   # Training script
inference.py               # Inference script
requirements.txt           # Dependencies
README.md                  # This file
```

## Quick Start

### Training with Synthetic Data

```bash
# Train with synthetic Bezier curve trajectories
python train.py --synthetic --num_trajectories 5000 --epochs 50

# Train with different trajectory types
python train.py --synthetic --trajectory_type polynomial --epochs 100
python train.py --synthetic --trajectory_type sine --epochs 100
```


## Algorithm Details

### Flow Matching Training 

For each training step:

1. **Sample** expert trajectory $(q_1, \dot{q}_1, \ddot{q}_1)$ from dataset
2. **Sample** flow time $t \sim \text{Uniform}(0, 1)$
3. **Sample** noise $\epsilon_q, \epsilon_{\dot{q}}, \epsilon_{\ddot{q}} \sim \mathcal{N}(0, I)$
4. **Interpolate** states (Eq. 6, 8, 10):
   - $q_t = t \cdot q_1 + (1-t) \cdot \epsilon_q$
   - $\dot{q}_t = t \cdot \dot{q}_1 + (1-t) \cdot \epsilon_{\dot{q}}$
   - $\ddot{q}_t = t \cdot \ddot{q}_1 + (1-t) \cdot \epsilon_{\ddot{q}}$
5. **Compute target fields** (as per FlowMP Algorithm 1):
   - $u_{\text{target}} = (q_1 - q_t) / (1-t)$
   - $v_{\text{target}} = (\dot{q}_1 - \dot{q}_t) / (1-t)$
   - $w_{\text{target}} = (\ddot{q}_1 - \ddot{q}_t) / (1-t)$
6. **Forward pass**: Tokenize $\{q_t, \dot{q}_t, \ddot{q}_t\}$, input to Transformer with $t$ and $c$
7. **Compute loss**: $L = \|\hat{u} - u_{\text{target}}\|^2 + \lambda_{\text{acc}}\|\hat{v} - v_{\text{target}}\|^2 + \lambda_{\text{jerk}}\|\hat{w} - w_{\text{target}}\|^2$

### ODE Integration (Inference with 8-Step Schedule)

Following "Unified Generation-Refinement Planning", we use a non-uniform time schedule for faster inference:

1. **Initialize** $x_0 \sim \mathcal{N}(0, I)$
2. **Time Schedule**: $t_{\text{steps}} = [0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]$
   - Large steps early (exploration phase)
   - Small steps near $t=1$ (refinement phase for detail preservation)
3. **Integrate** using RK4 with variable step sizes:
   ```
   for i in range(len(t_steps) - 1):
       t_curr, t_next = t_steps[i], t_steps[i+1]
       dt = t_next - t_curr
       # RK4 update
       k1 = model(x, t_curr, c)
       k2 = model(x + dt/2 * k1, t_curr + dt/2, c)
       k3 = model(x + dt/2 * k2, t_curr + dt/2, c)
       k4 = model(x + dt * k3, t_next, c)
       x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
   ```
4. **Extract** trajectory: $x_1 = (q_{\text{gen}}, \dot{q}_{\text{gen}}, \ddot{q}_{\text{gen}})$
5. **Smooth** with B-splines to eliminate numerical drift

## System Architecture

SemFlow-MPPI是一个三层分层架构，融合语义理解、流匹配轨迹生成和优化控制：

```
                     ┌─────────────────────────┐
                     │   L3: Vision Language   │
                     │   Model (VLM)           │
                     │  ┌─────────────────┐    │
                     │  │ Semantic Scene  │    │
                     │  │ Understanding   │    │
                     │  └────────┬────────┘    │
                     │           │ Cost Map    │
                     └───────────┼──────────────┘
                                 │
                    ┌────────────▼───────────────┐
                    │   L2: Safety-Embedded     │
                    │   Conditional Flow       │
                    │   Matching (CFM)         │
                    │  ┌──────────────────┐    │
                    │  │ Multi-Modal      │    │
                    │  │ Anchor Selection │    │
                    │  │ (K-Means)        │    │
                    │  ├──────────────────┤    │
                    │  │ CBF Constraints  │    │
                    │  │ Safety Guidance  │    │
                    │  └────────┬─────────┘    │
                    │           │ Trajectory  │
                    │           │ Anchors     │
                    └───────────┼──────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │   L1: Model Predictive   │
                    │   Path Integral Control  │
                    │   (MPPI)                 │
                    │  ┌──────────────────┐    │
                    │  │ Stochastic Opt   │    │
                    │  │ Real-Time Local  │    │
                    │  │ Refinement       │    │
                    │  └────────┬─────────┘    │
                    │           │ Final       │
                    │           │ Trajectory │
                    └───────────┼──────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │  Robot Control   │
                        │  Module          │
                        └──────────────────┘
```

### Layer Functions

- **L3 (VLM)**: 解析场景语义，生成代价图 (cost map)
- **L2 (CFM)**: 生成多条候选轨迹锚点，融合安全约束 (CBF)
- **L1 (MPPI)**: 在锚点基础上进行实时随机优化，输出最优执行轨迹

## Model Architecture

### L2 CFM with CBF Safety Constraints

```
┌──────────────────────────────────────────────────────────────┐
│          L2 Safety-Embedded Flow Matching Layer              │
└──────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
    ┌───────┐         ┌──────────┐        ┌──────────┐
    │ Cost  │         │ CFM Flow │        │ Start/  │
    │ Map   │         │ Matching │        │ Goal    │
    │Encoder│         │ Network  │        │Position │
    └───┬───┘         └────┬─────┘        └────┬────┘
        │                  │                    │
        └──────────────────┼────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Sample N   │
                    │  Trajectories
                    │  (Noisy z)  │
                    └──────┬──────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
┌──────────────────┐              ┌──────────────────────┐
│  CBF Guidance    │              │  Multi-Modal Anchor  │
│  Module          │              │  Selector (K-Means) │
│  ┌────────────┐  │              │  ┌────────────────┐ │
│  │ Barrier    │  │              │  │ Clustering     │ │
│  │ Function   │  │              │  │ Features Ext.  │ │
│  │ h(x)       │  │              │  └────────────────┘ │
│  ├────────────┤  │              │  ┌────────────────┐ │
│  │ Safety     │  │              │  │ Representative │ │
│  │ Constraint │  │              │  │ Anchor Select  │ │
│  │ Violation  │  │              │  └────────────────┘ │
│  │ Check      │  │              └──────────────────────┘
│  └────────────┘  │
└────────┬─────────┘
         │
         └─────────┬──────────────┐
                   │              │
                   ▼              ▼
            ┌────────────┐  ┌──────────────┐
            │  Safety    │  │  Anchors     │
            │  Modified  │  │  {τ_i}       │
            │  Trajectory│  │  (Discrete)  │
            └────────────┘  └──────────────┘
                   │              │
                   └──────┬───────┘
                          │
                   ┌──────▼──────────┐
                   │  Output to L1   │
                   │  MPPI Control   │
                   └─────────────────┘
```

**CBF Mathematical Foundation** (安全约束):
- Safety Set Invariance: `C_safe = {x | h(x) ≥ 0}`
- Invariance Condition: `dh/dt + α·h(x) ≥ 0`
- Violation Potential: `V_cbf = ReLU(-(dh/dt + α·h(x)))`
- Vector Field Correction: `v̄ = v + λ·∇S_CBF(x)`


### Transformer Vector Field Prediction

Based on "Unified Generation-Refinement Planning" and FlowMP official implementation.

┌─────────────────────────────────────────┐
│      Trajectory Tokenizer (Linear)      │
│         [B, T, 6] → [B, T, D]           │
│            (D=256 recommended)          │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│    Sinusoidal Positional Encoding       │
│         (sequence positions)            │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│    Time Embedding (Sinusoidal/Fourier)  │
│    t → [B, D], added to token embedding │
└────────────────┬────────────────────────┘
                 │
                 │    ┌───────────────────┐
                 │    │ Condition Encoder │
                 │    │   c → [B, D]      │
                 │    └────────┬──────────┘
                 │             │
                 ▼             ▼
         ┌──────────────────────────┐
         │   Combined Conditioning   │
         │   (AdaLN / Cross-Attn /  │
         │        Token Prepend)     │
         └────────────┬─────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           │
┌───────────────────────────────────┴─────┐
│    Transformer Encoder Blocks (L=8)     │
│  ┌────────────────────────────────────┐ │
│  │     AdaLN (conditioned on t, c)    │ │
│  │              ↓                     │ │
│  │  Multi-Head Self-Attention (H=8)   │ │
│  │              ↓                     │ │
│  │   (Optional) Cross-Attention       │ │
│  │              ↓                     │ │
│  │     AdaLN (conditioned on t, c)    │ │
│  │              ↓                     │ │
│  │     Feed-Forward (GeLU)            │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Final AdaLN                   │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│     Output Head (Unembedding)           │
│       [B, T, D] → [B, T, 6]             │
│   (u: ṗ field, v: p̈ field,            │
│    w: p⃛ field)                         │
└─────────────────────────────────────────┘
```

### Condition Injection Methods

1. **AdaLN** (default): Time and condition modulate LayerNorm parameters
2. **Cross-Attention**: Trajectory tokens attend to condition tokens
3. **Token Prepending**: Condition encoded as prefix tokens


## Data Format

### Expected Input Format

```python
{
    'positions': np.ndarray,      # [N, T, D] position trajectories
    'velocities': np.ndarray,     # [N, T, D] velocity trajectories (optional)
    'accelerations': np.ndarray,  # [N, T, D] acceleration trajectories (optional)
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cfm_flowmp,
  title={CFM FlowMP: Conditional Flow Matching for Trajectory Planning},
  author={},
  year={2024},
  url={https://github.com/your-repo/cfm-flowmp}
}
```
