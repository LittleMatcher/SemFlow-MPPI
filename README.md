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

### Training with Custom Data

```bash
# Train with your own trajectory data
python train.py --data_path /path/to/trajectories.npz --epochs 100

# With custom model configuration
python train.py \
    --data_path /path/to/data.npz \
    --model_variant large \
    --hidden_dim 512 \
    --num_layers 8 \
    --lr 5e-5 \
    --batch_size 32
```

### Generating Trajectories

```bash
# Generate a single trajectory
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --start 0,0 \
    --goal 2,2 \
    --visualize

# Generate multiple samples
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --start 0,0 \
    --goal 2,2 \
    --num_samples 10 \
    --output generated_trajectories.npz

# With custom ODE solver settings
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --start 0,0 \
    --goal 2,2 \
    --solver rk4 \
    --num_steps 50
```

## API Usage

### Training

```python
import torch
from cfm_flowmp.models import create_flowmp_transformer
from cfm_flowmp.training import CFMTrainer, TrainerConfig, FlowMatchingConfig
from cfm_flowmp.data import SyntheticTrajectoryDataset, create_dataloader

# Create model
model = create_flowmp_transformer(
    variant="base",
    state_dim=2,
    max_seq_len=64,
)

# Create dataset
dataset = SyntheticTrajectoryDataset(
    num_trajectories=5000,
    seq_len=64,
    trajectory_type="bezier",
)

train_loader = create_dataloader(dataset, batch_size=64)

# Setup training
config = TrainerConfig(
    num_epochs=100,
    learning_rate=1e-4,
    device="cuda",
)

trainer = CFMTrainer(model, config, train_loader)
trainer.train()
```

### Inference

```python
import torch
from cfm_flowmp.models import create_flowmp_transformer
from cfm_flowmp.inference import TrajectoryGenerator, GeneratorConfig

# Load trained model
model = create_flowmp_transformer(variant="base")
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create generator
gen_config = GeneratorConfig(
    solver_type="rk4",
    num_steps=20,
    use_bspline_smoothing=True,
)
generator = TrajectoryGenerator(model, gen_config)

# Generate trajectories
start_pos = torch.tensor([[0.0, 0.0]])
goal_pos = torch.tensor([[2.0, 2.0]])

result = generator.generate(
    start_pos=start_pos,
    goal_pos=goal_pos,
    num_samples=5,
)

positions = result['positions']  # [5, 64, 2]
velocities = result['velocities']  # [5, 64, 2]
```

## Algorithm Details

### Flow Matching Training (Algorithm 1)

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

## Model Architecture

Based on "Unified Generation-Refinement Planning" and FlowMP official implementation.

```
Input: x_t [B, T, 6] (position, velocity, acceleration)
       t [B] (flow time ∈ [0,1])
       c (start_pos, goal_pos, start_vel)

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

## Configuration

### Model Variants

| Variant | Hidden Dim | Layers | Heads | Params | Notes |
|---------|-----------|--------|-------|--------|-------|
| small   | 128       | 4      | 4     | ~3M    | Fast prototyping |
| base    | 256       | 8      | 8     | ~15M   | Recommended (Mizuta et al.) |
| large   | 512       | 12     | 16    | ~50M   | High capacity |

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 1e-4 | Adam learning rate |
| weight_decay | 0.01 | L2 regularization |
| warmup_steps | 1000 | LR warmup steps |
| lambda_acc | 1.0 | Acceleration loss weight |
| lambda_jerk | 1.0 | Jerk loss weight |
| condition_type | "adaln" | Condition injection method |

### Inference Time Schedule

| Schedule | Steps | Description |
|----------|-------|-------------|
| 8-step (default) | `[0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]` | Fast inference with refinement |
| uniform-10 | 10 equally spaced | Standard uniform stepping |
| uniform-20 | 20 equally spaced | Higher accuracy |

## Data Format

### Expected Input Format

```python
{
    'positions': np.ndarray,      # [N, T, D] position trajectories
    'velocities': np.ndarray,     # [N, T, D] velocity trajectories (optional)
    'accelerations': np.ndarray,  # [N, T, D] acceleration trajectories (optional)
}
```

If velocities/accelerations are not provided, they will be computed from positions using finite differences.

### Supported File Formats

- `.npy` - Single numpy array (positions only)
- `.npz` - Numpy archive (multiple arrays)
- `.h5`/`.hdf5` - HDF5 files
- `.pkl` - Pickle files

## References

- **FlowMP**: [mkhangg/flow_mp](https://github.com/mkhangg/flow_mp) - Official FlowMP implementation
- **Unified Generation-Refinement Planning**: Non-uniform ODE scheduling for efficient inference
- **Conditional Flow Matching**: [Lipman et al., 2023](https://arxiv.org/abs/2210.02747)
- **Rectified Flow**: [Liu et al., 2022](https://arxiv.org/abs/2209.03003)
- **DiT (Diffusion Transformers)**: [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)
- **Mizuta et al.**: Transformer-based trajectory planning architecture guidelines

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
