# Warm Start (Predict-Revert-Refine) Strategy

## Alternative: Zero-Shot Warm Start via Flow Inversion

### 更优方案：基于流反转的零样本热启动

#### 核心思想

CFM 学习的是一个确定的常微分方程（ODE）：

$$\frac{dx}{dt} = v_\theta(x, t)$$

这意味着映射 $f: z \to x_1$ 是一个双射（Bijective）。既然我们可以从 $t=0$ 积分到 $t=1$ 生成轨迹，我们当然也可以从 $t=1$ 积分到 $t=0$ 反推噪声。

#### 具体做法

1. **输入**：L1 MPPI 上一帧的最优轨迹 $\mathbf{u}_{t-1}^*$（经过 Shift 操作后作为 $t=1$ 时刻的状态）

2. **反向积分**：调用 ODE 求解器，时间从 $t=1$ 走向 $t=0$：

   $$z^* = x_0 = \mathbf{u}_{t-1}^* + \int_{1}^{0} v_\theta(x_t, t, c) dt$$

3. **输出**：直接得到生成该轨迹所需的精确噪声 $z^*$

#### 优势

- **速度极快**：只需要做 1 次 ODE 求解（且不需要计算梯度），计算量等同于一次常规生成
- **数学精确**：在数值误差范围内，这个 $z^*$ 是能够重构出输入轨迹的"完美潜变量"
- **流形约束**：如果输入的轨迹完全不符合动力学（Off-manifold），反向积分会将其"投影"回 $z$ 空间。当我们用这个 $z$ 正向生成时，得到的轨迹会自动被"修复"为符合 L2 动力学的轨迹

---

## Overview

The Warm Start mechanism implements a **Predict-Revert-Refine** strategy to accelerate L2 trajectory generation and ensure temporal consistency in the SemFlow-MPPI framework. This approach combines principles from:

- **SDEdit** (Meng et al., 2021): Image editing by denoising diffusion models at intermediate timesteps
- **MPC** (Model Predictive Control): Temporal consistency through trajectory shifting
- **On-Policy RL**: Policy continuation across timesteps

## Theoretical Background

### Problem Statement

In closed-loop control scenarios, we need to replan trajectories at every timestep. Standard Conditional Flow Matching (CFM) generates trajectories by solving the ODE from `t=0` (pure Gaussian noise) to `t=1` (data). This is computationally expensive and ignores temporal structure.

### Solution: Warm Start

Instead of generating from pure noise at every step, we leverage the optimal trajectory from the previous timestep as a strong prior:

1. **Predict (Shift)**: We assume the optimal control sequence from timestep `t-1` is a good estimator for timestep `t`. We shift it forward in time:

   ```
   u_pred = Shift(u*_{t-1})
   ```

2. **Revert (Noise Injection)**: Instead of starting generation from `t=0` (pure Gaussian), we construct a latent state at intermediate time `τ_warm` (e.g., 0.8) using the **Optimal Transport (OT)** interpolation formula:

   ```
   z_τ = τ_warm · u_pred + (1 - τ_warm) · ε,  where ε ~ N(0, I)
   ```

   This "reverts" the trajectory to a noisy state, allowing the model to correct errors (SDEdit principle).

3. **Refine (Partial Integration)**: We solve the CFM ODE only from `τ_warm` to `1.0`:

   ```
   u_new = z_τ + ∫_{τ_warm}^{1} v_θ(z_s, s, c) ds
   ```

   This reduces computational cost by **~80%** (from 20 steps to 4-5 steps) and ensures the trajectory stays close to the previous solution.

## Mathematical Formulation

### OT Interpolation Path

The Conditional Flow Matching model is trained using the Optimal Transport path:

```
x_t = t · x_1 + (1 - t) · x_0
```

where:
- `x_0 ~ N(0, I)` is the source distribution (Gaussian noise)
- `x_1` is the target distribution (data)
- `t ∈ [0, 1]` is the flow time

### Warm Start Latent Construction

Given the previous optimal trajectory `u*_{t-1}`, we construct the warm start latent at time `τ_warm`:

```
1. u_shift = Shift(u*_{t-1})              # Predict: shift forward
2. ε ~ N(0, I)                             # Sample noise
3. z_τ = τ_warm · u_shift + (1-τ) · ε     # Revert: OT interpolation
```

This ensures that `z_τ` lies on the OT path between the shifted prior and noise, maintaining consistency with the model's training objective.

### Partial ODE Integration

Instead of solving from `t=0` to `t=1`:

```
x_1 = x_0 + ∫_0^1 v_θ(x_t, t, c) dt     # Standard CFM: 20 steps
```

We solve from `t=τ_warm` to `t=1`:

```
x_1 = z_τ + ∫_{τ_warm}^1 v_θ(x_t, t, c) dt     # Warm Start: 4-5 steps
```

## Implementation

### L1 Controller Methods

#### `shift_trajectory(trajectory: Tensor) -> Tensor`

Shifts the trajectory forward by one timestep (Predict step).

**Input:**
- `trajectory`: `[T, D]` or `[B, T, D]` - Previous optimal trajectory

**Output:**
- `shifted_traj`: Same shape - Shifted trajectory

**Padding modes:**
- `"zero_pad"`: Append zeros (deceleration/stop)
- `"extrapolate"`: Linear extrapolation from last two points (constant velocity)
- `"repeat_last"`: Repeat last state

**Example:**
```python
# Previous optimal trajectory [64, 6] (pos + vel + acc)
prev_optimal = mppi_result['optimal_control']

# Shift forward
shifted = l1_controller.shift_trajectory(prev_optimal)
# Shape: [64, 6]
```

#### `prepare_warm_start_latent(prev_opt_traj: Tensor, tau_warm: float) -> Tensor`

Prepares the warm start latent state using OT interpolation (Revert step).

**Input:**
- `prev_opt_traj`: `[T, D]` or `[B, T, D]` - Previous optimal trajectory
  - If `D = state_dim` (positions only), automatically computes velocities and accelerations
  - If `D = state_dim * 3` (full state), uses directly
- `tau_warm`: Float in `[0, 1]` - OT interpolation parameter
  - `0.0`: Pure noise (standard CFM, no warm start benefit)
  - `1.0`: Fully deterministic (no exploration, may get stuck)
  - `0.8`: Recommended (80% prior, 20% noise)

**Output:**
- `z_tau`: `[T, D*3]` or `[B, T, D*3]` - Warm start latent state

**Example:**
```python
# Prepare warm start latent at τ=0.8
z_tau = l1_controller.prepare_warm_start_latent(
    prev_opt_traj=prev_optimal,
    tau_warm=0.8,
)
# Shape: [64, 6] -> latent state at t=0.8
```

### L2 Generator Methods

#### `generate_with_warm_start(condition: Dict, warm_start_latent: Tensor, t_start: float, steps: int) -> Dict`

Generates trajectories starting from intermediate time `t_start` (Refine step).

**Input:**
- `condition`: Dictionary containing:
  - `'start_pos'`: `[B, D]` - Starting position
  - `'goal_pos'`: `[B, D]` - Goal position
  - `'start_vel'`: `[B, D]` (optional) - Starting velocity
  - `'goal_vel'`: `[B, D]` (optional) - Goal velocity (for L2 layer)
  - `'env_encoding'`: `[B, env_dim]` (optional) - Environment encoding
- `warm_start_latent`: `[B, T, D*3]` - Warm start latent from L1
- `t_start`: Float in `[0, 1)` - Starting time for ODE integration (e.g., 0.8)
- `steps`: Int - Number of ODE steps from `t_start` to `1.0` (e.g., 5)
- `cost_map`: `[B, H, W]` (optional) - Semantic cost map for CBF safety filtering

**Output:**
- Dictionary containing:
  - `'positions'`: `[B, T, D]` - Generated position trajectories
  - `'velocities'`: `[B, T, D]` - Generated velocity trajectories
  - `'accelerations'`: `[B, T, D]` - Generated acceleration trajectories
  - `'raw_output'`: `[B, T, D*3]` - Raw model output
  - `'t_start'`: Float - Starting time (metadata)
  - `'steps'`: Int - Number of steps (metadata)
  - `'warm_start'`: Bool - Always `True` (metadata)

**Example:**
```python
# Generate with warm start
result = generator.generate_with_warm_start(
    condition={
        'start_pos': start_pos,
        'goal_pos': goal_pos,
    },
    warm_start_latent=z_tau,
    t_start=0.8,
    steps=5,
)
# Output: trajectories generated from t=0.8 to t=1.0
```

## Usage in Control Loop

### Complete Example

```python
import torch
from cfm_flowmp.inference.generator import TrajectoryGenerator, GeneratorConfig
from cfm_flowmp.inference.l1_reactive_control import L1ReactiveController, L1Config

# Setup
device = torch.device('cuda')
generator = TrajectoryGenerator(model, GeneratorConfig())
l1_controller = L1ReactiveController(L1Config(use_warm_start=True))

# Initialize
prev_optimal_traj = None
tau_warm = 0.8
refine_steps = 5

# Control loop
for timestep in range(num_timesteps):
    # Update conditions
    start_pos = get_current_position()
    goal_pos = get_goal_position()
    
    # ============ L2: Trajectory Generation ============
    if timestep == 0:
        # First timestep: generate from scratch
        l2_output = generator.generate(
            start_pos=start_pos,
            goal_pos=goal_pos,
            return_raw=True,
        )
    else:
        # Subsequent timesteps: use warm start
        
        # Step 1 & 2: Prepare warm start latent (L1)
        z_tau = l1_controller.prepare_warm_start_latent(
            prev_opt_traj=prev_optimal_traj,
            tau_warm=tau_warm,
        )
        
        # Step 3: Generate with warm start (L2)
        l2_output = generator.generate_with_warm_start(
            condition={
                'start_pos': start_pos,
                'goal_pos': goal_pos,
            },
            warm_start_latent=z_tau,
            t_start=tau_warm,
            steps=refine_steps,
        )
    
    # ============ L1: MPPI Optimization ============
    l1_controller.initialize_from_l2_output(l2_output)
    mppi_result = l1_controller.optimize(n_iterations=10)
    optimal_control = mppi_result['optimal_control']
    
    # ============ Update for Next Iteration ============
    prev_optimal_traj = optimal_control  # Store for next timestep
    
    # Execute first control action
    execute_control(optimal_control[0])
```

### Key Points

1. **First Timestep**: Always generate from scratch (no prior trajectory available)
2. **Subsequent Timesteps**: Use warm start for acceleration
3. **Storage**: Save the optimal trajectory at each timestep
4. **Batch Compatibility**: All methods support batched inputs

## Performance Characteristics

### Computational Speedup

| Method | ODE Steps | Time Range | Relative Time |
|--------|-----------|------------|---------------|
| Standard Generation | 20 | t=0 → t=1 | 1.0x (baseline) |
| Warm Start (τ=0.8) | 5 | t=0.8 → t=1 | 0.25x (~4x speedup) |
| Warm Start (τ=0.9) | 3 | t=0.9 → t=1 | 0.15x (~6.7x speedup) |

### Trade-offs

**τ_warm Selection:**

- **τ = 0.5-0.7**: More exploration, better error correction, slower (~50% of original time)
- **τ = 0.8**: **Recommended** - Good balance of speed and quality
- **τ = 0.9-0.95**: Maximum speed, but less exploration (may miss better solutions)

**Number of Refine Steps:**

- **3-5 steps**: Fast, suitable for real-time control (>30 Hz)
- **5-10 steps**: Standard, good accuracy-speed trade-off
- **10-15 steps**: High accuracy, for critical applications

## Benefits

1. **Computational Efficiency**: Reduces ODE integration steps by 75-80%
2. **Temporal Consistency**: Trajectories evolve smoothly across timesteps
3. **Error Correction**: Noise injection allows model to fix prior mistakes
4. **On-Policy Behavior**: Similar to on-policy RL, decisions build on previous decisions
5. **MPC Compatibility**: Natural fit for Model Predictive Control frameworks

## Mathematical Guarantees

### Consistency with CFM Training

The OT interpolation formula:
```
z_τ = τ · u_shift + (1 - τ) · ε
```

is **exactly** the same interpolation used during CFM training. This ensures that `z_τ` is a valid point on the OT path, and the model's velocity field `v_θ(z_τ, τ, c)` is well-defined.

### Convergence

As long as:
1. The environment doesn't change drastically between timesteps
2. `tau_warm` is chosen appropriately (not too close to 1.0)
3. The shifted trajectory is a reasonable prior

The warm start method will converge to high-quality trajectories comparable to full generation.

## Advanced Usage

### Dynamic τ_warm Adjustment

Adapt `tau_warm` based on environment dynamics:

```python
# More stable environment -> use larger τ (faster)
if environment_is_stable:
    tau_warm = 0.9
    refine_steps = 3
else:
    # More dynamic -> use smaller τ (more exploration)
    tau_warm = 0.7
    refine_steps = 8
```

### Multi-Timestep Warm Start

For very fast replanning, maintain a sliding window:

```python
# Keep last N optimal trajectories
trajectory_history = []

# Use weighted combination
z_tau = 0.6 * shifted_traj_t1 + 0.3 * shifted_traj_t2 + 0.1 * noise
```

### Safety-Critical Applications

Combine with CBF safety filtering:

```python
result = generator.generate_with_warm_start(
    condition={...},
    warm_start_latent=z_tau,
    t_start=0.8,
    steps=5,
    cost_map=semantic_cost_map,  # Enable CBF filtering
)
```

## Troubleshooting

### Issue: Trajectories diverge over time

**Cause**: `tau_warm` too close to 1.0, insufficient exploration

**Solution**: Reduce `tau_warm` to 0.7-0.8, increase refine steps

### Issue: Not seeing expected speedup

**Cause**: Bottleneck in other components (MPPI, smoothing)

**Solution**: Profile code, disable smoothing for benchmarking

### Issue: First control action jitters

**Cause**: Large shift between timesteps, warm start assumptions violated

**Solution**: 
- Reduce control timestep (increase control frequency)
- Use smaller `tau_warm` (more exploration)
- Add trajectory smoothing post-processing

## References

1. **SDEdit**: Meng et al., "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations", ICLR 2022
2. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
3. **Optimal Transport**: Villani, "Optimal Transport: Old and New", Springer 2009
4. **MPC**: Camacho & Alba, "Model Predictive Control", Springer 2013

## Citation

If you use this Warm Start implementation, please cite:

```bibtex
@article{semflow-mppi,
  title={SemFlow-MPPI: Hierarchical Robot Navigation with Conditional Flow Matching},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## See Also

- [`example_warm_start_usage.py`](./example_warm_start_usage.py) - Complete working examples
- [`L1_README.md`](./L1_README.md) - L1 MPPI controller documentation
- [`generator.py`](./generator.py) - L2 generator implementation
- [`l1_reactive_control.py`](./l1_reactive_control.py) - L1 controller implementation
