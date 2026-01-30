# Data Pipeline (CFM FlowMP)

This document summarizes how trajectory data flows through the training
pipeline. It mirrors the behavior in `cfm_flowmp/data/dataset.py` and
the training scripts.

## 1. Data sources

There are two entry points:

- **Synthetic data** via `SyntheticTrajectoryDataset`
- **On-disk data** via `TrajectoryDataset`

### Supported file formats (TrajectoryDataset)

`TrajectoryDataset` loads trajectories from:

- `.npy` (single array, positions only)
- `.npz` (archive with multiple arrays)
- `.h5` / `.hdf5`
- `.pkl`

## 2. Expected arrays

All arrays are shaped `[N, T, D]`:

- `positions`: required
- `velocities`: optional (computed if missing)
- `accelerations`: optional (computed if missing)

If velocities or accelerations are omitted, they are computed via finite
differences using the provided `dt` (default `0.1`).

## 3. Derivative computation

`TrajectoryDataset` computes:

- velocity via central differences (endpoints use forward/backward)
- acceleration by applying the same finite difference to velocity

These computations are performed on the full batch of trajectories.

## 4. Normalization

If `normalize=True` (default), mean and standard deviation are computed
per component across `[N, T]` for positions, velocities, and
accelerations. Each sample is normalized on access in `__getitem__`.

Normalization stats can be retrieved with:

```python
stats = dataset.get_normalization_stats()
```

Use these stats during inference if you want to normalize inputs in the
same way as training.

## 5. Sample schema

`__getitem__` returns a dictionary with:

```
{
  "positions": [T, D],
  "velocities": [T, D],
  "accelerations": [T, D],
  "start_pos": [D],
  "goal_pos": [D],
  "start_vel": [D],
}
```

`start_pos` and `goal_pos` are derived from the first and last position
in the sequence. `start_vel` is the first velocity element.

## 6. DataLoader and splits

Utility helpers live in `cfm_flowmp/data/dataset.py`:

- `create_dataloader(dataset, batch_size, shuffle, num_workers, ...)`
- `split_dataset(dataset, train_ratio, val_ratio, test_ratio, seed)`

The training script (`train.py`) performs a 90/10 split and creates
train/val loaders using `create_dataloader`.

## 7. Minimal usage example

```python
from cfm_flowmp.data import TrajectoryDataset, create_dataloader

dataset = TrajectoryDataset(
    data_path="data/trajectories.npz",
    normalize=True,
    compute_derivatives=True,
    dt=0.1,
)

train_loader = create_dataloader(dataset, batch_size=64, num_workers=4)
```

## 8. Notes

- Data loading can be CPU-bound; adjust `num_workers` based on your
  machine.
- Large datasets should be stored in formats that support memory
  mapping or efficient chunking (HDF5 is recommended).
