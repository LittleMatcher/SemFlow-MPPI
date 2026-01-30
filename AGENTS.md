# AGENTS

This repository contains a PyTorch implementation of the CFM FlowMP
trajectory planning model. Use this document as a quick guide for
automation or agent-based contributions.

## Repository layout

- `cfm_flowmp/` - library code (models, training, inference, data, utils)
- `train.py` - training entrypoint
- `inference.py` - inference entrypoint
- `requirements.txt` - Python dependencies
- `README.md` - full project documentation

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Common commands

Training on synthetic data:

```bash
python train.py --synthetic --num_trajectories 5000 --epochs 50
```

Inference (requires a checkpoint):

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pt \
  --start 0,0 \
  --goal 2,2 \
  --visualize
```

## Tests and linting

There is no automated test or lint setup in this repository. If you add
tests or checks, document the commands here.

## Notes for agents

- Training and inference can be slow and GPU-intensive. Avoid running
  long jobs unless explicitly requested.
- Follow the existing code style; keep changes minimal and focused.
