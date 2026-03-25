"""
Quick demo for SemanticConditionedDiffusionPlanner.

This script demonstrates:
1) Building the semantic-conditioned diffusion planner
2) Running one forward diffusion loss pass
3) Sampling K trajectory proposals
"""

from __future__ import annotations

import torch

from cfm_flowmp.models import create_semantic_diffusion_planner


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_semantic_diffusion_planner(
        horizon=64,
        action_dim=2,
        robot_state_dim=6,
        map_channels=1,
        map_latent_dim=256,
        hidden_dim=256,
        diffusion_steps=30,
    ).to(device)

    bsz = 2
    horizon = 64
    action_dim = 2

    robot_state = torch.randn(bsz, 6, device=device)
    local_map = torch.randn(bsz, 1, 64, 64, device=device)
    semantic_cost_map = torch.randn(bsz, 1, 64, 64, device=device)
    goal_state = torch.randn(bsz, 2, device=device)
    goal_room_id = torch.randint(0, 10, (bsz,), device=device)

    # Training objective
    x0 = torch.randn(bsz, horizon, action_dim, device=device)
    loss = model.diffusion_loss(
        x0=x0,
        robot_state=robot_state,
        local_map=local_map,
        semantic_cost_map=semantic_cost_map,
        goal_state=goal_state,
        goal_room_id=goal_room_id,
    )
    print(f"Diffusion loss: {loss.item():.6f}")

    # Inference sampling
    out = model.sample(
        robot_state=robot_state,
        local_map=local_map,
        semantic_cost_map=semantic_cost_map,
        goal_state=goal_state,
        goal_room_id=goal_room_id,
        num_samples=8,
    )

    print("trajectories:", tuple(out["trajectories"].shape))
    print("proposal_scores:", tuple(out["proposal_scores"].shape))
    print("mode_weights_sum:", float(out["mode_weights"].sum().item()))
    print("semantic_tags_example:", out["semantic_tags"][0] if out["semantic_tags"] else "N/A")


if __name__ == "__main__":
    main()
