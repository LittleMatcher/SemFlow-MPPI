import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from cfm_flowmp.utils.visualization import visualize_trajectory

root = Path("traj_data") / "cfm_env_hot"   # 换成 "cfm_env_rrt" 也可以
data = np.load(root / "data.npz")

positions = torch.from_numpy(data["positions"])       # [N, T, 2]
velocities = torch.from_numpy(data["velocities"])     # [N, T, 2]
start_states = torch.from_numpy(data["start_states"]) # [N, 6]
goal_states = torch.from_numpy(data["goal_states"])   # [N, 4]
cost_maps = torch.from_numpy(data["cost_maps"])       # [N, 1, H, W]

print("positions:", positions.shape)
print("cost_maps:", cost_maps.shape)

# 取第 idx 条样本
idx = 311
pos = positions[idx]          # [T, 2]
vel = velocities[idx]         # [T, 2]
start_pos = start_states[idx, :2]
goal_pos = goal_states[idx, :2]
cost_map = cost_maps[idx, 0].numpy()  # [H, W]

# 画 cost map 背景 + 轨迹
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(cost_map, origin="lower", cmap="gray", extent=[0, 1, 0, 1])
ax.set_title("Cost map with trajectory")

# 用项目里的可视化函数叠加轨迹
visualize_trajectory(
    positions=pos,
    velocities=vel,
    start_pos=start_pos,
    goal_pos=goal_pos,
    show_velocity=True,
    figsize=(6, 6),
)

plt.show()