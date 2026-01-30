# CFM 训练框架（FlowMP 风格）实现方案与代码骨架

本仓库提供一个**可运行的最小实现**，用于复现你描述的 FlowMP/CFM 训练逻辑：

- 模型：条件向量场预测网络 \(v_\theta(x_t, t, c)\)，采用 **Transformer Encoder + AdaLN/FiLM** 条件控制
- 训练：采样 \(t\sim U(0,1)\)，噪声 \(\epsilon\sim \mathcal N(0,I)\)，构造插值状态 \(x_t\) 与目标场 \((u,v,w)_{target}\)，联合 MSE
- 推理：从噪声 \(x_0\) 出发，使用 **RK4** 积分从 \(t=0\) 走到 \(t=1\) 得到轨迹

> 说明：你在描述中引用的 Eq.6/8/10 未给出具体表达式。本实现提供一个 **可插拔的插值模块**，默认使用最常见的 CFM 线性桥（noise→data）：  
> \(x_t=(1-t)\epsilon + t x_1\)，于是 \((x_1-x_t)/(1-t)=x_1-\epsilon\)。如需严格对齐 FlowMP 论文公式，只需替换 `cfm/training/interpolation.py` 中的函数。

## 安装

```bash
python -m pip install -U pip
pip install -e .
pip install -e ".[torch]"
```

## 数据格式（最小约定）

推荐使用 `.npz`：

- `q1`: `(N, T, 2)` 位置 \((x,y)\)
- `dq1`: `(N, T, 2)` 速度 \((\dot x,\dot y)\)
- `ddq1`: `(N, T, 2)` 加速度 \((\ddot x,\ddot y)\)
- `q_start`: `(N, 2)`（可选；默认取 `q1[:,0]`）
- `dq_start`: `(N, 2)`（可选；默认取 `dq1[:,0]`）
- `q_goal`: `(N, 2)`（可选；默认取 `q1[:,-1]`）

## 训练

```bash
cfm-train --config configs/default.yaml --data data/demo.npz
```

## 推理采样（RK4）

```bash
cfm-sample --ckpt runs/exp/latest.pt --out samples.npz --n 8
```

## 目录结构

- `cfm/models/`: Transformer + 时间嵌入 + AdaLN
- `cfm/training/`: 插值、loss、训练循环
- `cfm/inference/`: RK4 与采样脚本
- `configs/`: 默认配置

