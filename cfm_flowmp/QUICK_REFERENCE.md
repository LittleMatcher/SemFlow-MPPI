# 接口系统快速参考

## 5 秒快速开始

### 1. 我想添加一个新方法

**步骤：**

```bash
# 1️⃣ 检查接口
python -c "
from cfm_flowmp.interface_checker import print_interface_report
from cfm_flowmp.interfaces import ModelName  # 替换为实际接口
print_interface_report(ModelName)
"

# 2️⃣ 实现方法
# 在你的类中实现所有抽象方法

# 3️⃣ 验证
python -c "
from cfm_flowmp import check_implementation
from cfm_flowmp.interfaces import ModelName
from your_module import YourClass
check_implementation(YourClass, ModelName)
"
```

---

## 常用命令速查

### 查看接口报告

```bash
python -c "
from cfm_flowmp.interface_checker import print_interface_report
from cfm_flowmp.interfaces import ODESolver
print_interface_report(ODESolver)
"
```

### 生成实现模板

```bash
python -c "
from cfm_flowmp.interface_checker import print_implementation_template
from cfm_flowmp.interfaces import Trainer
print_implementation_template(Trainer)
"
```

### 验证实现

```bash
python -c "
from cfm_flowmp import check_implementation
from cfm_flowmp.interfaces import BaseModel
from your_module import YourModel
check_implementation(YourModel, BaseModel)
"
```

### 运行完整检查

```bash
python check_interfaces.py              # 基本检查
python check_interfaces.py --verbose    # 详细输出
python check_interfaces.py --report     # 详细报告
python check_interfaces.py --stats      # 统计信息
```

---

## 接口速查表

| 接口名 | 用途 | 位置 |
|--------|------|------|
| `BaseModel` | 所有模型的基类 | `cfm_flowmp/models/` |
| `EmbeddingBase` | 嵌入层基类 | `cfm_flowmp/models/` |
| `ConditionalModule` | 条件模块基类 | `cfm_flowmp/models/` |
| `ODESolver` | ODE求解器基类 | `cfm_flowmp/inference/` |
| `TrajectoryGeneratorBase` | 轨迹生成器基类 | `cfm_flowmp/inference/` |
| `Smoother` | 轨迹平滑器基类 | `cfm_flowmp/inference/` |
| `DataInterpolator` | 数据插值器基类 | `cfm_flowmp/training/` |
| `LossFunction` | 损失函数基类 | `cfm_flowmp/training/` |
| `Trainer` | 训练器基类 | `cfm_flowmp/training/` |
| `Dataset` | 数据集基类 | `cfm_flowmp/data/` |
| `DataLoader` | 数据加载器基类 | `cfm_flowmp/data/` |
| `Visualizer` | 可视化工具基类 | `cfm_flowmp/utils/` |
| `Metric` | 评估指标基类 | `cfm_flowmp/utils/` |

---

## 常见场景

### 场景 1: 添加新的 ODE 求解器

```bash
# 1. 查看接口要求
python -c "
from cfm_flowmp.interface_checker import print_interface_report
from cfm_flowmp.interfaces import ODESolver
print_interface_report(ODESolver)
"

# 2. 生成模板
python -c "
from cfm_flowmp.interface_checker import print_implementation_template
from cfm_flowmp.interfaces import ODESolver
print_implementation_template(ODESolver)
" > my_solver.py

# 3. 实现方法（编辑 my_solver.py）

# 4. 集成到项目
# 编辑 cfm_flowmp/inference/__init__.py，添加你的求解器

# 5. 验证
python -c "
from cfm_flowmp import check_implementation
from cfm_flowmp.interfaces import ODESolver
from cfm_flowmp.inference import MySolver
check_implementation(MySolver, ODESolver)
"
```

### 场景 2: 实现新的损失函数

```bash
# 1. 查看接口
python -c "
from cfm_flowmp.interface_checker import print_interface_report
from cfm_flowmp.interfaces import LossFunction
print_interface_report(LossFunction)
"

# 2. 创建类
# class MyLoss(LossFunction):
#     def compute_loss(self, ...): ...
#     def __call__(self, ...): ...

# 3. 验证 & 注册
# - 继承 LossFunction
# - 实现所有抽象方法
# - 在 cfm_flowmp/training/__init__.py 中导入和注册
```

### 场景 3: 修改现有接口

```bash
# ❌ 不推荐！如果必须：

# 1. 编辑 cfm_flowmp/interfaces.py
# 2. 更新所有实现类
# 3. 运行检查
python check_interfaces.py --verbose

# 4. 更新文档和测试
```

---

## 代码示例

### 最小实现示例

```python
from cfm_flowmp.interfaces import ODESolver
import torch

class MyODESolver(ODESolver):
    """我的自定义ODE求解器"""
    
    def solve(
        self,
        vector_field,
        initial_state,
        t_span,
        **kwargs
    ) -> torch.Tensor:
        """求解微分方程"""
        # 你的实现
        return trajectory
    
    def step(
        self,
        vector_field,
        state,
        t,
        dt,
        **kwargs
    ) -> torch.Tensor:
        """单步求解"""
        # 你的实现
        return next_state
```

### 注册实现

编辑 `cfm_flowmp/inference/__init__.py`：

```python
from .my_solver import MyODESolver
from cfm_flowmp.interfaces import InterfaceRegistry

# 自动注册
InterfaceRegistry.register_implementation(MyODESolver)

__all__ = ['MyODESolver']
```

### 验证实现

```python
from cfm_flowmp import check_implementation
from cfm_flowmp.interfaces import ODESolver
from cfm_flowmp.inference import MyODESolver

# 验证
try:
    check_implementation(MyODESolver, ODESolver)
    print("✓ 实现正确")
except Exception as e:
    print(f"✗ 验证失败: {e}")
```

---

## 故障排除

### 问题 1: "未继承接口"

```
错误: MyClass 未继承 ODESolver
解决: 检查类定义 - 必须是 class MyClass(ODESolver):
```

### 问题 2: "缺少方法"

```
错误: 缺少方法: solve
解决: 实现所有抽象方法（通过模板生成）
```

### 问题 3: "仍为抽象方法"

```
错误: 方法 solve 仍为抽象方法
解决: 移除 @abstractmethod 装饰器，提供实现体
```

### 问题 4: "参数数量不匹配"

```
错误: 方法 solve 参数数量不匹配: 接口 3 vs 实现 2
解决: 检查接口定义和实现的参数列表是否一致
```

---

## 工具链集成

### VS Code 集成

在 `.vscode/tasks.json` 中添加：

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "检查接口",
      "type": "shell",
      "command": "python",
      "args": ["check_interfaces.py", "--verbose"],
      "group": "test"
    }
  ]
}
```

### Git Pre-commit Hook

创建 `.git/hooks/pre-commit`：

```bash
#!/bin/bash
python check_interfaces.py || exit 1
```

---

## 需要帮助？

1. **查看接口定义**：`cfm_flowmp/interfaces.py`
2. **查看工作流**：`cfm_flowmp/INTERFACE_WORKFLOW.md`
3. **查看示例**：`cfm_flowmp/interface_checker.py` 的 `__main__` 部分
4. **运行检查**：`python check_interfaces.py --report`

---

## 总结

| 动作 | 命令 |
|------|------|
| 查看接口 | `python -c "from cfm_flowmp.interface_checker import print_interface_report; ..."` |
| 获取模板 | `python -c "from cfm_flowmp.interface_checker import print_implementation_template; ..."` |
| 验证实现 | `python -c "from cfm_flowmp import check_implementation; ..."` |
| 完整检查 | `python check_interfaces.py` |
| 详细报告 | `python check_interfaces.py --report` |

**记住：接口优先！写代码前先看接口。✓**
