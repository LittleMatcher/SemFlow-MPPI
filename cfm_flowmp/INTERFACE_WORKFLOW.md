# 接口定义工作流指南

## 概述

本指南规范了在 CFM FlowMP 项目中添加和实现接口的流程。

**核心原则：**
1. **接口优先** - 在实现前必须定义接口
2. **检查优先** - 写新方法前检查 `interfaces.py`
3. **验证强制** - 新方法完成后必须验证并注册
4. **文档保持同步** - 接口和实现文档必须保持一致

---

## 工作流步骤

### 步骤 1: 需要实现新功能时

**检查清单：**
- [ ] 是否可以扩展现有接口？
- [ ] 是否需要创建新接口？
- [ ] 是否违反现有接口契约？

**操作：**

```bash
# 1. 检查相关接口文件
cat cfm_flowmp/interfaces.py

# 2. 查看接口报告
python -c "
from cfm_flowmp.interface_checker import print_interface_report
from cfm_flowmp.interfaces import YourInterfaceName
print_interface_report(YourInterfaceName)
"
```

### 步骤 2: 如果需要创建新接口

**编辑 `cfm_flowmp/interfaces.py`：**

```python
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

class MyNewInterface(ABC):
    '''
    接口文档：描述这个接口的目的和责任
    
    这个接口定义了...的标准实现
    '''
    
    @abstractmethod
    def do_something(self, param1: str, param2: int) -> bool:
        """
        方法文档：描述这个方法做什么
        
        Args:
            param1: 参数1说明
            param2: 参数2说明
        
        Returns:
            返回值说明
        
        Raises:
            ValueError: 什么情况下抛出
        """
        pass
    
    @property
    @abstractmethod
    def some_property(self) -> str:
        """属性说明"""
        pass
```

**更新 InterfaceRegistry（位于 `interfaces.py` 底部）：**

```python
# 在 __init__.py 中注册
_INTERFACES_REGISTRY.register_interface(MyNewInterface)
```

### 步骤 3: 实现新的接口方法

**第 1 小步：生成模板**

```bash
python -c "
from cfm_flowmp.interface_checker import print_implementation_template
from cfm_flowmp.interfaces import MyNewInterface
print_implementation_template(MyNewInterface)
"
```

**第 2 小步：创建实现文件**

在相应模块中创建类，例如在 `cfm_flowmp/models/my_model.py`：

```python
from cfm_flowmp.interfaces import MyNewInterface

class MyImplementation(MyNewInterface):
    '''具体实现类的文档'''
    
    def do_something(self, param1: str, param2: int) -> bool:
        """完整实现"""
        # 实现代码
        return True
    
    @property
    def some_property(self) -> str:
        """实现属性"""
        return "value"
```

**第 3 小步：在模块初始化文件中注册**

编辑相应的 `__init__.py`：

```python
from .my_model import MyImplementation
from cfm_flowmp.interfaces import InterfaceRegistry

# 注册实现
InterfaceRegistry.register_implementation(MyImplementation)

__all__ = ['MyImplementation']
```

### 步骤 4: 验证接口实现

**操作：**

```bash
# 方式1：在代码中验证
python -c "
from cfm_flowmp.interface_checker import check_implementation
from cfm_flowmp.interfaces import MyNewInterface
from cfm_flowmp.models import MyImplementation

try:
    check_implementation(MyImplementation, MyNewInterface)
    print('✓ 实现正确')
except Exception as e:
    print(f'✗ 验证失败: {e}')
"

# 方式2：生成完整报告
python -c "
from cfm_flowmp.interface_checker import print_interface_report
from cfm_flowmp.interfaces import MyNewInterface
print_interface_report(MyNewInterface)
"
```

**常见错误：**

| 错误 | 原因 | 解决方案 |
|------|------|--------|
| "未继承接口" | 忘记在类定义中继承 | 检查 `class MyImpl(Interface):` |
| "缺少方法" | 没有实现所有抽象方法 | 用模板补齐方法 |
| "仍为抽象方法" | 没有实现方法体 | 删除 `@abstractmethod` 和方法体的 `pass` |
| "参数数量不匹配" | 参数个数与接口定义不同 | 检查接口签名并修正 |

### 步骤 5: 更新接口（如需要）

**何时需要更新接口：**
- 发现接口设计有缺陷
- 添加新的必要功能
- 优化现有方法签名

**更新流程：**

1. **编辑 `cfm_flowmp/interfaces.py`**
   ```python
   # 修改抽象方法或属性
   @abstractmethod
   def new_method(self) -> None:
       """新添加的方法"""
       pass
   ```

2. **更新所有现有实现**
   ```bash
   # 找出所有实现
   grep -r "class.*MyNewInterface" cfm_flowmp/
   
   # 逐个更新实现文件
   ```

3. **验证所有实现**
   ```bash
   python -c "
   from cfm_flowmp.interface_checker import print_interface_report
   from cfm_flowmp.interfaces import MyNewInterface
   print_interface_report(MyNewInterface)
   "
   ```

---

## 文件结构规范

```
cfm_flowmp/
├── interfaces.py              # ← 接口定义（必须维护）
├── interface_checker.py       # ← 接口检查工具
├── models/
│   ├── __init__.py           # ← 注册所有实现
│   ├── my_model.py           # ← 实现示例
│   └── ...
├── inference/
│   ├── __init__.py
│   ├── ode_solver.py
│   └── ...
├── training/
│   ├── __init__.py
│   └── ...
└── ...
```

---

## 快速参考

### 检查是否实现接口

```python
from cfm_flowmp.interface_checker import check_implementation
from cfm_flowmp.interfaces import SomeInterface
from my_module import MyImplementation

check_implementation(MyImplementation, SomeInterface)  # 抛异常if失败
```

### 生成实现模板

```bash
python -c "
from cfm_flowmp.interface_checker import print_implementation_template
from cfm_flowmp.interfaces import ODESolver
print_implementation_template(ODESolver)
"
```

### 生成接口报告

```bash
python -c "
from cfm_flowmp.interface_checker import print_interface_report
from cfm_flowmp.interfaces import TrajectoryGeneratorBase
print_interface_report(TrajectoryGeneratorBase)
"
```

### 列出接口的所有必要方法

```python
from cfm_flowmp.interface_checker import InterfaceChecker
from cfm_flowmp.interfaces import SomeInterface

methods = InterfaceChecker.list_abstract_methods(SomeInterface)
for method_name, doc in methods.items():
    print(f"{method_name}: {doc}")
```

---

## 团队协作规范

### Pull Request 检查清单

在提交 PR 前，请验证：

- [ ] 新类是否继承了正确的接口？
- [ ] 所有抽象方法都已实现？
- [ ] 接口验证是否通过？
  ```bash
  python -c "from cfm_flowmp.interface_checker import check_implementation; ..."
  ```
- [ ] 是否在 `__init__.py` 中注册了实现？
- [ ] 是否更新了 `interfaces.py`（如有新接口）？
- [ ] 文档字符串是否完整？

### Code Review 关注点

Reviewer 应检查：

1. **接口遵循性**
   - 实现类是否继承了接口？
   - 所有方法签名是否匹配？

2. **注册**
   - 是否在相应 `__init__.py` 中注册？
   - InterfaceRegistry 是否能找到实现？

3. **文档**
   - 抽象方法文档是否完整？
   - 参数和返回值是否有类型注释？

---

## 常见问题

**Q: 如果我要修改现有接口，需要更新所有实现吗？**

A: 是的。接口是契约，修改接口意味着所有实现都必须遵守新契约。使用接口报告找出所有实现。

**Q: 我能在一个类中实现多个接口吗？**

A: 可以。Python 支持多继承：
```python
class MyClass(Interface1, Interface2, Interface3):
    # 实现所有接口的方法
    pass
```

**Q: InterfaceRegistry 有什么用？**

A: 它跟踪所有接口及其实现，用于：
- 动态查找实现
- 验证接口遵循性
- 生成接口报告

**Q: 如果我忘记在 `__init__.py` 中注册，会怎样？**

A: InterfaceRegistry 仍然能找到实现（通过反射），但建议显式注册以使意图清晰。

---

## 自动化工具

### 预提交检查脚本

创建 `.git/hooks/pre-commit`：

```bash
#!/bin/bash
# 检查所有新增接口实现

python -c "
from cfm_flowmp.interface_checker import InterfaceChecker
from cfm_flowmp.interfaces import InterfaceRegistry

failed = []
for interface, impls in InterfaceRegistry._implementations.items():
    for impl in impls:
        try:
            InterfaceChecker.check_implementation(impl, interface)
        except Exception as e:
            failed.append(f'{impl.__name__}: {e}')

if failed:
    print('✗ 接口验证失败:')
    for msg in failed:
        print(f'  {msg}')
    exit(1)
else:
    print('✓ 接口验证通过')
    exit(0)
"
```

### 接口验证测试

创建 `tests/test_interfaces.py`：

```python
import pytest
from cfm_flowmp.interface_checker import InterfaceChecker
from cfm_flowmp.interfaces import (
    BaseModel, ODESolver, TrajectoryGeneratorBase, 
    Trainer, Dataset
)

class TestInterfaces:
    """测试所有接口实现"""
    
    def test_all_implementations_valid(self):
        """验证所有实现都符合接口"""
        from cfm_flowmp.interface_checker import InterfaceRegistry
        
        for interface_name, implementations in InterfaceRegistry._implementations.items():
            for impl in implementations:
                # 找到接口类
                interface_cls = globals().get(interface_name)
                if interface_cls:
                    passed, errors = InterfaceChecker.check_implementation(
                        impl, interface_cls, raise_error=False
                    )
                    assert passed, f"{impl.__name__} 实现 {interface_name} 失败:\n" + \
                                   "\n".join(errors)
```

---

## 总结

遵循本工作流可确保：

✓ 接口一致性 - 所有实现都符合契约
✓ 可维护性 - 清晰的接口定义便于维护
✓ 扩展性 - 新功能可以安全地添加
✓ 团队协作 - 统一的标准降低沟通成本
✓ 质量控制 - 自动化验证防止错误

**记住：接口优先，验证强制！**
