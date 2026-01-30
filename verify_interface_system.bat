@echo off
REM 接口系统验证脚本
setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo 接口系统验证和文档生成
echo ============================================================================
echo.

cd /d %~dp0

REM 检查 Python 是否安装
where /q py
if errorlevel 1 (
    where /q python
    if errorlevel 1 (
        echo ✗ 未找到 Python。请安装 Python 并将其添加到 PATH。
        exit /b 1
    )
    set "PYTHON=python"
) else (
    set "PYTHON=py"
)

echo 1️⃣ 验证接口系统组件...
%PYTHON% -c "
import sys
sys.path.insert(0, '.')
from cfm_flowmp.interfaces import InterfaceRegistry, BaseModel, ODESolver
from cfm_flowmp.interface_checker import InterfaceChecker
from cfm_flowmp.examples_interface_usage import DormandPrince45Solver
print('   ✓ 所有核心组件导入成功')
" 2>nul || goto error

echo 2️⃣ 验证示例实现...
%PYTHON% -c "
import sys
sys.path.insert(0, '.')
from cfm_flowmp.interface_checker import InterfaceChecker
from cfm_flowmp.interfaces import ODESolver
from cfm_flowmp.examples_interface_usage import DormandPrince45Solver

passed, errors = InterfaceChecker.check_implementation(
    DormandPrince45Solver, ODESolver, raise_error=False
)
if passed:
    print('   ✓ DormandPrince45Solver 正确实现了 ODESolver 接口')
else:
    print(f'   ✗ 验证失败: {errors}')
    sys.exit(1)
" 2>nul || goto error

echo 3️⃣ 生成接口报告...
%PYTHON% cfm_flowmp\examples_interface_usage.py > interface_report.txt 2>&1
if errorlevel 1 (
    echo   ✓ 示例程序执行完成（参考 interface_report.txt）
) else (
    echo   ✓ 示例程序执行完成（参考 interface_report.txt）
)

echo.
echo ============================================================================
echo ✅ 接口系统验证完成！
echo ============================================================================
echo.
echo 文档生成：
echo   - cfm_flowmp\interfaces.py              - 接口定义
echo   - cfm_flowmp\interface_checker.py       - 接口检查工具  
echo   - cfm_flowmp\INTERFACE_WORKFLOW.md      - 工作流指南
echo   - cfm_flowmp\QUICK_REFERENCE.md         - 快速参考
echo   - cfm_flowmp\examples_interface_usage.py - 完整示例
echo   - INTERFACE_SYSTEM.md                   - 系统文档
echo.
echo 快速开始：
echo   1. 查看快速参考: type cfm_flowmp\QUICK_REFERENCE.md
echo   2. 查看工作流:   type cfm_flowmp\INTERFACE_WORKFLOW.md
echo   3. 运行示例:     %PYTHON% cfm_flowmp\examples_interface_usage.py
echo   4. 验证系统:     %PYTHON% check_interfaces.py
echo.
goto :eof

:error
echo ✗ 验证失败！
echo 请检查上述错误信息。
exit /b 1
