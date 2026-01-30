#!/usr/bin/env python3
"""
接口验证脚本

用途：验证项目中的所有类是否正确实现了接口

使用方式：
    python check_interfaces.py              # 检查所有实现
    python check_interfaces.py --verbose    # 详细输出
    python check_interfaces.py --report     # 生成详细报告
"""

import sys
import argparse
from typing import Dict, List, Tuple
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from cfm_flowmp.interface_checker import InterfaceChecker, InterfaceValidationError
from cfm_flowmp.interfaces import InterfaceRegistry


def check_all_implementations(verbose: bool = False) -> Tuple[int, int, List[str]]:
    """
    检查所有注册的实现
    
    Returns:
        (通过数, 失败数, 错误列表)
    """
    passed_count = 0
    failed_count = 0
    errors = []
    
    # 从 __init__.py 导入所有接口和实现
    try:
        from cfm_flowmp.interfaces import (
            BaseModel, EmbeddingBase, ConditionalModule,
            ODESolver, TrajectoryGeneratorBase, Smoother,
            DataInterpolator, LossFunction, Trainer,
            Dataset, DataLoader,
            Visualizer, Metric
        )
        interfaces = [
            BaseModel, EmbeddingBase, ConditionalModule,
            ODESolver, TrajectoryGeneratorBase, Smoother,
            DataInterpolator, LossFunction, Trainer,
            Dataset, DataLoader,
            Visualizer, Metric
        ]
    except ImportError as e:
        print(f"✗ 导入接口失败: {e}")
        return 0, 1, [str(e)]
    
    # 检查每个接口的所有已知实现
    for interface in interfaces:
        interface_name = interface.__name__
        implementations = InterfaceRegistry.get_implementations(interface_name)
        
        if not implementations:
            if verbose:
                print(f"ℹ {interface_name}: 暂无已知实现")
            continue
        
        for impl_class in implementations:
            try:
                InterfaceChecker.check_implementation(impl_class, interface, raise_error=True)
                passed_count += 1
                if verbose:
                    print(f"✓ {impl_class.__name__} 正确实现 {interface_name}")
            except InterfaceValidationError as e:
                failed_count += 1
                error_msg = f"✗ {impl_class.__name__} 实现 {interface_name} 失败"
                errors.append(error_msg)
                if verbose:
                    print(error_msg)
                    print(f"  {e}\n")
                else:
                    errors.append(str(e))
    
    return passed_count, failed_count, errors


def print_interface_statistics() -> None:
    """打印接口统计信息"""
    print("\n" + "="*70)
    print("接口统计")
    print("="*70)
    
    try:
        from cfm_flowmp.interfaces import (
            BaseModel, EmbeddingBase, ConditionalModule,
            ODESolver, TrajectoryGeneratorBase, Smoother,
            DataInterpolator, LossFunction, Trainer,
            Dataset, DataLoader,
            Visualizer, Metric
        )
        interfaces = [
            BaseModel, EmbeddingBase, ConditionalModule,
            ODESolver, TrajectoryGeneratorBase, Smoother,
            DataInterpolator, LossFunction, Trainer,
            Dataset, DataLoader,
            Visualizer, Metric
        ]
    except ImportError:
        print("✗ 无法导入接口")
        return
    
    total_interfaces = len(interfaces)
    total_implementations = 0
    total_methods = 0
    
    for interface in interfaces:
        interface_name = interface.__name__
        implementations = InterfaceRegistry.get_implementations(interface_name)
        methods = len(getattr(interface, '__abstractmethods__', set()))
        
        print(f"\n{interface_name}")
        print(f"  - 必要方法: {methods}")
        print(f"  - 已知实现: {len(implementations)}")
        
        total_implementations += len(implementations)
        total_methods += methods
    
    print("\n" + "-"*70)
    print(f"总计: {total_interfaces} 个接口, {total_implementations} 个实现, {total_methods} 个方法")
    print("="*70 + "\n")


def print_detailed_report() -> None:
    """打印详细报告"""
    print("\n" + "="*70)
    print("接口详细报告")
    print("="*70)
    
    try:
        from cfm_flowmp.interfaces import (
            BaseModel, EmbeddingBase, ConditionalModule,
            ODESolver, TrajectoryGeneratorBase, Smoother,
            DataInterpolator, LossFunction, Trainer,
            Dataset, DataLoader,
            Visualizer, Metric
        )
        interfaces = [
            BaseModel, EmbeddingBase, ConditionalModule,
            ODESolver, TrajectoryGeneratorBase, Smoother,
            DataInterpolator, LossFunction, Trainer,
            Dataset, DataLoader,
            Visualizer, Metric
        ]
    except ImportError:
        print("✗ 无法导入接口")
        return
    
    for interface in interfaces:
        print(InterfaceChecker.generate_interface_report(interface))


def main():
    parser = argparse.ArgumentParser(
        description='检查项目中的接口实现',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python check_interfaces.py              # 检查所有实现
  python check_interfaces.py --verbose    # 详细输出
  python check_interfaces.py --report     # 生成详细报告
  python check_interfaces.py --stats      # 显示统计信息
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出检查结果'
    )
    
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='生成详细的接口报告'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='显示接口统计信息'
    )
    
    args = parser.parse_args()
    
    if args.report:
        print_detailed_report()
        return
    
    if args.stats:
        print_interface_statistics()
        return
    
    # 执行检查
    print("\n" + "="*70)
    print("接口验证")
    print("="*70 + "\n")
    
    passed, failed, errors = check_all_implementations(args.verbose)
    
    print("\n" + "="*70)
    print("验证结果")
    print("="*70)
    print(f"✓ 通过: {passed}")
    print(f"✗ 失败: {failed}")
    
    if errors:
        print("\n错误详情:")
        for error in errors:
            print(f"  {error}")
    
    print("="*70 + "\n")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("🎉 所有接口检查通过！\n")
        sys.exit(0)


if __name__ == '__main__':
    main()
