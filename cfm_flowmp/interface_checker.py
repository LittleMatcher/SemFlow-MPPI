"""
Interface Checker and Validator

用于检查和验证类是否正确实现了接口。
在添加新方法和类时使用此工具进行检查。

使用示例：
    checker = InterfaceChecker()
    checker.check_implementation(MyClass, BaseInterface)
    checker.generate_implementation_template(BaseInterface)
"""

import inspect
from typing import Type, Dict, List, Tuple, Any
from abc import ABC, abstractmethod
import textwrap

from cfm_flowmp.interfaces import InterfaceRegistry


class InterfaceChecker:
    """接口检查工具"""
    
    @staticmethod
    def check_implementation(
        impl_class: Type,
        interface_class: Type,
        raise_error: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        检查类是否正确实现了接口
        
        Args:
            impl_class: 实现类
            interface_class: 接口类
            raise_error: 检查失败时是否抛出异常
        
        Returns:
            (是否通过检查, 错误列表)
        """
        errors = []
        
        # 检查1：是否继承自接口
        if not issubclass(impl_class, interface_class):
            errors.append(f"{impl_class.__name__} 未继承 {interface_class.__name__}")
        
        # 检查2：抽象方法是否实现
        abstract_methods = getattr(interface_class, '__abstractmethods__', set())
        for method_name in abstract_methods:
            if not hasattr(impl_class, method_name):
                errors.append(f"缺少方法: {method_name}")
            else:
                method = getattr(impl_class, method_name)
                if getattr(method, '__isabstractmethod__', False):
                    errors.append(f"方法 {method_name} 仍为抽象方法")
        
        # 检查3：抽象属性是否实现
        abstract_properties = [
            name for name, value in inspect.getmembers(interface_class)
            if isinstance(value, property) and hasattr(value.fget, '__isabstractmethod__')
        ]
        for prop_name in abstract_properties:
            if not hasattr(impl_class, prop_name):
                errors.append(f"缺少属性: {prop_name}")
        
        # 检查4：方法签名是否匹配
        for method_name in abstract_methods:
            if hasattr(impl_class, method_name):
                interface_method = getattr(interface_class, method_name)
                impl_method = getattr(impl_class, method_name)
                
                # 获取签名
                try:
                    interface_sig = inspect.signature(interface_method)
                    impl_sig = inspect.signature(impl_method)
                    
                    # 检查参数个数（忽略self）
                    interface_params = list(interface_sig.parameters.keys())[1:]
                    impl_params = list(impl_sig.parameters.keys())[1:]
                    
                    if len(interface_params) != len(impl_params):
                        errors.append(
                            f"方法 {method_name} 参数数量不匹配: "
                            f"接口 {len(interface_params)} vs 实现 {len(impl_params)}"
                        )
                except Exception as e:
                    pass  # 某些内置方法可能无法获取签名
        
        passed = len(errors) == 0
        
        if not passed and raise_error:
            raise InterfaceValidationError(
                f"\n{impl_class.__name__} 未正确实现 {interface_class.__name__}:\n" +
                "\n".join(f"  ✗ {error}" for error in errors)
            )
        
        return passed, errors
    
    @staticmethod
    def generate_implementation_template(interface_class: Type) -> str:
        """
        生成接口实现模板
        
        Args:
            interface_class: 接口类
        
        Returns:
            代码模板字符串
        """
        class_name = interface_class.__name__.replace('Base', '').replace('Interface', '')
        if not class_name or class_name[0].islower():
            class_name = 'My' + class_name
        
        template = f"from cfm_flowmp.interfaces import {interface_class.__name__}\n\n"
        template += f"class {class_name}({interface_class.__name__}):\n"
        template += f'    """实现 {interface_class.__name__} 的具体类"""\n\n'
        
        # 添加抽象方法
        abstract_methods = getattr(interface_class, '__abstractmethods__', set())
        for method_name in sorted(abstract_methods):
            method = getattr(interface_class, method_name)
            try:
                sig = inspect.signature(method)
                template += f"    def {method_name}{sig}:\n"
            except Exception:
                template += f"    def {method_name}(self, *args, **kwargs):\n"
            
            # 添加文档字符串
            if method.__doc__:
                doc = textwrap.indent(method.__doc__, "        ")
                template += f'        """{method.__doc__}"""\n'
            
            template += "        # TODO: 实现此方法\n"
            template += "        pass\n\n"
        
        # 添加抽象属性
        for name, value in inspect.getmembers(interface_class):
            if isinstance(value, property) and hasattr(value.fget, '__isabstractmethod__'):
                template += f"    @property\n"
                template += f"    def {name}(self):\n"
                if value.fget.__doc__:
                    template += f'        """{value.fget.__doc__}"""\n'
                template += "        # TODO: 实现此属性\n"
                template += "        pass\n\n"
        
        return template
    
    @staticmethod
    def list_abstract_methods(interface_class: Type) -> Dict[str, str]:
        """
        列出接口的所有抽象方法
        
        Args:
            interface_class: 接口类
        
        Returns:
            {方法名: 文档字符串} 的字典
        """
        methods = {}
        abstract_methods = getattr(interface_class, '__abstractmethods__', set())
        
        for method_name in sorted(abstract_methods):
            method = getattr(interface_class, method_name)
            doc = inspect.getdoc(method) or "无文档"
            methods[method_name] = doc
        
        return methods
    
    @staticmethod
    def list_abstract_properties(interface_class: Type) -> Dict[str, str]:
        """
        列出接口的所有抽象属性
        
        Args:
            interface_class: 接口类
        
        Returns:
            {属性名: 文档字符串} 的字典
        """
        properties = {}
        
        for name, value in inspect.getmembers(interface_class):
            if isinstance(value, property) and hasattr(value.fget, '__isabstractmethod__'):
                doc = inspect.getdoc(value.fget) or "无文档"
                properties[name] = doc
        
        return properties
    
    @staticmethod
    def generate_interface_report(interface_class: Type) -> str:
        """
        生成接口的详细报告
        
        Args:
            interface_class: 接口类
        
        Returns:
            报告字符串
        """
        report = f"\n{'='*60}\n"
        report += f"接口报告: {interface_class.__name__}\n"
        report += f"{'='*60}\n\n"
        
        report += f"模块: {interface_class.__module__}\n"
        report += f"文档:\n{textwrap.indent(interface_class.__doc__ or '无', '  ')}\n\n"
        
        # 抽象方法
        methods = InterfaceChecker.list_abstract_methods(interface_class)
        if methods:
            report += f"【必要方法】(共 {len(methods)} 个)\n"
            for i, (name, doc) in enumerate(methods.items(), 1):
                report += f"\n  {i}. {name}()\n"
                report += textwrap.indent(doc, "     ") + "\n"
        
        # 抽象属性
        properties = InterfaceChecker.list_abstract_properties(interface_class)
        if properties:
            report += f"\n【必要属性】(共 {len(properties)} 个)\n"
            for i, (name, doc) in enumerate(properties.items(), 1):
                report += f"\n  {i}. {name}\n"
                report += textwrap.indent(doc, "     ") + "\n"
        
        # 实现统计
        report += f"\n【已知实现】\n"
        implementations = InterfaceRegistry.get_implementations(interface_class.__name__)
        if implementations:
            for impl_class in implementations:
                passed, errors = InterfaceChecker.check_implementation(
                    impl_class, interface_class, raise_error=False
                )
                status = "✓ 通过" if passed else "✗ 失败"
                report += f"  {status}: {impl_class.__name__}\n"
                if errors:
                    for error in errors:
                        report += f"       - {error}\n"
        else:
            report += "  (暂无已知实现)\n"
        
        report += f"\n{'='*60}\n"
        return report


class InterfaceValidationError(Exception):
    """接口验证错误"""
    pass


# ============ 方便函数 ============

def check_implementation(impl_class: Type, interface_class: Type) -> bool:
    """快速检查实现"""
    passed, _ = InterfaceChecker.check_implementation(
        impl_class, interface_class, raise_error=True
    )
    return passed


def print_interface_report(interface_class: Type) -> None:
    """打印接口报告"""
    print(InterfaceChecker.generate_interface_report(interface_class))


def print_implementation_template(interface_class: Type) -> None:
    """打印实现模板"""
    print(InterfaceChecker.generate_implementation_template(interface_class))


if __name__ == "__main__":
    # 示例用法
    from cfm_flowmp.interfaces import ODESolver, TrajectoryGeneratorBase
    
    print("="*60)
    print("接口检查工具示例")
    print("="*60)
    
    # 生成ODE求解器实现模板
    print("\n【ODE求解器实现模板】")
    print(InterfaceChecker.generate_implementation_template(ODESolver))
    
    # 打印接口报告
    print("\n【接口报告】")
    print_interface_report(ODESolver)
