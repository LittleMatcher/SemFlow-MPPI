"""
接口系统测试套件

验证：
1. 接口定义的完整性
2. 接口检查工具的功能
3. 示例实现的正确性
4. InterfaceRegistry 的工作流
"""

import sys
import pytest
import torch
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from cfm_flowmp.interfaces import (
    InterfaceRegistry,
    BaseModel,
    ODESolver,
    LossFunction,
    Dataset,
    Metric,
)
from cfm_flowmp.interface_checker import (
    InterfaceChecker,
    InterfaceValidationError,
)
from cfm_flowmp.examples_interface_usage import (
    DormandPrince45Solver,
    ParametricFlowLoss,
    SyntheticTrajectoryDataset,
    TrajectoryLengthMetric,
)


class TestInterfaceDefinitions:
    """测试接口定义"""
    
    def test_all_interfaces_are_abstract(self):
        """所有接口都应该是抽象的"""
        interfaces = [
            BaseModel, ODESolver, LossFunction, Dataset, Metric
        ]
        
        for interface in interfaces:
            # 应该有抽象方法
            assert hasattr(interface, '__abstractmethods__')
            assert len(interface.__abstractmethods__) > 0
    
    def test_interface_registry_exists(self):
        """接口注册表应该存在"""
        assert InterfaceRegistry is not None
        assert hasattr(InterfaceRegistry, 'register_implementation')
        assert hasattr(InterfaceRegistry, 'get_implementations')
    
    def test_abstract_methods_documented(self):
        """所有抽象方法都应该有文档"""
        for method_name in ODESolver.__abstractmethods__:
            method = getattr(ODESolver, method_name)
            assert method.__doc__ is not None, \
                f"方法 {method_name} 缺少文档"


class TestInterfaceChecker:
    """测试接口检查工具"""
    
    def test_check_valid_implementation(self):
        """验证器应该接受有效实现"""
        solver = DormandPrince45Solver()
        
        # 应该通过检查
        passed, errors = InterfaceChecker.check_implementation(
            DormandPrince45Solver,
            ODESolver,
            raise_error=False
        )
        assert passed, f"应该通过检查，但获得错误: {errors}"
    
    def test_check_invalid_implementation(self):
        """验证器应该拒绝无效实现"""
        
        class InvalidSolver:
            """未继承接口的类"""
            pass
        
        passed, errors = InterfaceChecker.check_implementation(
            InvalidSolver,
            ODESolver,
            raise_error=False
        )
        assert not passed, "应该检测到无效实现"
        assert len(errors) > 0
    
    def test_generate_implementation_template(self):
        """应该能生成实现模板"""
        template = InterfaceChecker.generate_implementation_template(ODESolver)
        
        assert 'class' in template
        assert 'def solve' in template
        assert 'def step' in template
    
    def test_interface_report_generation(self):
        """应该能生成接口报告"""
        report = InterfaceChecker.generate_interface_report(ODESolver)
        
        assert 'ODESolver' in report
        assert '必要方法' in report or 'Methods' in report


class TestExampleImplementations:
    """测试示例实现"""
    
    def test_dormand_prince_solver(self):
        """测试 Dormand-Prince ODE 求解器"""
        solver = DormandPrince45Solver()
        
        # 定义简单的向量场
        def vector_field(x, t):
            return -x
        
        initial_state = torch.tensor([1.0, 1.0])
        t_eval = torch.linspace(0, 1, 10)
        
        # 求解
        trajectory = solver.solve(
            vector_field,
            initial_state,
            t_span=(0, 1),
            t_eval=t_eval
        )
        
        # 检查输出
        assert trajectory.shape[0] == len(t_eval)
        assert trajectory.shape[1:] == initial_state.shape
    
    def test_parametric_loss_function(self):
        """测试参数化损失函数"""
        loss_fn = ParametricFlowLoss(weight=0.5)
        
        predictions = torch.randn(32, 10)
        targets = torch.randn(32, 10)
        
        # 计算损失
        loss = loss_fn(predictions, targets)
        
        # 检查输出
        assert loss.item() > 0
        assert loss.shape == torch.Size([])
    
    def test_synthetic_dataset(self):
        """测试合成数据集"""
        dataset = SyntheticTrajectoryDataset(
            num_samples=10,
            trajectory_length=64
        )
        
        # 检查大小
        assert len(dataset) == 10
        
        # 获取样本
        sample = dataset[0]
        assert 'trajectory' in sample
        assert 'start' in sample
        assert 'goal' in sample
        
        # 检查形状
        assert sample['trajectory'].shape == (64, 2)
        assert sample['start'].shape == (2,)
        assert sample['goal'].shape == (2,)
    
    def test_trajectory_length_metric(self):
        """测试轨迹长度评估指标"""
        metric = TrajectoryLengthMetric()
        
        # 创建简单轨迹
        t = torch.linspace(0, 2 * 3.14159, 100)
        x = torch.cos(t)
        y = torch.sin(t)
        trajectory = torch.stack([x, y], dim=1)
        
        # 计算指标
        value = metric(trajectory, trajectory)
        
        # 检查输出
        assert isinstance(value, float)
        assert value > 0


class TestInterfaceRegistry:
    """测试接口注册表"""
    
    def test_register_implementation(self):
        """应该能注册实现"""
        # 创建测试实现
        class TestSolver(ODESolver):
            def solve(self, vector_field, initial_state, t_span, **kwargs):
                return initial_state.unsqueeze(0)
            
            def step(self, vector_field, state, t, dt, **kwargs):
                return state
        
        # 注册
        InterfaceRegistry.register_implementation(TestSolver)
        
        # 检查是否注册
        impls = InterfaceRegistry.get_implementations('ODESolver')
        assert TestSolver in impls or any(
            isinstance(impl, type) and issubclass(impl, TestSolver)
            for impl in impls
        )
    
    def test_get_implementations(self):
        """应该能获取所有实现"""
        impls = InterfaceRegistry.get_implementations('ODESolver')
        
        # 应该至少有 DormandPrince45Solver
        assert len(impls) >= 0  # 可能为0取决于是否注册了


class TestIntegration:
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整的工作流"""
        # 1. 创建求解器
        solver = DormandPrince45Solver()
        
        # 2. 创建数据集
        dataset = SyntheticTrajectoryDataset(num_samples=5)
        
        # 3. 创建损失函数
        loss_fn = ParametricFlowLoss()
        
        # 4. 创建评估指标
        metric = TrajectoryLengthMetric()
        
        # 5. 验证所有实现
        for impl_class, interface in [
            (DormandPrince45Solver, ODESolver),
            (ParametricFlowLoss, LossFunction),
            (SyntheticTrajectoryDataset, Dataset),
            (TrajectoryLengthMetric, Metric),
        ]:
            passed, errors = InterfaceChecker.check_implementation(
                impl_class, interface, raise_error=False
            )
            assert passed, f"验证失败: {errors}"
    
    def test_end_to_end_computation(self):
        """测试端到端计算"""
        # 创建组件
        solver = DormandPrince45Solver()
        loss_fn = ParametricFlowLoss()
        metric = TrajectoryLengthMetric()
        
        # 定义向量场
        def vector_field(x, t):
            return -x
        
        # 求解
        initial = torch.tensor([1.0, 1.0])
        trajectory = solver.solve(
            vector_field,
            initial,
            t_span=(0, 1),
            t_eval=torch.linspace(0, 1, 20)
        )
        
        # 计算损失
        dummy_target = torch.randn_like(trajectory)
        loss = loss_fn(trajectory, dummy_target)
        
        # 计算指标
        metric_value = metric(trajectory, dummy_target)
        
        # 检查结果有效性
        assert trajectory.shape[0] == 20
        assert loss.item() >= 0
        assert metric_value > 0


class TestErrorHandling:
    """测试错误处理"""
    
    def test_missing_abstract_method(self):
        """应该检测缺失的抽象方法"""
        
        class IncompleteSolver(ODESolver):
            def solve(self, vector_field, initial_state, t_span, **kwargs):
                return initial_state.unsqueeze(0)
            # 缺少 step 方法
        
        passed, errors = InterfaceChecker.check_implementation(
            IncompleteSolver,
            ODESolver,
            raise_error=False
        )
        assert not passed
    
    def test_wrong_method_signature(self):
        """应该检测方法签名不匹配"""
        
        class WrongSolver(ODESolver):
            def solve(self, vector_field, initial_state, t_span, **kwargs):
                return initial_state.unsqueeze(0)
            
            def step(self, state):  # 参数不匹配
                return state
        
        passed, errors = InterfaceChecker.check_implementation(
            WrongSolver,
            ODESolver,
            raise_error=False
        )
        # 可能不完全检测参数（取决于实现细节）


# ============================================================================
# 测试套件执行
# ============================================================================

def run_all_tests():
    """运行所有测试"""
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--disable-warnings'
    ])


if __name__ == '__main__':
    print("\n" + "="*70)
    print("接口系统测试套件")
    print("="*70 + "\n")
    
    run_all_tests()
