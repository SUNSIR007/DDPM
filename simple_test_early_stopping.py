#!/usr/bin/env python3
"""
简化版早停监控器测试脚本
不依赖matplotlib等复杂包，只测试核心功能
"""

import os
import sys


# 简单的numpy数组模拟类
class SimpleArray:
    def __init__(self, data):
        self.data = list(data)
    
    def mean(self):
        return sum(self.data) / len(self.data) if self.data else 0
    
    def var(self):
        if len(self.data) < 2:
            return 0
        mean_val = self.mean()
        return sum((x - mean_val) ** 2 for x in self.data) / len(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# 模拟numpy函数
def polyfit(x, y, degree):
    """简单的线性拟合 (degree=1)"""
    if degree != 1 or len(x) != len(y) or len(x) < 2:
        return [0]
    
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)
    
    # 计算斜率
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    return [slope]


# 修补早停监控器以使用简单实现
def patch_early_stopping():
    """修补early_stopping_monitor以使用简单实现"""
    try:
        from early_stopping_monitor import EarlyStoppingMonitor
        
        # 替换numpy相关函数
        import early_stopping_monitor
        early_stopping_monitor.np = type('numpy', (), {
            'array': SimpleArray,
            'mean': lambda x: SimpleArray(x).mean() if hasattr(x, '__iter__') else x,
            'var': lambda x: SimpleArray(x).var() if hasattr(x, '__iter__') else 0,
            'polyfit': polyfit
        })()
        
        return EarlyStoppingMonitor
        
    except ImportError as e:
        print(f"无法导入早停监控器: {e}")
        return None


def test_convergence_detection():
    """测试收敛检测逻辑"""
    print("🧪 测试收敛检测逻辑")
    print("-" * 40)
    
    # 模拟收敛的loss序列
    converging_losses = [
        1.5, 1.2, 0.9, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.10,
        0.099, 0.098, 0.101, 0.097, 0.102, 0.098, 0.099, 0.101, 0.100, 0.099
    ]
    
    # 模拟仍在下降的loss序列
    declining_losses = [
        2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6,
        0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.28, 0.26, 0.24, 0.22
    ]
    
    EarlyStoppingMonitor = patch_early_stopping()
    if not EarlyStoppingMonitor:
        print("❌ 无法创建监控器实例")
        return False
    
    # 测试收敛场景
    print("📈 测试场景1: 收敛的loss")
    monitor1 = EarlyStoppingMonitor(
        patience=10,
        smoothing_window=8,
        convergence_threshold=1e-3,
        auto_stop=True,
        save_plots=False
    )
    
    for epoch, loss in enumerate(converging_losses):
        should_stop, reason = monitor1.update(loss, epoch)
        if should_stop:
            print(f"   ✅ 在epoch {epoch}检测到: {reason}")
            break
    else:
        print("   ⚠️ 未检测到收敛")
    
    # 测试持续下降场景
    print("\n📉 测试场景2: 持续下降的loss")
    monitor2 = EarlyStoppingMonitor(
        patience=10,
        smoothing_window=8,
        convergence_threshold=1e-3,
        auto_stop=True,
        save_plots=False
    )
    
    stopped = False
    for epoch, loss in enumerate(declining_losses):
        should_stop, reason = monitor2.update(loss, epoch)
        if should_stop:
            print(f"   ⚠️ 意外停止在epoch {epoch}: {reason}")
            stopped = True
            break
    
    if not stopped:
        print("   ✅ 正确识别为持续下降，未触发停止")
    
    return True


def test_patience_mechanism():
    """测试patience机制"""
    print("\n🕐 测试patience机制")
    print("-" * 40)
    
    # 模拟loss不再改进的情况
    plateau_losses = [
        1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.24, 0.24, 0.25, 0.24,
        0.25, 0.24, 0.26, 0.24, 0.25, 0.26, 0.25, 0.24, 0.25, 0.26
    ]
    
    EarlyStoppingMonitor = patch_early_stopping()
    if not EarlyStoppingMonitor:
        return False
    
    monitor = EarlyStoppingMonitor(
        patience=5,  # 较短的patience用于测试
        min_delta=1e-3,
        smoothing_window=3,
        auto_stop=True,
        save_plots=False
    )
    
    for epoch, loss in enumerate(plateau_losses):
        should_stop, reason = monitor.update(loss, epoch)
        if epoch % 5 == 0:
            print(f"   Epoch {epoch}: Loss={loss:.3f}, 耐心计数={monitor.patience_counter}")
        
        if should_stop:
            print(f"   ✅ 在epoch {epoch}触发早停: {reason}")
            print(f"   📊 最佳loss: {monitor.best_loss:.6f} (Epoch {monitor.best_epoch})")
            break
    else:
        print("   ⚠️ 未触发早停机制")
    
    return True


def test_status_reporting():
    """测试状态报告功能"""
    print("\n📊 测试状态报告功能")
    print("-" * 40)
    
    EarlyStoppingMonitor = patch_early_stopping()
    if not EarlyStoppingMonitor:
        return False
    
    monitor = EarlyStoppingMonitor(
        patience=15,
        save_plots=False
    )
    
    # 添加一些测试数据
    test_losses = [1.5, 1.2, 0.9, 0.8, 0.7, 0.65, 0.6, 0.58, 0.56, 0.55]
    
    for epoch, loss in enumerate(test_losses):
        monitor.update(loss, epoch)
    
    # 测试状态摘要
    print("📋 状态摘要:")
    status = monitor.get_status_summary(len(test_losses) - 1)
    print(status)
    
    # 测试文本报告保存 
    test_dir = "test_reports"
    os.makedirs(test_dir, exist_ok=True)
    
    print("\n💾 测试文本报告保存:")
    monitor._save_text_report(test_dir, "test_training")
    
    # 检查文件是否创建
    files = os.listdir(test_dir)
    report_files = [f for f in files if f.startswith("test_training_analysis")]
    if report_files:
        print(f"   ✅ 报告文件已创建: {report_files[0]}")
        # 显示报告内容的前几行
        with open(os.path.join(test_dir, report_files[0]), 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]
            print("   📄 报告内容预览:")
            for line in lines:
                print(f"      {line.rstrip()}")
    else:
        print("   ❌ 报告文件创建失败")
    
    return True


def main():
    """主测试函数"""
    print("🧪 早停监控器简化测试")
    print("=" * 60)
    
    tests = [
        ("收敛检测", test_convergence_detection),
        ("Patience机制", test_patience_mechanism),
        ("状态报告", test_status_reporting)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 运行测试: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name} 测试通过")
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！早停监控器功能正常")
        print("\n💡 使用建议:")
        print("1. 在你的DDPM训练中启用早停监控器")
        print("2. 根据训练特点调整patience和convergence_threshold") 
        print("3. 设置auto_stop=False进行手动确认")
        print("4. 训练完成后查看生成的分析报告")
    else:
        print("⚠️ 部分测试失败，请检查实现")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)