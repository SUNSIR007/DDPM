#!/usr/bin/env python3
"""
测试早停监控器功能
模拟训练过程并演示监控器的各种功能
"""

import numpy as np
import matplotlib.pyplot as plt
from early_stopping_monitor import EarlyStoppingMonitor
import os

def test_early_stopping_monitor():
    """测试早停监控器的功能"""
    print("🧪 早停监控器功能测试")
    print("=" * 60)
    
    # 创建测试目录
    test_dir = "test_early_stopping_results"
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建监控器实例
    monitor = EarlyStoppingMonitor(
        patience=15,
        min_delta=1e-5,
        smoothing_window=8,
        convergence_threshold=1e-4,
        auto_stop=False,
        save_plots=True
    )
    
    # 模拟不同类型的训练损失曲线
    test_scenarios = {
        "快速收敛": generate_converging_loss(),
        "震荡但下降": generate_oscillating_loss(),  
        "过拟合": generate_overfitting_loss(),
        "平缓收敛": generate_slow_converging_loss()
    }
    
    for scenario_name, losses in test_scenarios.items():
        print(f"\n📊 测试场景: {scenario_name}")
        print("-" * 40)
        
        # 重置监控器
        monitor = EarlyStoppingMonitor(
            patience=15,
            min_delta=1e-5,
            smoothing_window=8,
            convergence_threshold=1e-4,
            auto_stop=True,  # 自动停止以便测试
            save_plots=True
        )
        
        # 模拟训练过程
        for epoch, loss in enumerate(losses):
            should_stop, reason = monitor.update(loss, epoch)
            
            if epoch % 20 == 0 or should_stop:
                print(f"  Epoch {epoch:3d}: Loss = {loss:.6f}")
                if should_stop:
                    print(f"  🛑 停止原因: {reason}")
                    break
        
        # 保存分析图表
        monitor.save_loss_plot(test_dir, f"test_{scenario_name.replace(' ', '_')}")
        print(f"  ✅ 场景 '{scenario_name}' 测试完成")
    
    print(f"\n🎉 所有测试完成！结果保存在 {test_dir}/ 目录中")
    return test_dir

def generate_converging_loss(epochs=100):
    """生成快速收敛的loss曲线"""
    np.random.seed(42)
    losses = []
    for i in range(epochs):
        if i < 30:
            # 快速下降阶段
            loss = 2.0 * np.exp(-i/8) + 0.05 + np.random.normal(0, 0.02)
        else:
            # 收敛阶段
            loss = 0.05 + np.random.normal(0, 0.005)
        losses.append(max(0.01, loss))
    return losses

def generate_oscillating_loss(epochs=150):
    """生成震荡但整体下降的loss曲线"""
    np.random.seed(123)
    losses = []
    for i in range(epochs):
        # 整体下降趋势 + 周期性震荡
        trend = 1.5 * np.exp(-i/40) + 0.1
        oscillation = 0.1 * np.sin(i * 0.3) * np.exp(-i/60)
        noise = np.random.normal(0, 0.02)
        loss = trend + oscillation + noise
        losses.append(max(0.05, loss))
    return losses

def generate_overfitting_loss(epochs=120):
    """生成过拟合的loss曲线（先下降后上升）"""
    np.random.seed(456)
    losses = []
    for i in range(epochs):
        if i < 60:
            # 下降阶段
            loss = 1.8 * np.exp(-i/15) + 0.2 + np.random.normal(0, 0.03)
        else:
            # 过拟合阶段（缓慢上升）
            loss = 0.2 + (i - 60) * 0.002 + np.random.normal(0, 0.02)
        losses.append(max(0.1, loss))
    return losses

def generate_slow_converging_loss(epochs=200):
    """生成缓慢收敛的loss曲线"""
    np.random.seed(789)
    losses = []
    for i in range(epochs):
        if i < 50:
            # 初期快速下降
            loss = 3.0 * np.exp(-i/12) + 0.5
        elif i < 120:
            # 中期缓慢下降
            loss = 0.8 * np.exp(-(i-50)/30) + 0.2
        else:
            # 后期非常缓慢的改进
            loss = 0.2 - (i - 120) * 0.0005 + np.random.normal(0, 0.01)
        
        losses.append(max(0.05, loss + np.random.normal(0, 0.01)))
    return losses

def analyze_training_results(test_dir):
    """分析测试结果"""
    print(f"\n🔍 训练结果分析")
    print("=" * 60)
    
    # 检查生成的文件
    files = os.listdir(test_dir)
    plot_files = [f for f in files if f.endswith('.png')]
    
    print(f"📊 生成的分析图表: {len(plot_files)} 个")
    for plot_file in plot_files:
        print(f"  - {plot_file}")
    
    print("\n💡 使用建议:")
    print("1. 📈 查看生成的loss曲线图来理解不同训练场景")
    print("2. 🎯 观察收敛检测点和改进点的标记")
    print("3. ⚙️ 根据你的具体需求调整监控器参数:")
    print("   - patience: 控制早停的敏感度")
    print("   - min_delta: 设置最小有效改进幅度")
    print("   - smoothing_window: 调整loss平滑程度")
    print("   - convergence_threshold: 调整收敛检测的严格度")

def create_usage_guide():
    """创建使用指南"""
    guide = """
# 📖 早停监控器使用指南

## 🚀 快速开始

### 1. 基本使用
```python
from early_stopping_monitor import EarlyStoppingMonitor

# 创建监控器
early_stopping = EarlyStoppingMonitor(
    patience=20,        # 20个epoch无改进就提醒
    min_delta=1e-6,     # 最小改进阈值
    auto_stop=False     # 手动确认停止
)

# 在训练循环中使用
for epoch in range(max_epochs):
    # ... 训练代码 ...
    avg_loss = calculate_epoch_loss()
    
    should_stop, reason = early_stopping.update(avg_loss, epoch)
    if should_stop:
        print(f"训练停止: {reason}")
        break
```

### 2. 参数说明

- **patience**: 容忍多少个epoch没有改进就提醒停止
- **min_delta**: 多小的改进算作"有效改进"
- **smoothing_window**: 用多少个epoch来平滑loss曲线
- **convergence_threshold**: 收敛检测的敏感度
- **auto_stop**: 是否自动停止（False更安全）
- **save_plots**: 是否保存分析图表

### 3. 收敛检测原理

监控器通过以下方式检测收敛:
1. 计算最近几个epoch的loss方差
2. 分析loss变化趋势
3. 当方差和趋势都很小时，判断为收敛

### 4. 实际应用建议

**对于DDPM训练:**
- patience = 20-30 (扩散模型收敛较慢)
- min_delta = 1e-6 (loss变化通常较小)
- smoothing_window = 10-15
- auto_stop = False (手动确认更安全)

**对于其他模型:**
- 根据训练速度调整patience
- 根据loss量级调整min_delta
- 较短的训练可以用较小的smoothing_window

### 5. 保存的文件

- `*_converged_*.pt`: 收敛时的模型
- `*_early_stopped_*.pt`: 早停时的模型  
- `*_best_model.pt`: 最佳性能模型
- `*_loss_analysis_*.png`: loss分析图表

### 6. 故障排除

**Q: 监控器提醒收敛但loss还在下降怎么办?**
A: 调小convergence_threshold或增大smoothing_window

**Q: 一直不收敛怎么办?**
A: 检查min_delta设置是否过小，或者模型确实需要更长训练

**Q: 误报过多怎么办?**
A: 增大patience值，或者设置auto_stop=False手动确认
"""
    
    with open("early_stopping_usage_guide.md", "w", encoding='utf-8') as f:
        f.write(guide)
    
    print("📚 使用指南已保存到 early_stopping_usage_guide.md")

if __name__ == "__main__":
    # 运行测试
    test_dir = test_early_stopping_monitor()
    
    # 分析结果
    analyze_training_results(test_dir)
    
    # 创建使用指南
    create_usage_guide()
    
    print("\n" + "="*60)
    print("🎯 测试总结:")
    print("✅ 早停监控器功能测试完成")
    print("📊 分析图表已生成")
    print("📚 使用指南已创建")
    print("\n💡 你现在可以:")
    print("1. 查看生成的图表了解不同场景的表现")
    print("2. 阅读使用指南了解参数调整")
    print("3. 在实际训练中应用早停监控器")