#!/usr/bin/env python3
"""
MPS基准测试脚本
用于测试Mac上MPS设备的可用性和性能，并与CPU进行对比
"""

import torch
import time
import sys
from modules import UNet_conditional


def check_mps_availability():
    """检查MPS设备的可用性和配置信息"""
    print("🔍 MPS设备检查")
    print("=" * 50)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Python版本: {sys.version}")
    
    # 检查MPS是否构建
    if torch.backends.mps.is_built():
        print("✅ MPS已构建到PyTorch中")
    else:
        print("❌ MPS未构建到PyTorch中")
        print("💡 请安装支持MPS的PyTorch版本 (1.12.0+)")
        return False
    
    # 检查MPS是否可用
    if torch.backends.mps.is_available():
        print("✅ MPS设备可用")
        print("🍎 检测到Apple Silicon或Metal兼容GPU")
        return True
    else:
        print("❌ MPS设备不可用")
        print("💡 需要macOS 12.3+和Apple Silicon (M1/M2)或Metal兼容GPU")
        return False


def benchmark_basic_operations():
    """基准测试基本张量操作"""
    print("\n🧪 基本操作性能测试")
    print("=" * 50)
    
    # 测试配置
    sizes = [(1000, 1000), (2000, 2000), (3000, 3000)]
    operations = {
        "矩阵乘法": lambda x: torch.matmul(x, x),
        "元素相加": lambda x: x + x,
        "ReLU激活": lambda x: torch.relu(x),
        "卷积操作": lambda x: torch.nn.functional.conv2d(
            x.view(1, 1, x.shape[0], x.shape[1]), 
            torch.randn(1, 1, 3, 3, device=x.device), 
            padding=1
        )
    }
    
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    
    results = {}
    
    for device_name in devices:
        print(f"\n📱 测试设备: {device_name.upper()}")
        device = torch.device(device_name)
        results[device_name] = {}
        
        for size in sizes:
            print(f"  📏 张量大小: {size[0]}x{size[1]}")
            results[device_name][size] = {}
            
            # 创建测试张量
            x = torch.randn(size, device=device)
            
            for op_name, op_func in operations.items():
                try:
                    # 预热
                    for _ in range(3):
                        _ = op_func(x)
                    
                    # 计时
                    start_time = time.time()
                    for _ in range(10):
                        result = op_func(x)
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 10
                    results[device_name][size][op_name] = avg_time
                    print(f"    {op_name}: {avg_time:.4f}s")
                    
                except Exception as e:
                    print(f"    {op_name}: ❌ 失败 ({str(e)[:50]}...)")
                    results[device_name][size][op_name] = None
    
    return results


def benchmark_unet_model():
    """基准测试UNet模型性能"""
    print("\n🏗️ UNet模型性能测试")
    print("=" * 50)
    
    # 测试配置
    batch_sizes = [1, 2, 4]
    image_size = 64
    num_classes = 10
    
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    
    results = {}
    
    for device_name in devices:
        print(f"\n📱 测试设备: {device_name.upper()}")
        device = torch.device(device_name)
        results[device_name] = {}
        
        try:
            # 创建模型
            model = UNet_conditional(num_classes=num_classes, device=device_name).to(device)
            model.eval()
            
            for batch_size in batch_sizes:
                print(f"  📦 批次大小: {batch_size}")
                
                # 创建测试数据
                x = torch.randn(batch_size, 3, image_size, image_size, device=device)
                t = torch.randint(1, 1000, (batch_size,), device=device)
                y = torch.randint(0, num_classes, (batch_size,), device=device)
                
                try:
                    # 预热
                    with torch.no_grad():
                        for _ in range(3):
                            _ = model(x, t, y)
                    
                    # 计时前向传播
                    start_time = time.time()
                    with torch.no_grad():
                        for _ in range(5):
                            output = model(x, t, y)
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 5
                    results[device_name][batch_size] = avg_time
                    
                    print(f"    前向传播: {avg_time:.4f}s")
                    print(f"    输出形状: {output.shape}")
                    print(f"    内存使用: {torch.cuda.memory_allocated() if device_name == 'cuda' else 'N/A'}")
                    
                except Exception as e:
                    print(f"    ❌ 批次大小 {batch_size} 失败: {str(e)[:100]}...")
                    results[device_name][batch_size] = None
                    
        except Exception as e:
            print(f"  ❌ 模型创建失败: {str(e)[:100]}...")
            results[device_name] = None
    
    return results


def print_performance_summary(basic_results, unet_results):
    """打印性能总结"""
    print("\n📊 性能总结")
    print("=" * 50)
    
    if "mps" in basic_results and "cpu" in basic_results:
        print("🚀 MPS vs CPU 基本操作加速比:")
        
        for size in basic_results["cpu"]:
            print(f"\n  📏 张量大小 {size[0]}x{size[1]}:")
            for op_name in basic_results["cpu"][size]:
                cpu_time = basic_results["cpu"][size][op_name]
                mps_time = basic_results["mps"][size][op_name]
                
                if cpu_time and mps_time and cpu_time > 0:
                    speedup = cpu_time / mps_time
                    if speedup > 1:
                        print(f"    {op_name}: {speedup:.2f}x 更快 🚀")
                    else:
                        print(f"    {op_name}: {1/speedup:.2f}x 更慢 🐌")
                else:
                    print(f"    {op_name}: 无法比较")
    
    if "mps" in unet_results and "cpu" in unet_results:
        print("\n🏗️ MPS vs CPU UNet模型加速比:")
        
        for batch_size in unet_results["cpu"]:
            cpu_time = unet_results["cpu"][batch_size]
            mps_time = unet_results["mps"][batch_size]
            
            if cpu_time and mps_time and cpu_time > 0:
                speedup = cpu_time / mps_time
                if speedup > 1:
                    print(f"  批次大小 {batch_size}: {speedup:.2f}x 更快 🚀")
                else:
                    print(f"  批次大小 {batch_size}: {1/speedup:.2f}x 更慢 🐌")


def main():
    """主函数"""
    print("🍎 Mac MPS性能基准测试")
    print("=" * 50)
    
    # 检查MPS可用性
    if not check_mps_availability():
        print("\n❌ MPS不可用，仅进行CPU测试")
    
    # 运行基准测试
    basic_results = benchmark_basic_operations()
    unet_results = benchmark_unet_model()
    
    # 打印总结
    print_performance_summary(basic_results, unet_results)
    
    print("\n✅ 基准测试完成!")
    print("💡 如果MPS显示良好的加速效果，您可以在训练中使用MPS设备")


if __name__ == "__main__":
    main()
