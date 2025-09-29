#!/usr/bin/env python3
"""
MPS设置测试脚本
快速验证MPS环境是否正确配置，并测试DDPM模型的基本功能
"""

import torch
import sys
import os
from datetime import datetime


def test_pytorch_installation():
    """测试PyTorch安装"""
    print("🔍 测试PyTorch安装")
    print("=" * 50)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch版本是否支持MPS
    pytorch_version = torch.__version__.split('.')
    major, minor = int(pytorch_version[0]), int(pytorch_version[1])
    
    if major > 1 or (major == 1 and minor >= 12):
        print("✅ PyTorch版本支持MPS")
        return True
    else:
        print("❌ PyTorch版本过低，需要1.12.0+")
        return False


def test_mps_availability():
    """测试MPS可用性"""
    print("\n🍎 测试MPS可用性")
    print("=" * 50)
    
    # 检查MPS是否构建
    if torch.backends.mps.is_built():
        print("✅ MPS已构建到PyTorch中")
    else:
        print("❌ MPS未构建到PyTorch中")
        print("💡 请安装支持MPS的PyTorch版本")
        return False
    
    # 检查MPS是否可用
    if torch.backends.mps.is_available():
        print("✅ MPS设备可用")
        print("🚀 可以使用Apple Silicon GPU加速")
        return True
    else:
        print("❌ MPS设备不可用")
        print("💡 需要macOS 12.3+和Apple Silicon (M1/M2)或Metal兼容GPU")
        return False


def test_basic_mps_operations():
    """测试基本MPS操作"""
    print("\n🧪 测试基本MPS操作")
    print("=" * 50)
    
    if not torch.backends.mps.is_available():
        print("⚠️ MPS不可用，跳过测试")
        return False
    
    try:
        device = torch.device("mps")
        
        # 测试张量创建
        print("📝 测试张量创建...")
        x = torch.randn(100, 100, device=device)
        print(f"   ✅ 创建张量: {x.shape} on {x.device}")
        
        # 测试基本运算
        print("🔢 测试基本运算...")
        y = x + x
        print(f"   ✅ 加法运算: {y.shape}")
        
        z = torch.matmul(x, x)
        print(f"   ✅ 矩阵乘法: {z.shape}")
        
        # 测试激活函数
        print("⚡ 测试激活函数...")
        relu_result = torch.relu(x)
        print(f"   ✅ ReLU激活: {relu_result.shape}")
        
        # 测试设备间数据传输
        print("🔄 测试设备间传输...")
        x_cpu = x.to('cpu')
        x_mps_again = x_cpu.to('mps')
        print(f"   ✅ MPS->CPU->MPS: {x_mps_again.device}")
        
        print("🎉 所有基本操作测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ MPS操作测试失败: {str(e)}")
        return False


def test_ddpm_imports():
    """测试DDPM模块导入"""
    print("\n📦 测试DDPM模块导入")
    print("=" * 50)
    
    try:
        print("📝 导入utils模块...")
        from utils import setup_logging, get_data
        print("   ✅ utils模块导入成功")
        
        print("📝 导入modules模块...")
        from modules import UNet_conditional, EMA
        print("   ✅ modules模块导入成功")
        
        print("📝 导入ddpm_conditional模块...")
        from ddpm_conditional import Diffusion, get_optimal_device
        print("   ✅ ddpm_conditional模块导入成功")
        
        print("📝 导入监控模块...")
        from mps_training_monitor import get_monitor
        print("   ✅ 监控模块导入成功")
        
        print("🎉 所有模块导入成功!")
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n🏗️ 测试模型创建")
    print("=" * 50)
    
    try:
        from modules import UNet_conditional
        
        # 测试CPU模型创建
        print("💻 测试CPU模型创建...")
        model_cpu = UNet_conditional(num_classes=10, device="cpu")
        print(f"   ✅ CPU模型创建成功，参数数量: {sum(p.numel() for p in model_cpu.parameters()):,}")
        
        # 测试MPS模型创建（如果可用）
        if torch.backends.mps.is_available():
            print("🚀 测试MPS模型创建...")
            model_mps = UNet_conditional(num_classes=10, device="mps").to("mps")
            print(f"   ✅ MPS模型创建成功，参数数量: {sum(p.numel() for p in model_mps.parameters()):,}")
            
            # 测试前向传播
            print("🔄 测试MPS前向传播...")
            x = torch.randn(1, 3, 64, 64, device="mps")
            t = torch.randint(1, 1000, (1,), device="mps")
            y = torch.randint(0, 10, (1,), device="mps")
            
            with torch.no_grad():
                output = model_mps(x, t, y)
            print(f"   ✅ 前向传播成功，输出形状: {output.shape}")
        
        print("🎉 模型创建和测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {str(e)}")
        return False


def test_device_selection():
    """测试设备选择逻辑"""
    print("\n🎯 测试设备选择逻辑")
    print("=" * 50)
    
    try:
        from ddpm_conditional import get_optimal_device
        
        device_name, device_obj, device_info = get_optimal_device()
        print(f"✅ 选择的设备: {device_name}")
        print(f"📋 设备信息: {device_info}")
        print(f"🔧 设备对象: {device_obj}")
        
        return True
        
    except Exception as e:
        print(f"❌ 设备选择测试失败: {str(e)}")
        return False


def test_data_path():
    """测试数据路径"""
    print("\n📁 测试数据路径")
    print("=" * 50)
    
    data_path = "/Users/ryuichi/Documents/GitHub/DDPM/datasets/cifar10-64/train"
    
    if os.path.exists(data_path):
        print(f"✅ 数据路径存在: {data_path}")
        
        # 检查子目录
        subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        print(f"📂 发现 {len(subdirs)} 个类别目录: {subdirs[:5]}{'...' if len(subdirs) > 5 else ''}")
        
        if len(subdirs) == 10:
            print("✅ CIFAR-10数据集结构正确")
            return True
        else:
            print(f"⚠️ 预期10个类别，实际发现{len(subdirs)}个")
            return False
    else:
        print(f"❌ 数据路径不存在: {data_path}")
        print("💡 请确保已下载并解压CIFAR-10数据集到正确位置")
        return False


def main():
    """主测试函数"""
    print("🧪 DDPM MPS设置测试")
    print("=" * 60)
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("PyTorch安装", test_pytorch_installation),
        ("MPS可用性", test_mps_availability),
        ("基本MPS操作", test_basic_mps_operations),
        ("DDPM模块导入", test_ddpm_imports),
        ("模型创建", test_model_creation),
        ("设备选择", test_device_selection),
        ("数据路径", test_data_path),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试出现异常: {str(e)}")
            results.append((test_name, False))
    
    # 打印测试总结
    print("\n📊 测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} {status}")
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过! 您的MPS环境配置正确，可以开始训练了!")
        print("💡 运行 'python ddpm_conditional.py' 开始训练")
    else:
        print(f"\n⚠️ {total - passed} 个测试失败，请检查配置")
        print("💡 请根据上述错误信息修复问题后重新测试")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
