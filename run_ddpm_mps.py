#!/usr/bin/env python3
"""
DDPM MPS训练启动脚本
提供便捷的命令行接口来启动MPS加速的DDPM训练
"""

import argparse
import sys
import os
from datetime import datetime


def print_banner():
    """打印启动横幅"""
    print("🚀 DDPM Conditional Training with MPS Acceleration")
    print("=" * 60)
    print("🍎 Apple Silicon GPU加速的扩散模型训练")
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def check_prerequisites():
    """检查运行前提条件"""
    print("🔍 检查运行环境...")
    
    # 检查必要文件
    required_files = [
        "ddpm_conditional.py",
        "modules.py", 
        "utils.py",
        "mps_training_monitor.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {', '.join(missing_files)}")
        return False
    
    # 检查数据目录
    data_path = "/Users/ryuichi/Documents/GitHub/DDPM/datasets/cifar10-64/train"
    if not os.path.exists(data_path):
        print(f"❌ 数据集路径不存在: {data_path}")
        print("💡 请确保已下载CIFAR-10数据集")
        return False
    
    print("✅ 环境检查通过")
    return True


def run_benchmark():
    """运行性能基准测试"""
    print("\n🧪 运行MPS性能基准测试...")
    try:
        import mps_benchmark
        mps_benchmark.main()
        return True
    except Exception as e:
        print(f"❌ 基准测试失败: {str(e)}")
        return False


def run_setup_test():
    """运行设置测试"""
    print("\n🔧 运行环境设置测试...")
    try:
        import test_mps_setup
        return test_mps_setup.main()
    except Exception as e:
        print(f"❌ 设置测试失败: {str(e)}")
        return False


def run_training(args):
    """运行训练"""
    print("\n🎯 开始DDPM训练...")
    
    try:
        # 导入训练模块
        from ddpm_conditional import launch
        
        # 设置训练参数（如果需要的话）
        if args.epochs:
            print(f"📅 设置训练轮数: {args.epochs}")
        if args.batch_size:
            print(f"📦 设置批次大小: {args.batch_size}")
        if args.lr:
            print(f"📈 设置学习率: {args.lr}")
        
        # 启动训练
        launch()
        
        print("\n🎉 训练完成!")
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="DDPM Conditional Training with MPS Acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_ddpm_mps.py                    # 运行完整流程（测试+训练）
  python run_ddpm_mps.py --test-only        # 仅运行环境测试
  python run_ddpm_mps.py --benchmark-only   # 仅运行性能基准测试
  python run_ddpm_mps.py --train-only       # 跳过测试直接训练
  python run_ddpm_mps.py --epochs 100       # 设置训练轮数
        """
    )
    
    # 运行模式选项
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--test-only", 
        action="store_true",
        help="仅运行环境设置测试"
    )
    mode_group.add_argument(
        "--benchmark-only", 
        action="store_true",
        help="仅运行性能基准测试"
    )
    mode_group.add_argument(
        "--train-only", 
        action="store_true",
        help="跳过测试直接开始训练"
    )
    
    # 训练参数选项
    parser.add_argument(
        "--epochs", 
        type=int,
        help="训练轮数 (默认: 300)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int,
        help="批次大小 (默认: 自动选择)"
    )
    parser.add_argument(
        "--lr", 
        type=float,
        help="学习率 (默认: 3e-4)"
    )
    
    # 其他选项
    parser.add_argument(
        "--skip-prereq-check", 
        action="store_true",
        help="跳过前提条件检查"
    )
    
    args = parser.parse_args()
    
    # 打印启动横幅
    print_banner()
    
    # 检查前提条件
    if not args.skip_prereq_check:
        if not check_prerequisites():
            print("\n❌ 前提条件检查失败，请修复问题后重试")
            sys.exit(1)
    
    success = True
    
    # 根据参数执行相应操作
    if args.test_only:
        success = run_setup_test()
        
    elif args.benchmark_only:
        success = run_benchmark()
        
    elif args.train_only:
        success = run_training(args)
        
    else:
        # 默认流程：测试 -> 基准测试 -> 训练
        print("\n🔄 执行完整流程: 环境测试 -> 性能基准 -> 开始训练")
        
        # 1. 环境测试
        if not run_setup_test():
            print("\n❌ 环境测试失败，无法继续")
            sys.exit(1)
        
        # 2. 询问是否运行基准测试
        print("\n❓ 是否运行性能基准测试? (推荐首次运行时执行)")
        response = input("输入 'y' 运行基准测试，其他键跳过: ").lower().strip()
        
        if response == 'y':
            if not run_benchmark():
                print("⚠️ 基准测试失败，但继续训练...")
        
        # 3. 询问是否开始训练
        print("\n❓ 是否开始训练?")
        response = input("输入 'y' 开始训练，其他键退出: ").lower().strip()
        
        if response == 'y':
            success = run_training(args)
        else:
            print("👋 用户选择退出")
            success = True
    
    # 打印最终结果
    if success:
        print("\n✅ 操作完成!")
        if not (args.test_only or args.benchmark_only):
            print("💡 可以使用 'tensorboard --logdir runs' 查看训练进度")
            print("🖼️  生成的图像保存在 'results/DDPM_conditional/' 目录")
            print("💾 模型检查点保存在 'models/DDPM_conditional/' 目录")
    else:
        print("\n❌ 操作失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
