#!/usr/bin/env python3
"""
从中断点恢复DDPM训练
Resume DDPM training from interrupted checkpoint
"""

import os
import sys
import torch
import argparse
from datetime import datetime

# 导入现有模块
from ddpm_conditional import Diffusion, train, get_optimal_device, optimize_batch_size_for_device
from modules import UNet_conditional, EMA
from utils import setup_logging, get_data
from mps_training_monitor import get_monitor
from early_stopping_monitor import EarlyStoppingMonitor

import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import copy
import logging

# 配置日志
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def load_interrupted_checkpoint(model_dir, run_name, device):
    """
    加载中断的检查点
    """
    interrupted_files = {
        'model': os.path.join(model_dir, 'interrupted_ckpt.pt'),
        'ema_model': os.path.join(model_dir, 'interrupted_ema_ckpt.pt'), 
        'optimizer': os.path.join(model_dir, 'interrupted_optim.pt')
    }
    
    # 检查文件是否存在
    missing_files = []
    for name, path in interrupted_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print(f"❌ 缺少中断检查点文件:")
        for missing in missing_files:
            print(f"   - {missing}")
        return None
    
    print(f"📁 找到中断检查点文件:")
    for name, path in interrupted_files.items():
        file_size = os.path.getsize(path) / (1024*1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(path))
        print(f"   ✅ {name}: {file_size:.1f}MB (修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    try:
        # 加载检查点
        model_checkpoint = torch.load(interrupted_files['model'], map_location=device)
        ema_checkpoint = torch.load(interrupted_files['ema_model'], map_location=device)  
        optim_checkpoint = torch.load(interrupted_files['optimizer'], map_location=device)
        
        print("✅ 检查点加载成功")
        return {
            'model_state_dict': model_checkpoint,
            'ema_model_state_dict': ema_checkpoint,
            'optimizer_state_dict': optim_checkpoint
        }
    except Exception as e:
        print(f"❌ 加载检查点时出错: {e}")
        return None


def resume_train(args):
    """
    恢复训练主函数
    """
    print("🔄 从中断点恢复DDPM训练")
    print("=" * 60)
    
    # 🔍 初始化训练监控器
    monitor = get_monitor(args.device)
    monitor.start_monitoring()
    monitor.print_system_info()
    
    # 获取设备信息
    device = torch.device(args.device)
    print(f"🎯 使用设备: {device}")
    
    # 创建训练目录
    setup_logging(args.run_name)
    
    # 加载中断的检查点
    model_dir = os.path.join("models", args.run_name)
    checkpoints = load_interrupted_checkpoint(model_dir, args.run_name, device)
    
    if checkpoints is None:
        print("❌ 无法加载检查点，退出程序")
        return False
    
    # 创建模型
    print("🏗️ 初始化模型...")
    model = UNet_conditional(num_classes=args.num_classes, device=args.device, img_size=args.image_size).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 加载检查点状态
    print("📂 恢复模型状态...")
    model.load_state_dict(checkpoints['model_state_dict'])
    ema_model.load_state_dict(checkpoints['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    
    # 创建其他组件
    dataloader = get_data(args)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    
    # 🛡️ 初始化早停监控器
    early_stopping = EarlyStoppingMonitor(
        patience=25,
        min_delta=1e-6,
        smoothing_window=10,
        convergence_threshold=1e-5,
        auto_stop=False,
        save_plots=True
    )
    
    print("✅ 模型恢复完成，开始继续训练...")
    print(f"📊 训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print("=" * 60)
    
    # 开始训练循环
    epoch_losses = []
    start_epoch = args.resume_epoch if hasattr(args, 'resume_epoch') else 0
    
    try:
        for epoch in range(start_epoch, args.epochs):
            # 🔍 记录epoch开始
            monitor.log_epoch_start(epoch, args.epochs)
            logging.info(f"恢复训练 Epoch {epoch}/{args.epochs}:")
            
            # 创建进度条
            from tqdm import tqdm
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            
            epoch_loss_sum = 0.0
            batch_count = 0
            
            # 训练循环
            for i, (images, labels) in enumerate(pbar):
                images = images.to(device)
                labels = labels.to(device)
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(images, t)
                
                # Classifier-Free Guidance训练策略
                import numpy as np
                if np.random.random() < 0.1:
                    labels = None
                    
                predicted_noise = model(x_t, t, labels)
                loss = mse(noise, predicted_noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.step_ema(ema_model, model)
                
                # 记录损失
                loss_value = loss.item()
                epoch_loss_sum += loss_value
                batch_count += 1
                
                # 🔍 记录批次进度
                monitor.log_batch_progress(i, len(dataloader), loss_value, args.lr)
                
                pbar.set_postfix(MSE=loss_value, Device=args.device)
                logger.add_scalar("MSE", loss_value, global_step=epoch * l + i)
            
            # 计算epoch平均损失
            avg_epoch_loss = epoch_loss_sum / batch_count if batch_count > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)
            
            # 🔍 记录epoch结束
            monitor.log_epoch_end(epoch, avg_epoch_loss)
            
            # 🛡️ 更新早停监控器
            should_stop, stop_reason = early_stopping.update(avg_epoch_loss, epoch)
            
            # 每5个epoch打印监控状态
            if epoch % 5 == 0:
                print("\n" + early_stopping.get_status_summary(epoch))

            # 手动停止检查
            manual_stop, manual_reason = early_stopping.manual_stop_check()

            stop_triggered = should_stop or manual_stop
            final_stop_reason = manual_reason if manual_stop else stop_reason

            # 当达到采样间隔或检测到停止时执行采样与检查点保存
            if stop_triggered or epoch % 10 == 0:
                labels = torch.arange(10).long().to(device)
                monitor.log_sampling_start(len(labels))

                sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
                ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)

                from utils import save_images
                main_image_path = os.path.join("results", args.run_name, f"{epoch}.jpg")
                ema_image_path = os.path.join("results", args.run_name, f"{epoch}_ema.jpg")

                save_images(sampled_images, main_image_path, nrow=len(labels))
                save_images(ema_sampled_images, ema_image_path, nrow=len(labels))

                monitor.log_sampling_end(len(labels), f"{main_image_path}, {ema_image_path}")

                torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
                torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
                torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

                print(f"✅ Epoch {epoch}: 模型检查点已保存")

            if stop_triggered:
                reason_to_report = final_stop_reason or "早停条件触发"
                print(f"\n🛑 训练停止条件已触发: {reason_to_report}")
                early_stopping.save_checkpoint_with_metadata(
                    model, ema_model, optimizer, epoch, avg_epoch_loss,
                    os.path.join("models", args.run_name), args.run_name
                )
                early_stopping.save_loss_plot(
                    os.path.join("results", args.run_name), args.run_name
                )
                break
    
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        monitor.print_training_summary(args.epochs)
        print("💾 正在保存当前进度...")
        
        # 保存中断时的状态
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"interrupted_ckpt.pt"))
        torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"interrupted_ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"interrupted_optim.pt"))
        
        # 保存早停监控器数据
        try:
            early_stopping.save_loss_plot(
                os.path.join("results", args.run_name), f"{args.run_name}_interrupted"
            )
            print("📈 训练分析图表已保存")
        except Exception as e:
            print(f"⚠️ 保存分析图表时出错: {e}")
        
        print("✅ 中断状态已保存")
    
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {str(e)}")
        monitor.print_training_summary(args.epochs)
        raise
    
    finally:
        # 打印最终总结
        print("\n🎉 训练完成!")
        monitor.print_training_summary(args.epochs)
        
        # 早停监控器总结
        if 'early_stopping' in locals():
            print("\n🛡️ 早停监控器总结:")
            if len(early_stopping.loss_history) > 0:
                print(f"   📉 总共记录了 {len(early_stopping.loss_history)} 个epoch的loss")
                print(f"   🏆 最佳loss: {early_stopping.best_loss:.6f} (Epoch {early_stopping.best_epoch})")
                print(f"   🎯 收敛状态: {'已收敛' if early_stopping.is_converged else '未收敛'}")
            
            # 保存最终分析图
            try:
                early_stopping.save_loss_plot(
                    os.path.join("results", args.run_name), f"{args.run_name}_resumed"
                )
                print("   📈 最终分析图表已保存")
            except Exception as e:
                print(f"   ⚠️ 保存最终分析图表时出错: {e}")
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Resume DDPM training from interrupted checkpoint")
    parser.add_argument('--epochs', type=int, default=300, help='Total epochs to train')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto if not specified)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch to resume from')
    
    args = parser.parse_args()
    
    print("🔄 DDPM训练恢复程序")
    print("=" * 50)
    
    # 设置基本参数
    args.run_name = "DDPM_conditional"
    args.image_size = 64
    args.num_classes = 10
    args.train = True
    
    # 数据集路径配置
    possible_paths = [
        "./datasets/cifar10-64/train",
        "./data/cifar10-64/train",
        "../datasets/cifar10-64/train",
        "/home/ryuichi/datasets/cifar10-64/train",
        "/tmp/cifar10-64/train"
    ]
    
    dataset_found = False
    for path in possible_paths:
        if os.path.exists(path):
            args.dataset_path = path
            dataset_found = True
            print(f"📊 找到数据集: {path}")
            break
    
    if not dataset_found:
        print("⚠️ 未找到数据集，使用默认路径")
        args.dataset_path = possible_paths[0]
    
    # 获取最优设备
    device_name, _, device_info = get_optimal_device()
    args.device = device_name
    
    # 优化批次大小
    if args.batch_size is None:
        base_batch_size = 4
        args.batch_size = optimize_batch_size_for_device(device_name, base_batch_size)
    
    # CUDA优化设置
    if device_name == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("🔧 已启用CUDA优化")
    
    print(f"\n📋 训练配置:")
    print(f"   🎯 运行名称: {args.run_name}")
    print(f"   🔄 总轮数: {args.epochs}")
    print(f"   📦 批次大小: {args.batch_size}")
    print(f"   🖼️ 图像尺寸: {args.image_size}x{args.image_size}")
    print(f"   🏷️ 类别数: {args.num_classes}")
    print(f"   📚 数据集: {args.dataset_path}")
    print(f"   🎛️ 设备: {device_info}")
    print(f"   📈 学习率: {args.lr}")
    print("=" * 50)
    
    # 开始恢复训练
    success = resume_train(args)
    
    if success:
        print("\n✅ 训练恢复完成!")
    else:
        print("\n❌ 训练恢复失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()
