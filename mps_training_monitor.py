#!/usr/bin/env python3
"""
MPS训练监控脚本
用于监控DDPM在MPS设备上的训练过程，包括性能指标和资源使用情况
"""

import torch
import time
import psutil
import os
import subprocess
from datetime import datetime


class MPSTrainingMonitor:
    """MPS训练监控器"""
    
    def __init__(self, device_name="mps"):
        self.device_name = device_name
        self.device = torch.device(device_name)
        self.start_time = None
        self.epoch_times = []
        self.batch_times = []
        self.memory_usage = []
        self.cpu_usage = []
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        print(f"🔍 开始监控 {self.device_name.upper()} 训练过程")
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def log_epoch_start(self, epoch, total_epochs):
        """记录epoch开始"""
        self.epoch_start_time = time.time()
        print(f"\n📅 Epoch {epoch}/{total_epochs} 开始")
        print(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}")
        
    def log_batch_progress(self, batch_idx, total_batches, loss, lr=None):
        """记录批次进度"""
        current_time = time.time()
        
        # 计算批次时间
        if hasattr(self, 'last_batch_time'):
            batch_time = current_time - self.last_batch_time
            self.batch_times.append(batch_time)
        self.last_batch_time = current_time
        
        # 获取系统资源使用情况
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory_info.percent)
        
        # 每10个批次打印一次详细信息
        if batch_idx % 10 == 0:
            window = self.batch_times[-10:]
            # 在第一个批次时还没有batch_times，避免除以0
            if len(window) > 0:
                avg_batch_time = sum(window) / len(window)
            else:
                # 使用自epoch开始的时间作为粗略估计，或回退为0
                avg_batch_time = (current_time - getattr(self, 'epoch_start_time', current_time))
            
            print(f"  📦 Batch {batch_idx}/{total_batches}")
            print(f"    📉 Loss: {loss:.6f}")
            if lr:
                print(f"    📈 LR: {lr:.2e}")
            print(f"    ⏱️  Batch Time: {avg_batch_time:.3f}s")
            print(f"    💾 CPU: {cpu_percent:.1f}% | RAM: {memory_info.percent:.1f}%")
            
            # 估算剩余时间（需要至少一个已测量的batch时间）
            if len(window) > 0:
                remaining_batches = max(0, total_batches - batch_idx)
                eta_seconds = remaining_batches * avg_batch_time
                eta_minutes = eta_seconds / 60 if eta_seconds > 0 else 0
                print(f"    🕐 ETA: {eta_minutes:.1f} minutes")
    
    def log_epoch_end(self, epoch, avg_loss):
        """记录epoch结束"""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        print(f"\n✅ Epoch {epoch} 完成")
        print(f"  ⏱️  耗时: {epoch_time:.2f}s ({epoch_time/60:.1f} minutes)")
        print(f"  📉 平均损失: {avg_loss:.6f}")
        
        # 计算平均epoch时间和预估总时间
        if len(self.epoch_times) > 0:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            print(f"  📊 平均Epoch时间: {avg_epoch_time:.2f}s")
    
    def log_sampling_start(self, num_images):
        """记录采样开始"""
        self.sampling_start_time = time.time()
        print(f"\n🎨 开始生成 {num_images} 张图像...")
        
    def log_sampling_end(self, num_images, save_path):
        """记录采样结束"""
        sampling_time = time.time() - self.sampling_start_time
        print(f"✅ 图像生成完成")
        print(f"  ⏱️  耗时: {sampling_time:.2f}s")
        if num_images > 0:
            print(f"  🖼️  每张图像: {sampling_time/num_images:.2f}s")
        print(f"  💾 保存路径: {save_path}")
    
    def get_system_info(self):
        """获取系统信息"""
        info = {
            "CPU": {
                "型号": self._get_cpu_model(),
                "核心数": psutil.cpu_count(logical=False),
                "线程数": psutil.cpu_count(logical=True),
                "频率": f"{psutil.cpu_freq().current:.0f} MHz" if psutil.cpu_freq() else "Unknown"
            },
            "内存": {
                "总量": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                "可用": f"{psutil.virtual_memory().available / (1024**3):.1f} GB"
            },
            "PyTorch": {
                "版本": torch.__version__,
                "MPS可用": torch.backends.mps.is_available(),
                "MPS构建": torch.backends.mps.is_built()
            }
        }
        return info
    
    def _get_cpu_model(self):
        """获取CPU型号"""
        try:
            if os.system("which sysctl > /dev/null 2>&1") == 0:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], 
                    capture_output=True, text=True
                )
                return result.stdout.strip()
        except:
            pass
        return "Unknown"
    
    def print_system_info(self):
        """打印系统信息"""
        info = self.get_system_info()
        print("\n💻 系统信息")
        print("=" * 50)
        
        for category, details in info.items():
            print(f"{category}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
            print()
    
    def print_training_summary(self, total_epochs):
        """打印训练总结"""
        if not self.start_time:
            return
            
        total_time = time.time() - self.start_time
        
        print("\n📊 训练总结")
        print("=" * 50)
        print(f"🕐 总训练时间: {total_time:.2f}s ({total_time/3600:.2f} hours)")
        
        if self.epoch_times:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            print(f"📅 平均Epoch时间: {avg_epoch_time:.2f}s")
            print(f"🔄 完成的Epochs: {len(self.epoch_times)}")
            
            if len(self.epoch_times) < total_epochs:
                remaining_epochs = total_epochs - len(self.epoch_times)
                eta_total = remaining_epochs * avg_epoch_time
                print(f"⏳ 预计剩余时间: {eta_total/3600:.2f} hours")
        
        if self.batch_times:
            avg_batch_time = sum(self.batch_times) / len(self.batch_times)
            print(f"📦 平均Batch时间: {avg_batch_time:.3f}s")
            print(f"🔄 处理的Batches: {len(self.batch_times)}")
        
        if self.cpu_usage:
            avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage)
            print(f"💻 平均CPU使用率: {avg_cpu:.1f}%")
        
        if self.memory_usage:
            avg_memory = sum(self.memory_usage) / len(self.memory_usage)
            print(f"💾 平均内存使用率: {avg_memory:.1f}%")
        
        print(f"🎯 设备: {self.device_name.upper()}")
        
        # 性能建议
        self._print_performance_recommendations()
    
    def _print_performance_recommendations(self):
        """打印性能建议"""
        print("\n💡 性能建议")
        print("-" * 30)
        
        if self.batch_times and len(self.batch_times) > 10:
            avg_batch_time = sum(self.batch_times) / len(self.batch_times)
            
            if avg_batch_time > 1.0:
                print("⚠️  批次处理时间较长，建议:")
                print("   - 减少批次大小")
                print("   - 检查数据加载是否为瓶颈")
                print("   - 考虑使用更简单的模型进行测试")
            elif avg_batch_time < 0.1:
                print("✅ 批次处理速度良好，可以考虑:")
                print("   - 增加批次大小以提高GPU利用率")
                print("   - 使用更复杂的模型")
        
        if self.cpu_usage and len(self.cpu_usage) > 10:
            avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage)
            
            if avg_cpu > 80:
                print("⚠️  CPU使用率较高，建议:")
                print("   - 减少数据预处理的复杂度")
                print("   - 使用更多的数据加载工作进程")
            elif avg_cpu < 30:
                print("💡 CPU使用率较低，可以考虑:")
                print("   - 增加数据加载工作进程")
                print("   - 进行更复杂的数据增强")


# 全局监控器实例
monitor = None

def get_monitor(device_name="mps"):
    """获取监控器实例"""
    global monitor
    if monitor is None:
        monitor = MPSTrainingMonitor(device_name)
    return monitor


if __name__ == "__main__":
    # 测试监控器
    monitor = MPSTrainingMonitor("mps")
    monitor.print_system_info()
