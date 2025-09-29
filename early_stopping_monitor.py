#!/usr/bin/env python3
"""
早停监控器 (Early Stopping Monitor)
用于监控DDPM训练过程中的loss变化，检测收敛并实现优雅停止
"""

# import numpy as np  # 使用原生Python实现
import torch
import os
import time
from datetime import datetime
from collections import deque
# import matplotlib.pyplot as plt  # 可选依赖
import logging


class EarlyStoppingMonitor:
    """
    早停监控器
    监控训练损失的变化趋势，在检测到收敛时提供停止建议
    """
    
    def __init__(self, 
                 patience=20,           # 容忍没有改进的epoch数
                 min_delta=1e-6,        # 最小改进阈值
                 smoothing_window=10,   # 平滑窗口大小
                 convergence_threshold=1e-4,  # 收敛阈值（默认更宽松）
                 auto_stop=False,       # 是否自动停止
                 save_plots=True):      # 是否保存loss曲线图
        
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.convergence_threshold = convergence_threshold
        self.auto_stop = auto_stop
        self.save_plots = save_plots
        
        # 监控状态
        self.loss_history = []
        self.smoothed_losses = deque(maxlen=smoothing_window)
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.is_converged = False
        self.should_stop = False
        self.pending_manual_stop_reason = None

        # 统计信息
        self.convergence_detection_history = []
        self.improvement_history = []

        # 判断当前是否处于交互式终端，非交互环境下不触发 input
        try:
            import sys
            self.interactive_session = sys.stdin.isatty()
        except Exception:
            self.interactive_session = False
        
        print(f"🛡️ 早停监控器已启用")
        print(f"   📊 参数配置:")
        print(f"      - 耐心值: {patience} epochs")
        print(f"      - 最小改进: {min_delta}")
        print(f"      - 平滑窗口: {smoothing_window}")
        print(f"      - 收敛阈值: {convergence_threshold}")
        print(f"      - 自动停止: {'是' if auto_stop else '否'}")
    
    def update(self, current_loss, epoch):
        """
        更新监控状态
        参数:
            current_loss: 当前epoch的loss值
            epoch: 当前epoch数
        返回:
            (should_stop, reason): (是否应该停止, 停止原因)
        """
        self.loss_history.append(current_loss)
        self.smoothed_losses.append(current_loss)
        
        # 计算平滑后的loss
        current_smoothed_loss = sum(self.smoothed_losses) / len(self.smoothed_losses)
        
        # 检查是否有改进
        improved = False
        if current_smoothed_loss < self.best_loss - self.min_delta:
            self.best_loss = current_smoothed_loss
            self.best_epoch = epoch
            self.patience_counter = 0
            improved = True
        else:
            self.patience_counter += 1
        
        self.improvement_history.append(improved)
        
        # 收敛检测
        convergence_detected = self._detect_convergence()
        self.convergence_detection_history.append(convergence_detected)
        
        # 决定是否停止
        stop_reason = None
        
        if convergence_detected and not self.is_converged:
            self.is_converged = True
            stop_reason = "收敛检测"
            print(f"\n🎯 检测到收敛! (Epoch {epoch})")
            print(f"   📉 当前loss: {current_loss:.6f}")
            print(f"   📊 平滑loss: {current_smoothed_loss:.6f}")
            print(f"   🏆 最佳loss: {self.best_loss:.6f} (Epoch {self.best_epoch})")
            
            if self.auto_stop:
                self.should_stop = True
            else:
                self.pending_manual_stop_reason = stop_reason
                print(f"   ❓ 是否要停止训练? (自动停止已禁用)")

        elif self.patience_counter >= self.patience:
            stop_reason = f"耐心值耗尽 ({self.patience} epochs无改进)"
            print(f"\n⏰ 触发早停条件! (Epoch {epoch})")
            print(f"   📉 当前loss: {current_loss:.6f}")
            print(f"   🏆 最佳loss: {self.best_loss:.6f} (Epoch {self.best_epoch})")
            print(f"   ⏳ 无改进epochs: {self.patience_counter}")

            if self.auto_stop:
                self.should_stop = True
            else:
                self.pending_manual_stop_reason = stop_reason
                print(f"   ❓ 是否要停止训练? (自动停止已禁用)")

        return self.should_stop, stop_reason
    
    def _detect_convergence(self):
        """检测是否收敛"""
        if len(self.smoothed_losses) < self.smoothing_window:
            return False
        
        # 计算loss变化的方差
        recent_losses = list(self.smoothed_losses)
        if len(recent_losses) < 2:
            return False
            
        # 计算方差
        mean_loss = sum(recent_losses) / len(recent_losses)
        loss_variance = sum((x - mean_loss) ** 2 for x in recent_losses) / len(recent_losses)
        
        # 计算loss变化的趋势(简单的线性拟合)
        if len(recent_losses) >= 3:
            n = len(recent_losses)
            x_values = list(range(n))
            sum_x = sum(x_values)
            sum_y = sum(recent_losses)
            sum_xy = sum(x * y for x, y in zip(x_values, recent_losses))
            sum_x2 = sum(x * x for x in x_values)
            
            # 计算斜率(趋势)
            if n * sum_x2 - sum_x * sum_x != 0:
                trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            else:
                trend = 0
                
            # 使用同时约束趋势与方差；为了兼容不同量级，在阈值上应用相对尺度
            dynamic_threshold = self.convergence_threshold
            if self.best_loss not in (0, float('inf')):
                dynamic_threshold = max(dynamic_threshold, abs(self.best_loss) * 1e-3)

            return abs(trend) < dynamic_threshold and loss_variance < (dynamic_threshold ** 2)

        return False
    
    def get_status_summary(self, epoch):
        """获取状态摘要"""
        if len(self.loss_history) == 0:
            return "监控器尚未接收到loss数据"
        
        current_loss = self.loss_history[-1]
        current_smoothed = sum(self.smoothed_losses) / len(self.smoothed_losses) if self.smoothed_losses else current_loss
        
        status = f"📊 监控状态 (Epoch {epoch}):\n"
        status += f"   📉 当前loss: {current_loss:.6f}\n"
        status += f"   📈 平滑loss: {current_smoothed:.6f}\n"
        status += f"   🏆 最佳loss: {self.best_loss:.6f} (Epoch {self.best_epoch})\n"
        status += f"   ⏳ 耐心计数: {self.patience_counter}/{self.patience}\n"
        status += f"   🎯 收敛状态: {'已收敛' if self.is_converged else '训练中'}\n"
        
        if len(self.loss_history) >= self.smoothing_window:
            recent_losses = list(self.smoothed_losses)
            mean_loss = sum(recent_losses) / len(recent_losses)
            recent_variance = sum((x - mean_loss) ** 2 for x in recent_losses) / len(recent_losses)
            status += f"　　📆 最近方差: {recent_variance:.8f}\n"
        
        return status
    
    def save_loss_plot(self, save_path, run_name):
        """保存loss曲线图"""
        if not self.save_plots or len(self.loss_history) < 2:
            return
        
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            
            # 绘制原始loss
            epochs = range(len(self.loss_history))
            plt.subplot(2, 1, 1)
            plt.plot(epochs, self.loss_history, 'b-', alpha=0.6, label='原始Loss')
            
            # 绘制平滑后的loss
            if len(self.loss_history) >= self.smoothing_window:
                smoothed_all = []
                for i in range(len(self.loss_history)):
                    start_idx = max(0, i - self.smoothing_window + 1)
                    end_idx = i + 1
                    window_data = self.loss_history[start_idx:end_idx]
                    smoothed_all.append(sum(window_data) / len(window_data))
                
                plt.plot(epochs, smoothed_all, 'r-', linewidth=2, label=f'平滑Loss (窗口={self.smoothing_window})')
            
            # 标记最佳点
            plt.axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7, label=f'最佳Epoch ({self.best_epoch})')
            plt.axhline(y=self.best_loss, color='g', linestyle='--', alpha=0.7)
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{run_name} - 训练Loss曲线')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制收敛检测历史
            plt.subplot(2, 1, 2)
            if self.convergence_detection_history:
                convergence_epochs = [i for i, detected in enumerate(self.convergence_detection_history) if detected]
                if convergence_epochs:
                    plt.scatter(convergence_epochs, [1] * len(convergence_epochs), 
                              c='red', s=50, marker='o', label='收敛检测点', zorder=5)
            
            improvement_epochs = [i for i, improved in enumerate(self.improvement_history) if improved]
            if improvement_epochs:
                plt.scatter(improvement_epochs, [0.5] * len(improvement_epochs), 
                          c='green', s=30, marker='^', label='改进点', alpha=0.7, zorder=4)
            
            plt.ylim(-0.1, 1.1)
            plt.xlabel('Epoch')
            plt.ylabel('事件')
            plt.title('训练事件时间线')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            plot_filename = f"{run_name}_loss_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_path = os.path.join(save_path, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Loss分析图已保存: {plot_path}")
            
        except ImportError:
            print("⚠️ matplotlib不可用，跳过图表保存")
            # 保存文本版本的分析报告
            self._save_text_report(save_path, run_name)
        except Exception as e:
            print(f"⚠️ 保存loss图表时出错: {e}")
            self._save_text_report(save_path, run_name)
    
    def save_checkpoint_with_metadata(self, model, ema_model, optimizer, epoch, loss, save_dir, run_name):
        """保存带有元数据的检查点"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 准备元数据
        metadata = {
            'epoch': epoch,
            'loss': loss,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'is_converged': self.is_converged,
            'patience_counter': self.patience_counter,
            'loss_history': self.loss_history,
            'timestamp': timestamp,
            'monitor_config': {
                'patience': self.patience,
                'min_delta': self.min_delta,
                'smoothing_window': self.smoothing_window,
                'convergence_threshold': self.convergence_threshold
            }
        }
        
        # 保存检查点
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': metadata
        }
        
        # 决定文件名
        if self.is_converged:
            filename = f"{run_name}_converged_{timestamp}.pt"
        elif self.patience_counter >= self.patience:
            filename = f"{run_name}_early_stopped_{timestamp}.pt"
        else:
            filename = f"{run_name}_best_checkpoint.pt"
        
        checkpoint_path = os.path.join(save_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        print(f"💾 检查点已保存: {checkpoint_path}")
        
        # 如果是最佳模型，也保存一个简单的版本
        if epoch == self.best_epoch:
            best_path = os.path.join(save_dir, f"{run_name}_best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"🏆 最佳模型已保存: {best_path}")
        
        return checkpoint_path
    
    def _save_text_report(self, save_path, run_name):
        """保存文本版本的分析报告"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"{run_name}_analysis_{timestamp}.txt"
            report_path = os.path.join(save_path, report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"早停监控器分析报告\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"训练名称: {run_name}\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if len(self.loss_history) > 0:
                    f.write(f"训练统计:\n")
                    f.write(f"  总训练epochs: {len(self.loss_history)}\n")
                    f.write(f"  最初loss: {self.loss_history[-1]:.6f}\n")
                    f.write(f"  最佳loss: {self.best_loss:.6f} (Epoch {self.best_epoch})\n")
                    f.write(f"  收敛状态: {'\u5df2收敛' if self.is_converged else '未收敛'}\n")
                    f.write(f"  耐心计数: {self.patience_counter}/{self.patience}\n\n")
                    
                    # 记录loss历史 (最近100个值)
                    recent_losses = self.loss_history[-100:]
                    f.write(f"最近loss值 (最近{len(recent_losses)}个epoch):\n")
                    for i, loss in enumerate(recent_losses):
                        epoch_num = len(self.loss_history) - len(recent_losses) + i
                        f.write(f"  Epoch {epoch_num}: {loss:.6f}")
                        if epoch_num == self.best_epoch:
                            f.write(" (最佳)")
                        f.write("\n")
                    
                    # 收敛检测点
                    if self.convergence_detection_history:
                        convergence_epochs = [i for i, detected in enumerate(self.convergence_detection_history) if detected]
                        if convergence_epochs:
                            f.write(f"\n收敛检测点: {convergence_epochs}\n")
                    
                    # 性能分析
                    if len(self.loss_history) >= 10:
                        initial_loss = self.loss_history[0]
                        final_loss = self.loss_history[-1]
                        improvement = initial_loss - final_loss
                        improvement_pct = (improvement / initial_loss) * 100 if initial_loss > 0 else 0
                        f.write(f"\n性能分析:\n")
                        f.write(f"  初始 loss: {initial_loss:.6f}\n")
                        f.write(f"  最终 loss: {final_loss:.6f}\n")
                        f.write(f"  绝对改进: {improvement:.6f}\n")
                        f.write(f"  相对改进: {improvement_pct:.2f}%\n")
                
                f.write(f"\n监控器参数:\n")
                f.write(f"  耐心值: {self.patience}\n")
                f.write(f"  最小改进: {self.min_delta}\n")
                f.write(f"  平滑窗口: {self.smoothing_window}\n")
                f.write(f"  收敛阈值: {self.convergence_threshold}\n")
            
            print(f"📄 文本分析报告已保存: {report_path}")
            
        except Exception as e:
            print(f"⚠️ 保存文本报告时出错: {e}")
    
    def manual_stop_check(self):
        """手动停止检查 - 在训练循环中调用"""
        if self.pending_manual_stop_reason and not self.auto_stop:
            try:
                if not self.interactive_session:
                    print("⚠️ 当前环境非交互式，跳过手动停止确认")
                    self.pending_manual_stop_reason = None
                    return False, None

                response = input(f"\n❓ {self.pending_manual_stop_reason}，是否停止训练? [y/N]: ").strip().lower()
                if response in ['y', 'yes', '是', '停止']:
                    self.should_stop = True
                    reason = self.pending_manual_stop_reason
                    self.pending_manual_stop_reason = None
                    return True, reason or "用户手动停止"
                # 用户选择继续训练
                self.pending_manual_stop_reason = None
            except (EOFError, KeyboardInterrupt):
                # 输入不可用或被中断，保留状态供下次检查
                print("⚠️ 手动停止确认被中断，稍后将再次询问")
                return False, None

        return False, None


def integrate_early_stopping_to_training():
    """
    将早停监控集成到现有训练代码的示例
    """
    code_example = '''
    # 在训练开始前创建监控器
    early_stopping = EarlyStoppingMonitor(
        patience=20,              # 20个epoch无改进就提醒
        min_delta=1e-6,          # 最小改进阈值
        smoothing_window=10,     # 10个epoch的平滑窗口  
        auto_stop=False,         # 手动确认停止
        save_plots=True          # 保存分析图表
    )
    
    # 在训练循环中使用
    for epoch in range(args.epochs):
        # ... 训练代码 ...
        
        # 计算epoch平均损失
        avg_epoch_loss = epoch_loss_sum / batch_count
        
        # 更新监控器
        should_stop, stop_reason = early_stopping.update(avg_epoch_loss, epoch)
        
        # 打印状态
        if epoch % 5 == 0:
            print(early_stopping.get_status_summary(epoch))
        
        # 检查是否需要停止
        if should_stop:
            print(f"\\n🛑 训练停止: {stop_reason}")
            # 保存最终模型
            early_stopping.save_checkpoint_with_metadata(
                model, ema_model, optimizer, epoch, avg_epoch_loss,
                os.path.join("models", args.run_name), args.run_name
            )
            # 保存分析图表
            early_stopping.save_loss_plot(
                os.path.join("results", args.run_name), args.run_name
            )
            break
        
        # 手动停止检查（仅当检测到收敛时）
        manual_stop, manual_reason = early_stopping.manual_stop_check()
        if manual_stop:
            print(f"\\n🛑 训练停止: {manual_reason}")
            break
    '''
    
    return code_example


if __name__ == "__main__":
    # 演示用法
    print("🔍 早停监控器演示")
    print("=" * 50)
    
    # 模拟loss数据
    np.random.seed(42)
    simulated_losses = []
    
    # 模拟收敛过程
    for i in range(100):
        if i < 20:
            # 初期快速下降
            loss = 2.0 * np.exp(-i/10) + 0.1 + np.random.normal(0, 0.05)
        elif i < 60:
            # 中期缓慢下降
            loss = 0.3 * np.exp(-(i-20)/20) + 0.1 + np.random.normal(0, 0.02)
        else:
            # 后期收敛
            loss = 0.1 + np.random.normal(0, 0.005)
        
        simulated_losses.append(max(0.05, loss))  # 确保loss不会太小
    
    # 测试监控器
    monitor = EarlyStoppingMonitor(patience=15, auto_stop=True)
    
    for epoch, loss in enumerate(simulated_losses):
        should_stop, reason = monitor.update(loss, epoch)
        
        if epoch % 10 == 0:
            print(f"\\nEpoch {epoch}: Loss = {loss:.6f}")
            print(monitor.get_status_summary(epoch))
        
        if should_stop:
            print(f"\\n🛑 模拟训练停止: {reason}")
            break
    
    print("\\n📊 演示完成")
    print(integrate_early_stopping_to_training())
