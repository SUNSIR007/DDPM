#!/usr/bin/env python3
"""
独立早停监控器 - 不依赖任何外部包
用于监控训练损失并检测收敛状态
"""

import os
import time
from datetime import datetime
from collections import deque


class StandaloneEarlyStoppingMonitor:
    """
    独立早停监控器
    监控训练损失的变化趋势，在检测到收敛时提供停止建议
    """
    
    def __init__(self, 
                 patience=20,           # 容忍没有改进的epoch数
                 min_delta=1e-6,        # 最小改进阈值
                 smoothing_window=10,   # 平滑窗口大小
                 convergence_threshold=1e-5,  # 收敛阈值
                 auto_stop=False,       # 是否自动停止
                 save_reports=True):    # 是否保存分析报告
        
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.convergence_threshold = convergence_threshold
        self.auto_stop = auto_stop
        self.save_reports = save_reports
        
        # 监控状态
        self.loss_history = []
        self.smoothed_losses = deque(maxlen=smoothing_window)
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.is_converged = False
        self.should_stop = False
        
        # 统计信息
        self.convergence_detection_history = []
        self.improvement_history = []
        
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
                
            # 如果趋势接近0且方差很小，认为已收敛
            return abs(trend) < self.convergence_threshold and loss_variance < self.convergence_threshold
        
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
            status += f"   📊 最近方差: {recent_variance:.8f}\n"
        
        return status
    
    def save_text_report(self, save_path, run_name):
        """保存文本版本的分析报告"""
        if not self.save_reports:
            return
            
        try:
            os.makedirs(save_path, exist_ok=True)
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
                    f.write(f"  最初loss: {self.loss_history[0]:.6f}\n")
                    f.write(f"  最终loss: {self.loss_history[-1]:.6f}\n")
                    f.write(f"  最佳loss: {self.best_loss:.6f} (Epoch {self.best_epoch})\n")
                    f.write(f"  收敛状态: {'已收敛' if self.is_converged else '未收敛'}\n")
                    f.write(f"  耐心计数: {self.patience_counter}/{self.patience}\n\n")
                    
                    # 记录loss历史 (最近50个值)
                    recent_losses = self.loss_history[-50:]
                    f.write(f"最近loss值 (最近{len(recent_losses)}个epoch):\n")
                    for i, loss in enumerate(recent_losses):
                        epoch_num = len(self.loss_history) - len(recent_losses) + i
                        f.write(f"  Epoch {epoch_num:3d}: {loss:.6f}")
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
            return report_path
            
        except Exception as e:
            print(f"⚠️ 保存文本报告时出错: {e}")
            return None
    
    def manual_stop_check(self):
        """手动停止检查 - 在训练循环中调用"""
        if self.is_converged and not self.auto_stop:
            try:
                response = input("\n❓ 检测到收敛，是否停止训练? [y/N]: ").strip().lower()
                if response in ['y', 'yes', '是', '停止']:
                    self.should_stop = True
                    return True, "用户手动停止"
            except (EOFError, KeyboardInterrupt):
                # 处理输入中断
                return False, None
        
        return False, None


# 演示函数
def demo_early_stopping():
    """演示早停监控器的使用"""
    print("🧪 早停监控器演示")
    print("=" * 50)
    
    # 创建监控器
    monitor = StandaloneEarlyStoppingMonitor(
        patience=15,
        min_delta=1e-5,
        smoothing_window=8,
        convergence_threshold=1e-4,
        auto_stop=True,
        save_reports=True
    )
    
    # 模拟训练loss (收敛场景)
    import random
    random.seed(42)
    
    # 生成模拟的loss序列
    losses = []
    for i in range(100):
        if i < 30:
            # 快速下降阶段
            base_loss = 2.0 * (0.9 ** i) + 0.1
        else:
            # 收敛阶段
            base_loss = 0.1 + random.uniform(-0.01, 0.01)
        losses.append(max(0.05, base_loss))
    
    print("\n🔄 开始模拟训练...")
    
    for epoch, loss in enumerate(losses):
        should_stop, reason = monitor.update(loss, epoch)
        
        # 每20个epoch显示一次状态
        if epoch % 20 == 0:
            print(f"\nEpoch {epoch}: Loss = {loss:.6f}")
            
        # 每10个epoch显示详细状态
        if epoch % 10 == 0:
            print(monitor.get_status_summary(epoch))
        
        if should_stop:
            print(f"\n🛑 训练停止: {reason}")
            break
    
    # 保存分析报告
    report_path = monitor.save_text_report("demo_results", "early_stopping_demo")
    
    print(f"\n📊 演示完成!")
    print(f"   📄 分析报告: {report_path}")
    print(f"   🏆 最佳loss: {monitor.best_loss:.6f}")
    print(f"   🎯 是否收敛: {'是' if monitor.is_converged else '否'}")


if __name__ == "__main__":
    demo_early_stopping()