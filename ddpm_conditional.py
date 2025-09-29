# 导入操作系统相关功能，用于路径操作和文件管理
import os
# 导入深拷贝功能，用于创建模型的独立副本（EMA模型）
import copy
# 导入NumPy数值计算库，用于数组操作和随机数生成
import numpy as np
# 导入PyTorch深度学习框架的核心模块
import torch
# 导入PyTorch的神经网络模块，包含损失函数和网络层
import torch.nn as nn
# 导入进度条库，用于显示训练进度
from tqdm import tqdm
# 导入PyTorch的优化器模块
from torch import optim
# 导入自定义的工具函数（数据加载、图像保存等）
from utils import *
# 导入自定义的网络模块（条件UNet和EMA）
from modules import UNet_conditional, EMA
# 导入日志记录模块，用于输出训练信息
import logging
# 导入TensorBoard写入器，用于记录训练指标和可视化
from torch.utils.tensorboard import SummaryWriter
# 导入MPS训练监控器和早停监控器
from mps_training_monitor import get_monitor
from early_stopping_monitor import EarlyStoppingMonitor

# 配置日志格式：显示时间戳、日志级别和消息内容
# level=logging.INFO表示记录INFO级别及以上的日志
# datefmt设置时间格式为小时:分钟:秒
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    """
    扩散模型的核心类，实现前向加噪过程和反向去噪过程
    包含噪声调度、时间步采样、图像加噪和采样生成等功能
    """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        """
        初始化扩散模型参数
        参数:
            noise_steps: 扩散过程的总时间步数，默认1000步
            beta_start: 噪声调度的起始值，控制初始噪声强度
            beta_end: 噪声调度的结束值，控制最终噪声强度
            img_size: 图像尺寸（正方形图像的边长）
            device: 计算设备（"cuda"或"cpu"）
        """
        # 存储扩散过程的总时间步数
        self.noise_steps = noise_steps
        # 存储噪声调度的起始值（较小，初期加噪较少）
        self.beta_start = beta_start
        # 存储噪声调度的结束值（较大，后期加噪较多）
        self.beta_end = beta_end
        # 存储图像尺寸
        self.img_size = img_size
        # 存储计算设备
        self.device = device

        # 生成噪声调度序列并移动到指定设备
        self.beta = self.prepare_noise_schedule().to(device)
        # 计算alpha序列：alpha_t = 1 - beta_t，表示保留原图像的比例
        self.alpha = 1. - self.beta
        # 计算累积乘积alpha_hat：用于直接从x_0计算x_t，避免逐步计算
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        # 预计算噪声系数，避免训练循环里重复分解调度表，提高计算效率

    def prepare_noise_schedule(self):
        """
        生成每个时间步的噪声强度beta值
        使用线性调度：从beta_start线性增长到beta_end
        返回: 形状为(noise_steps,)的tensor，包含每个时间步的beta值
        """
        # 创建从beta_start到beta_end的线性序列
        # 随着时间步增加，beta值增大，意味着加入更多噪声
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        """
        为每张图像随机采样时间步
        在训练时，每个样本需要随机选择一个时间步进行加噪
        参数:
            n: 批次大小（batch_size）
        返回:
            t: 时间步tensor，形状为(n,)，值在[1, noise_steps)范围内
        """
        # 为每个样本随机选择一个时间步
        # low=1表示从时间步1开始（避免t=0的边界情况）
        # high=self.noise_steps表示最大时间步（不包含）
        return torch.randint(low=1, high=self.noise_steps, size=(n, ))

    def noise_images(self, x, t):
        """
        前向加噪过程：将干净图像x_0直接加噪到时间步t得到x_t
        使用重参数化技巧，可以直接从x_0计算任意时间步的x_t
        参数:
            x: 输入图像，形状为(batch_size, channels, img_size, img_size)
            t: 时间步，形状为(batch_size,)
        返回:
            加噪后的图像x_t和对应的噪声epsilon
            公式: x_t = sqrt(alpha_hat[t]) * x_0 + sqrt(1 - alpha_hat[t]) * epsilon
        """
        # 获取时间步t对应的alpha_hat值的平方根
        # [:, None, None, None]用于广播到图像的所有维度
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        # 获取(1 - alpha_hat[t])的平方根，控制噪声的强度
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        # 生成与输入图像相同形状的随机噪声
        Ɛ = torch.randn_like(x)
        # 按扩散模型的重参数化公式混合真实图像与随机噪声
        # 返回加噪后的图像和真实噪声（用于训练时的损失计算）
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample(self, model, n, labels, cfg_scale=3):
        """
        反向去噪过程：从纯噪声x_T逐步去噪生成干净图像x_0
        实现Classifier-Free Guidance以提高条件生成的质量
        参数:
            model: 训练好的UNet模型，用于预测噪声
            n: 要生成的图像数量（批次大小）
            labels: 条件标签，形状为(n,)，值在[0, num_classes-1]范围内
            cfg_scale: Classifier-Free Guidance的缩放因子
                      0.0表示无条件生成，>1.0表示增强条件引导
        返回:
            生成的图像，形状为(n, 3, img_size, img_size)，像素值在[0, 255]
            去噪公式: x_{t-1} = 1/sqrt(alpha[t]) * (x_t - (1-alpha[t])/sqrt(1-alpha_hat[t]) * predicted_noise) + sqrt(beta[t]) * noise
        """
        # 记录开始生成图像的日志信息
        logging.info(f"Sampling {n} new images....")
        # 将模型设置为评估模式，禁用dropout和batch normalization的训练行为
        model.eval()
        # 使用torch.no_grad()禁用梯度计算，节省内存并加速推理
        with torch.no_grad():
            # 初始化为纯随机噪声，这是扩散过程的起点x_T
            # 形状为(n, 3, img_size, img_size)，3表示RGB三个通道
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # 从最大时间步开始，逐步向t=0去噪
            # reversed(range(1, self.noise_steps))生成[T-1, T-2, ..., 1]的序列
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # 为当前批次的所有样本创建相同的时间步tensor
                t = (torch.ones(n) * i).long().to(self.device)
                # 使用模型预测当前时间步的噪声（条件生成）
                predicted_noise = model(x, t, labels)
                # 实现Classifier-Free Guidance：结合条件和无条件预测
                if cfg_scale > 0:
                    # 获取无条件预测（labels=None）
                    uncond_predicted_noise = model(x, t, None)
                    # 使用线性插值结合两种预测，cfg_scale控制条件引导的强度
                    # cfg_scale > 1时会放大条件和无条件预测的差异，增强条件控制
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                # 获取当前时间步的扩散参数，并扩展维度以匹配图像形状
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                # 在非最后一步时添加随机噪声，保持扩散过程的随机性
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    # 最后一步(t=1->t=0)不添加噪声，确保最终结果确定
                    noise = torch.zeros_like(x)
                # 执行反向扩散的核心公式，从x_t计算x_{t-1}
                # 这个公式来源于扩散模型的数学推导，用于逐步去除噪声
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        # 恢复模型的训练模式
        model.train()
        # 将生成的图像从[-1, 1]范围映射到[0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        # 将像素值从[0, 1]缩放到[0, 255]并转换为uint8类型
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    """
    扩散模型的训练主函数
    训练过程：每个epoch中，对每个batch随机选择时间步，比较预测噪声与真实噪声
    实现Classifier-Free Guidance训练策略，提高条件生成质量
    现在集成了MPS监控功能，提供详细的训练进度和性能信息
    参数:
        args: 包含训练配置的参数对象
    """
    # 🔍 初始化训练监控器
    monitor = get_monitor(args.device)
    monitor.start_monitoring()
    monitor.print_system_info()
    
    # 🛡️ 初始化早停监控器
    early_stopping = EarlyStoppingMonitor(
        patience=25,              # 25个epoch无改进就提醒
        min_delta=1e-6,          # 最小改进阈值
        smoothing_window=10,     # 10个epoch的平滑窗口
        convergence_threshold=1e-5,  # 收敛阈值
        auto_stop=False,         # 手动确认停止（更安全）
        save_plots=True          # 保存分析图表
    )

    # 创建训练所需的目录结构（models、results等）
    setup_logging(args.run_name)
    # 获取计算设备（GPU或CPU）
    device = args.device
    # 创建数据加载器，用于批量加载和预处理训练数据
    dataloader = get_data(args)
    # 初始化条件UNet模型并移动到指定设备
    model = UNet_conditional(num_classes=args.num_classes, device=device, img_size=args.image_size).to(device)
    # 使用AdamW优化器，它在扩散模型训练中表现良好
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # 定义均方误差损失函数，用于比较预测噪声和真实噪声
    mse = nn.MSELoss()
    # 初始化扩散过程管理器
    diffusion = Diffusion(img_size=args.image_size, device=device)
    # 创建TensorBoard日志写入器，用于记录训练指标
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    # 获取数据加载器的长度，用于计算全局步数
    l = len(dataloader)
    # 初始化指数移动平均(EMA)，衰减率为0.995
    # EMA有助于稳定训练并提高生成质量
    ema = EMA(0.995)
    # 创建EMA模型：深拷贝主模型并设置为评估模式，禁用梯度计算
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # 开始训练循环，遍历所有epoch
    epoch_losses = []  # 记录每个epoch的损失

    try:
        for epoch in range(args.epochs):
            # 🔍 记录epoch开始
            monitor.log_epoch_start(epoch, args.epochs)

            # 记录当前epoch的开始信息
            logging.info(f"Starting epoch {epoch}/{args.epochs}:")
            # 创建进度条，显示当前epoch的训练进度
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

            epoch_loss_sum = 0.0
            batch_count = 0

            # 遍历数据加载器中的每个batch
            for i, (images, labels) in enumerate(pbar):
                # 将图像数据移动到指定设备（GPU/CPU/MPS）
                images = images.to(device)
                # 将标签数据移动到指定设备
                labels = labels.to(device)
                # 为当前batch的每张图像随机采样时间步
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                # 对图像进行前向加噪，获得加噪图像x_t和对应的真实噪声
                x_t, noise = diffusion.noise_images(images, t)
                # 实现Classifier-Free Guidance训练策略：
                # 10%的时间丢弃条件标签，训练无条件生成能力
                # 这使得模型既能进行条件生成，也能进行无条件生成
                if np.random.random() < 0.1:
                    labels = None
                # 使用模型预测噪声：输入加噪图像、时间步和条件标签
                predicted_noise = model(x_t, t, labels)
                # 计算预测噪声与真实噪声之间的均方误差损失
                loss = mse(noise, predicted_noise)

                # 清零优化器的梯度缓存
                optimizer.zero_grad()
                # 反向传播计算梯度
                loss.backward()
                # 更新模型参数
                optimizer.step()
                # 更新EMA模型：使用指数移动平均平滑主模型的参数
                # 这有助于提升生成样本的质量和稳定性
                ema.step_ema(ema_model, model)

                # 记录损失和批次信息
                loss_value = loss.item()
                epoch_loss_sum += loss_value
                batch_count += 1

                # 🔍 记录批次进度（使用监控器）
                monitor.log_batch_progress(i, len(dataloader), loss_value, args.lr)

                # 在进度条中显示当前的MSE损失值
                pbar.set_postfix(MSE=loss_value, Device=device)
                # 将损失值记录到TensorBoard，用于可视化训练过程
                logger.add_scalar("MSE", loss_value, global_step=epoch * l + i)

            # 计算epoch平均损失
            avg_epoch_loss = epoch_loss_sum / batch_count if batch_count > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)

            # 🔍 记录epoch结束
            monitor.log_epoch_end(epoch, avg_epoch_loss)
            
            # 🛡️ 更新早停监控器
            should_stop, stop_reason = early_stopping.update(avg_epoch_loss, epoch)

            # 每5个epoch打印一次详细的监控状态
            if epoch % 5 == 0:
                print("\n" + early_stopping.get_status_summary(epoch))

            # 检查是否需要手动确认停止
            manual_stop, manual_reason = early_stopping.manual_stop_check()

            stop_triggered = should_stop or manual_stop
            final_stop_reason = manual_reason if manual_stop else stop_reason

            # 当检测到停止条件或达到采样间隔时保存样本与检查点
            if stop_triggered or epoch % 10 == 0:
                labels = torch.arange(10).long().to(device)
                monitor.log_sampling_start(len(labels))

                sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
                ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)

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
                print(f"\n🛁 训练停止条件已触发: {reason_to_report}")
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

        # 保存中断时的模型状态
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"interrupted_ckpt.pt"))
        torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"interrupted_ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"interrupted_optim.pt"))
        
        # 🛡️ 保存早停监控器数据
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
        # 🔍 打印最终的训练总结
        print("\n🎉 训练完成!")
        monitor.print_training_summary(args.epochs)
        
        # 🛡️ 打印早停监控器总结
        if 'early_stopping' in locals():
            print("\n🛡️ 早停监控器总结:")
            if len(early_stopping.loss_history) > 0:
                print(f"   📉 总共记录了 {len(early_stopping.loss_history)} 个epoch的loss")
                print(f"   🏆 最佳loss: {early_stopping.best_loss:.6f} (Epoch {early_stopping.best_epoch})")
                print(f"   🎯 收敛状态: {'\u5df2收敛' if early_stopping.is_converged else '\u672a收敛'}")
            
            # 最终保存loss分析图
            try:
                early_stopping.save_loss_plot(
                    os.path.join("results", args.run_name), f"{args.run_name}_final"
                )
                print("   📈 最终分析图表已保存")
            except Exception as e:
                print(f"   ⚠️ 保存最终分析图表时出错: {e}")


def get_optimal_device():
    """
    获取最优的计算设备，优先级：CUDA > CPU
    返回: (device_name, device_object, device_info)
    """
    import time

    # 优先使用CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name()
        print(f"🔥 Using CUDA: {gpu_name}")
        
        # 显示CUDA相关信息
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 测试CUDA性能
        try:
            test_tensor = torch.randn(1000, 1000, device=device)
            torch.cuda.synchronize()  # 确保操作完成
            start_time = time.time()
            _ = torch.matmul(test_tensor, test_tensor)
            torch.cuda.synchronize()
            end_time = time.time()
            print(f"   CUDA performance test: {(end_time - start_time)*1000:.2f}ms")
        except Exception as e:
            print(f"   ⚠️  CUDA test failed: {e}")
        
        return "cuda", device, f"NVIDIA GPU: {gpu_name}"

    # 回退到CPU
    device = torch.device("cpu")
    print("💻 Using CPU")
    print("   ⚠️  CUDA not available - training will be significantly slower")
    print("   💡 For GPU acceleration, ensure CUDA-compatible GPU and drivers are installed")

    return "cpu", device, "CPU (fallback)"


def optimize_batch_size_for_device(device_name, image_size, base_batch_size=64):
    """
    根据设备类型优化批次大小
    参数:
        device_name: 设备名称 ("cuda", "cpu")
        base_batch_size: 基础批次大小
    返回:
        优化后的批次大小
    """
    if device_name == "cuda":
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        if image_size <= 32:
            if gpu_memory_gb >= 16.0:
                optimized_size = 256
            elif gpu_memory_gb >= 10.0:
                optimized_size = 192
            elif gpu_memory_gb >= 8.0:
                optimized_size = 128
            elif gpu_memory_gb >= 6.0:
                optimized_size = 96
            else:
                optimized_size = 64
        else:
            if gpu_memory_gb >= 16.0:
                optimized_size = 128
            elif gpu_memory_gb >= 10.0:
                optimized_size = 96
            elif gpu_memory_gb >= 8.0:
                optimized_size = 64
            elif gpu_memory_gb >= 6.0:
                optimized_size = 48
            else:
                optimized_size = max(32, base_batch_size // 2)

        print(f"   📊 Optimized batch size for CUDA ({gpu_memory_gb:.1f}GB, image {image_size}px): {optimized_size}")
        return optimized_size

    if device_name == "mps":
        optimized_size = 96 if image_size <= 32 else 64
        print(f"   📊 Optimized batch size for MPS (image {image_size}px): {optimized_size}")
        return optimized_size

    cpu_cores = os.cpu_count() or 4
    if image_size <= 32:
        optimized_size = max(16, min(32, cpu_cores * 2))
    else:
        optimized_size = max(8, min(16, cpu_cores))
    print(f"   📊 Optimized batch size for CPU (image {image_size}px, cores {cpu_cores}): {optimized_size}")
    return optimized_size


def launch():
    """
    启动训练的主函数，配置所有训练参数并开始训练过程
    现在支持智能设备选择，优先使用CUDA加速
    """
    # 导入命令行参数解析器
    import argparse
    import os
    # 创建参数解析器实例
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto if not specified)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    # 解析命令行参数
    args = parser.parse_args()

    print("🎯 DDPM Conditional Training Setup")
    print("=" * 50)

    # 设置实验运行名称，用于区分不同的训练实验
    args.run_name = "DDPM_conditional"
    # 设置图像尺寸（32x32像素，匹配精简数据集并加快迭代）
    args.image_size = 32
    # 设置类别数量（CIFAR-10有10个类别）
    args.num_classes = 10
    
    # 数据集路径配置 - 支持Linux环境
    if args.dataset_path is None:
        # 默认数据集路径列表（优先级顺序）
        possible_paths = [
            "./datasets/cifar10-32/train",
            "./data/cifar10-32/train",
            "../datasets/cifar10-32/train",
            "./datasets/cifar10-64/train",
            "./data/cifar10-64/train",
            "../datasets/cifar10-64/train",
            "/home/ryuichi/datasets/cifar10-32/train",
            "/home/ryuichi/datasets/cifar10-64/train",
            "/tmp/cifar10-32/train",
            "/tmp/cifar10-64/train"
        ]
        
        # 查找存在的数据集路径
        dataset_found = False
        for path in possible_paths:
            if os.path.exists(path):
                args.dataset_path = path
                dataset_found = True
                break
        
        if not dataset_found:
            print("⚠️  未找到数据集！")
            print("💡 可以使用以下方式解决：")
            print("   1. 使用 --dataset_path 参数指定数据集路径")
            print("   2. 将数据集放在以下位置之一：")
            for path in possible_paths[:3]:
                print(f"      - {path}")
            print("   3. 使用CIFAR-10自动下载（修改utils.py）")
            # 使用第一个路径作为默认值，但会提醒用户
            args.dataset_path = possible_paths[0]
    
    # 设置为训练模式（区别于推理模式）
    args.train = True

    # 🚀 智能设备选择和配置
    device_name, _, device_info = get_optimal_device()  # device_obj在这里不需要使用
    args.device = device_name

    # 📊 根据设备优化批次大小
    if args.batch_size is None:
        base_batch_size = 64
        args.batch_size = optimize_batch_size_for_device(device_name, args.image_size, base_batch_size)

    # 🔧 CUDA特定的优化设置
    if device_name == "cuda":
        # 启用CUDA优化
        torch.backends.cudnn.benchmark = True  # 优化固定尺寸输入的性能
        torch.backends.cudnn.deterministic = False  # 为了性能放弃一些确定性
        print("   🔧 Enabled CUDA optimizations (cudnn.benchmark=True)")
        
        # 根据显存调整其他参数
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb >= 8.0 and args.batch_size < 8:
            old_batch_size = args.batch_size
            args.batch_size = min(8, args.batch_size * 2)
            print(f"   📈 Increased batch size from {old_batch_size} to {args.batch_size} for large GPU")

    print(f"\n📋 Training Configuration:")
    print(f"   🎯 Run name: {args.run_name}")
    print(f"   🔄 Epochs: {args.epochs}")
    print(f"   📦 Batch size: {args.batch_size}")
    print(f"   🖼️  Image size: {args.image_size}x{args.image_size}")
    print(f"   🏷️  Classes: {args.num_classes}")
    print(f"   📚 Dataset: {args.dataset_path}")
    print(f"   🎛️  Device: {device_info}")
    print(f"   📈 Learning rate: {args.lr}")
    print("=" * 50)

    # 开始训练过程
    train(args)


# 程序入口点：当直接运行此脚本时执行
if __name__ == '__main__':
    # 启动训练过程
    launch()
    # 以下是推理代码示例（已注释）：
    # 用于加载训练好的模型并生成图像
    # device = "cuda"
    # # 创建模型实例
    # model = UNet_conditional(num_classes=10).to(device)
    # # 加载训练好的模型权重
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # # 创建扩散过程管理器
    # diffusion = Diffusion(img_size=64, device=device)
    # # 设置生成图像的数量
    # n = 8
    # # 创建条件标签（生成8张类别为6的图像）
    # y = torch.Tensor([6] * n).long().to(device)
    # # 生成图像（cfg_scale=0表示无Classifier-Free Guidance）
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # # 显示生成的图像
    # plot_images(x)
