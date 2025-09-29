# 导入操作系统相关功能模块，用于文件和目录操作
import os
# 导入PyTorch深度学习框架的核心模块
import torch
# 导入PyTorch的计算机视觉工具包，包含数据集、变换和工具函数
import torchvision
# 导入Python图像处理库，用于图像的读取、处理和保存
from PIL import Image
# 导入matplotlib绘图库的pyplot模块，用于图像可视化和绘制
from matplotlib import pyplot as plt
# 导入PyTorch的数据加载器，用于批量加载和处理训练数据
from torch.utils.data import DataLoader


def plot_images(images):
    """
    在本地窗口中可视化显示多张图像的网格拼接结果
    参数: images - 包含多张图像的tensor，形状为(batch_size, channels, height, width)
    """
    # 创建一个32x32英寸的大尺寸图形窗口，用于清晰显示图像网格
    plt.figure(figsize=(32, 32))
    # 使用imshow显示图像网格：
    # 1. 将images移到CPU上进行处理（如果在GPU上的话）
    # 2. 在最后一个维度（宽度）上水平拼接每张图像
    # 3. 在倒数第二个维度（高度）上垂直拼接所有行
    # 4. 调整维度顺序从(C,H,W)到(H,W,C)以符合matplotlib的显示格式
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),  # 水平拼接图像
    ], dim=-2).permute(1, 2, 0).cpu())  # 垂直拼接并调整维度顺序
    # 显示拼接后的图像网格
    plt.show()


def save_images(images, path, **kwargs):
    """
    将图像批次保存为网格格式的图片文件
    参数: images - 图像tensor批次
         path - 保存路径
         **kwargs - 传递给make_grid的额外参数（如nrow, padding等）
    """
    # 使用torchvision工具将图像批次制作成网格布局
    # make_grid自动处理图像的排列和间距
    grid = torchvision.utils.make_grid(images, **kwargs)
    # 将tensor的维度从(C,H,W)调整为(H,W,C)并转换为CPU上的numpy数组
    # 这是因为PIL.Image需要(H,W,C)格式的numpy数组
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    # 从numpy数组创建PIL图像对象
    im = Image.fromarray(ndarr)
    # 将图像保存到指定路径
    im.save(path)


def get_data(args):
    """
    创建并返回数据加载器，用于加载和预处理训练数据
    参数: args - 包含数据集配置参数的对象
    返回: DataLoader对象
    """
    # 定义图像预处理的变换序列，用于数据增强和标准化
    # 根据分辨率自适应选择数据增强策略：32x32 数据集保持随机裁剪 + 水平翻转，
    # 更高分辨率仍使用随机尺寸裁剪以保留多样性。
    transform_steps = []
    if args.image_size <= 32:
        transform_steps.extend([
            torchvision.transforms.RandomCrop(args.image_size, padding=4, padding_mode="reflect"),
            torchvision.transforms.RandomHorizontalFlip(),
        ])
    else:
        resized_edge = int(args.image_size * 1.25)
        transform_steps.extend([
            torchvision.transforms.Resize(resized_edge),
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(),
        ])

    transform_steps.extend([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transforms = torchvision.transforms.Compose(transform_steps)
    # 创建ImageFolder数据集，从指定路径加载图像数据并应用变换
    # ImageFolder假设数据按类别组织在不同的子文件夹中
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    # 备用选项：直接使用CIFAR-10数据集（已注释）
    # dataset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=args.train, transform=transforms, download=True)
    # 创建数据加载器，用于批量加载数据
    # batch_size: 每批次的样本数量
    # shuffle=True: 每个epoch开始时随机打乱数据顺序，提高训练效果
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # 返回配置好的数据加载器
    return dataloader


def setup_logging(run_name):
    """
    创建训练过程中需要的目录结构
    参数: run_name - 本次训练运行的名称，用于区分不同的实验
    """
    # 创建models根目录，用于存储训练好的模型权重文件
    # exist_ok=True表示如果目录已存在则不报错
    os.makedirs("models", exist_ok=True)
    # 创建results根目录，用于存储生成的图像结果和其他输出文件
    os.makedirs("results", exist_ok=True)
    # 为当前运行创建专门的模型存储目录，避免不同实验的文件混淆
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    # 为当前运行创建专门的结果存储目录，便于管理和比较不同实验的输出
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
