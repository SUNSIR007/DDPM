# 导入PyTorch深度学习框架的核心模块
import torch
# 导入PyTorch的神经网络模块，包含各种网络层和激活函数
import torch.nn as nn
# 导入PyTorch的函数式接口，提供各种操作函数
import torch.nn.functional as F


class EMA:
    """
    指数移动平均(Exponential Moving Average)类
    用于追踪模型参数的指数滑动平均，提供更稳定的推理性能
    在扩散模型中，EMA模型通常能生成更高质量、更稳定的图像
    """
    def __init__(self, beta):
        """
        初始化EMA对象
        参数:
            beta: 衰减率，通常接近1（如0.995），控制历史权重的保留程度
        """
        super().__init__()
        # 存储衰减率，决定新旧参数的混合比例
        self.beta = beta
        # 记录更新步数，用于控制EMA的启动时机
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        """
        更新EMA模型的所有参数
        遍历当前模型和EMA模型的对应参数，进行指数移动平均更新
        参数:
            ma_model: EMA模型（移动平均模型）
            current_model: 当前训练的模型
        """
        # 同时遍历两个模型的参数
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            # 获取EMA模型的旧权重和当前模型的新权重
            old_weight, up_weight = ma_params.data, current_params.data
            # 使用指数移动平均公式更新EMA模型的参数
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        执行指数移动平均的核心计算
        公式: new_avg = beta * old_avg + (1 - beta) * new_value
        参数:
            old: 旧的平均值
            new: 新的值
        返回:
            更新后的平均值
        """
        # 如果旧值为None（初始化情况），直接返回新值
        if old is None:
            return new
        # 应用指数移动平均公式：大部分保留旧值，小部分采用新值
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        执行一步EMA更新
        在训练初期直接复制参数，训练稳定后才开始真正的EMA更新
        参数:
            ema_model: EMA模型
            model: 当前训练模型
            step_start_ema: 开始EMA更新的步数阈值，默认2000步
        """
        # 如果训练步数还不够，直接复制当前模型的参数到EMA模型
        # 这是因为训练初期模型变化很大，EMA可能不稳定
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        # 训练稳定后，开始真正的指数移动平均更新
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        将当前模型的参数完全复制到EMA模型
        通常在训练初期或重置时使用
        参数:
            ema_model: 要更新的EMA模型
            model: 源模型
        """
        # 直接加载当前模型的状态字典，实现参数的完全复制
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    """
    自注意力机制模块
    在特征图的空间维度上应用多头自注意力，增强长程空间依赖关系
    这对于生成高质量图像特别重要，因为它能捕捉图像中远距离像素间的关系
    """
    def __init__(self, channels, size):
        """
        初始化自注意力模块
        参数:
            channels: 输入特征图的通道数
            size: 特征图的空间尺寸（假设为正方形）
        """
        super(SelfAttention, self).__init__()
        # 存储通道数，用于后续的维度变换
        self.channels = channels
        # 存储特征图尺寸，用于reshape操作
        self.size = size
        # 多头注意力机制，使用4个注意力头，batch_first=True表示批次维度在前
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        # 层归一化，用于稳定训练过程
        self.ln = nn.LayerNorm([channels])
        # 前馈神经网络，实现Transformer中的FFN结构
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),  # 层归一化
            nn.Linear(channels, channels),  # 线性变换
            nn.GELU(),  # GELU激活函数，比ReLU更平滑
            nn.Linear(channels, channels),  # 第二个线性变换
        )

    def forward(self, x):
        """
        前向传播过程
        参数:
            x: 输入特征图，形状为(batch_size, channels, size, size)
        返回:
            经过自注意力处理的特征图，形状不变
        """
        # 将特征图从(B, C, H, W)重塑为(B, H*W, C)，将空间维度展平为序列
        # 这样每个空间位置都成为序列中的一个token
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        # 对输入进行层归一化，稳定训练
        x_ln = self.ln(x)
        # 执行多头自注意力：query、key、value都是同一个输入（自注意力）
        # 返回注意力输出和注意力权重（这里忽略权重）
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        # 残差连接：将注意力输出与原始输入相加
        attention_value = attention_value + x
        # 通过前馈网络进一步处理，并再次使用残差连接
        attention_value = self.ff_self(attention_value) + attention_value
        # 将序列形式的输出重新reshape回特征图格式(B, C, H, W)
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    双卷积块，UNet的基本构建单元
    包含两个卷积层，每个卷积层后跟组归一化和激活函数
    可选择是否使用残差连接来改善梯度流动
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        """
        初始化双卷积块
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            mid_channels: 中间层通道数，如果为None则等于out_channels
            residual: 是否使用残差连接
        """
        super().__init__()
        # 标记是否使用残差连接
        self.residual = residual
        # 如果没有指定中间通道数，则使用输出通道数
        if not mid_channels:
            mid_channels = out_channels
        # 定义双卷积序列
        self.double_conv = nn.Sequential(
            # 第一个卷积层：3x3卷积，padding=1保持尺寸不变，不使用偏置
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # 组归一化：将通道分为1组（等价于LayerNorm），比BatchNorm更稳定
            nn.GroupNorm(1, mid_channels),
            # GELU激活函数：比ReLU更平滑，在Transformer和扩散模型中表现更好
            nn.GELU(),
            # 第二个卷积层
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # 第二个组归一化
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入特征图
        返回:
            处理后的特征图
        """
        if self.residual:
            # 使用残差连接：输入与卷积输出相加，然后应用激活函数
            # 残差连接有助于缓解梯度消失问题，保持特征稳定性
            return F.gelu(x + self.double_conv(x))
        else:
            # 不使用残差连接，直接返回卷积结果
            return self.double_conv(x)


class Down(nn.Module):
    """
    UNet的下采样模块（编码器部分）
    通过最大池化减小特征图尺寸，同时增加通道数以提取更抽象的特征
    集成时间步嵌入，使模型能够感知当前的扩散时间步
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        """
        初始化下采样模块
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            emb_dim: 时间步嵌入的维度
        """
        super().__init__()
        # 下采样和卷积序列
        self.maxpool_conv = nn.Sequential(
            # 2x2最大池化，将特征图尺寸减半
            nn.MaxPool2d(2),
            # 第一个双卷积块，使用残差连接保持特征稳定性
            DoubleConv(in_channels, in_channels, residual=True),
            # 第二个双卷积块，改变通道数
            DoubleConv(in_channels, out_channels),
        )

        # 时间步嵌入处理层
        self.emb_layer = nn.Sequential(
            # SiLU激活函数（Sigmoid Linear Unit），在扩散模型中表现良好
            nn.SiLU(),
            # 线性层将时间嵌入映射到输出通道数
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        """
        前向传播
        参数:
            x: 输入特征图，形状为(batch_size, in_channels, height, width)
            t: 时间步嵌入，形状为(batch_size, emb_dim)
        返回:
            融合时间信息的下采样特征图
        """
        # 执行下采样和卷积操作
        x = self.maxpool_conv(x)
        # 处理时间步嵌入并扩展到特征图的空间维度
        # [:, :, None, None]添加空间维度，repeat广播到整个特征图
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # 将时间步嵌入加到特征图上，注入扩散阶段信息
        # 这使得模型能够根据当前时间步调整特征表示
        return x + emb


class Up(nn.Module):
    """
    UNet的上采样模块（解码器部分）
    通过上采样恢复特征图尺寸，结合跳跃连接保留细节信息
    同样集成时间步嵌入以保持时间感知能力
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        """
        初始化上采样模块
        参数:
            in_channels: 输入通道数（包括跳跃连接的通道）
            out_channels: 输出通道数
            emb_dim: 时间步嵌入的维度
        """
        super().__init__()

        # 双线性插值上采样，将特征图尺寸放大2倍
        # align_corners=True确保角点对齐，提供更好的上采样质量
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # 卷积处理序列
        self.conv = nn.Sequential(
            # 第一个双卷积块，使用残差连接
            DoubleConv(in_channels, in_channels, residual=True),
            # 第二个双卷积块，减少通道数
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        # 时间步嵌入处理层
        self.emb_layer = nn.Sequential(
            # SiLU激活函数
            nn.SiLU(),
            # 线性层映射到输出通道数
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        """
        前向传播
        参数:
            x: 来自更深层的特征图（需要上采样）
            skip_x: 来自编码器对应层的跳跃连接特征图
            t: 时间步嵌入
        返回:
            融合跳跃连接和时间信息的上采样特征图
        """
        # 对输入特征图进行上采样，恢复空间分辨率
        x = self.up(x)
        # 在通道维度上拼接跳跃连接特征和上采样特征
        # 跳跃连接保留了编码阶段的细节信息，有助于精确重建
        x = torch.cat([skip_x, x], dim=1)
        # 通过卷积处理拼接后的特征
        x = self.conv(x)
        # 处理时间步嵌入并广播到特征图
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # 将时间嵌入加到特征图上
        # 编码时的skip_x与时间嵌入共同指导解码阶段的特征生成
        return x + emb


class UNet(nn.Module):
    """
    标准UNet架构，用于扩散模型的噪声预测
    采用编码器-解码器结构，结合跳跃连接和自注意力机制
    能够处理时间步信息，但不支持条件生成
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cpu", img_size=64):
        """
        初始化UNet模型
        参数:
            c_in: 输入图像通道数（RGB图像为3）
            c_out: 输出图像通道数（通常等于输入通道数）
            time_dim: 时间步嵌入的维度
            device: 计算设备
        """
        super().__init__()
        # 存储设备信息，用于张量操作
        self.device = device
        # 存储时间嵌入维度
        self.time_dim = time_dim
        self.img_size = img_size

        if img_size % 8 != 0:
            raise ValueError("img_size 必须能被 8 整除，以匹配当前UNet结构")

        attn_sizes = [img_size // 2, img_size // 4, img_size // 8]
        decoder_attn_sizes = [img_size // 4, img_size // 2, img_size]

        # 输入卷积层：将输入图像映射到64通道
        self.inc = DoubleConv(c_in, 64)

        # 编码器路径（下采样）
        self.down1 = Down(64, 128)      # 64->128通道，尺寸减半
        self.sa1 = SelfAttention(128, attn_sizes[0])
        self.down2 = Down(128, 256)     # 128->256通道，尺寸再减半
        self.sa2 = SelfAttention(256, attn_sizes[1])
        self.down3 = Down(256, 256)     # 保持256通道，尺寸继续减半
        self.sa3 = SelfAttention(256, attn_sizes[2])

        # 瓶颈层（最深层特征处理）
        self.bot1 = DoubleConv(256, 512)   # 扩展到512通道
        self.bot2 = DoubleConv(512, 512)   # 保持512通道
        self.bot3 = DoubleConv(512, 256)   # 压缩回256通道

        # 解码器路径（上采样）
        self.up1 = Up(512, 128)         # 512->128通道（包含跳跃连接）
        self.sa4 = SelfAttention(128, decoder_attn_sizes[0])
        self.up2 = Up(256, 64)          # 256->64通道
        self.sa5 = SelfAttention(64, decoder_attn_sizes[1])
        self.up3 = Up(128, 64)          # 128->64通道
        self.sa6 = SelfAttention(64, decoder_attn_sizes[2])

        # 输出卷积层：1x1卷积将64通道映射到输出通道数
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        """
        生成时间步的正弦位置编码
        使用正弦和余弦函数的组合来编码时间信息
        这种编码方式能够让模型更好地理解时间步的相对关系
        参数:
            t: 时间步tensor，形状为(batch_size, 1)
            channels: 编码的维度
        返回:
            位置编码tensor，形状为(batch_size, channels)
        """
        # 计算逆频率：用于生成不同频率的正弦波
        # 10000是一个经验值，channels//2是因为正弦和余弦各占一半维度
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        # 生成正弦编码：t与不同频率相乘后取正弦
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        # 生成余弦编码：相同的频率但使用余弦函数
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        # 拼接正弦和余弦编码，形成完整的位置编码
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        """
        UNet的前向传播
        参数:
            x: 输入图像，形状为(batch_size, c_in, height, width)
            t: 时间步，形状为(batch_size,)
        返回:
            预测的噪声，形状为(batch_size, c_out, height, width)
        """
        # 将时间步转换为浮点数并添加维度
        t = t.unsqueeze(-1).type(torch.float)
        # 生成时间步的位置编码
        t = self.pos_encoding(t, self.time_dim)

        # 编码器路径：逐步下采样并提取特征
        x1 = self.inc(x)                # 输入卷积
        x2 = self.down1(x1, t)          # 第一次下采样
        x2 = self.sa1(x2)               # 自注意力增强特征
        x3 = self.down2(x2, t)          # 第二次下采样
        x3 = self.sa2(x3)               # 自注意力
        x4 = self.down3(x3, t)          # 第三次下采样
        x4 = self.sa3(x4)               # 自注意力

        # 瓶颈层：在最深层进行特征处理
        x4 = self.bot1(x4)              # 扩展通道数
        x4 = self.bot2(x4)              # 深度特征处理
        x4 = self.bot3(x4)              # 压缩通道数

        # 解码器路径：逐步上采样并结合跳跃连接
        x = self.up1(x4, x3, t)         # 第一次上采样，结合x3跳跃连接
        x = self.sa4(x)                 # 自注意力
        x = self.up2(x, x2, t)          # 第二次上采样，结合x2跳跃连接
        x = self.sa5(x)                 # 自注意力
        x = self.up3(x, x1, t)          # 第三次上采样，结合x1跳跃连接
        x = self.sa6(x)                 # 最后的自注意力

        # 输出层：生成最终的噪声预测
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    """
    条件UNet架构，支持基于类别标签的条件生成
    在标准UNet基础上添加了标签嵌入层，实现条件扩散模型
    通过将标签信息与时间步编码结合，指导生成特定类别的图像
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cpu", img_size=64):
        """
        初始化条件UNet模型
        参数:
            c_in: 输入图像通道数
            c_out: 输出图像通道数
            time_dim: 时间步嵌入的维度
            num_classes: 类别数量，如果为None则不支持条件生成
            device: 计算设备
        """
        super().__init__()
        # 存储设备信息
        self.device = device
        # 存储时间嵌入维度
        self.time_dim = time_dim
        self.img_size = img_size

        if img_size % 8 != 0:
            raise ValueError("img_size 必须能被 8 整除，以匹配当前UNet结构")

        attn_sizes = [img_size // 2, img_size // 4, img_size // 8]
        decoder_attn_sizes = [img_size // 4, img_size // 2, img_size]

        # 网络结构与标准UNet完全相同
        # 输入卷积层
        self.inc = DoubleConv(c_in, 64)

        # 编码器路径
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, attn_sizes[0])
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, attn_sizes[1])
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, attn_sizes[2])

        # 瓶颈层
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # 解码器路径
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, decoder_attn_sizes[0])
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, decoder_attn_sizes[1])
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, decoder_attn_sizes[2])

        # 输出层
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        # 条件生成的关键：标签嵌入层
        # 如果指定了类别数量，创建嵌入层将类别ID映射到时间维度
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        """
        生成时间步的正弦位置编码
        与标准UNet中的实现完全相同
        参数:
            t: 时间步tensor
            channels: 编码维度
        返回:
            位置编码tensor
        """
        # 计算逆频率
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        # 生成正弦和余弦编码
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        # 拼接形成完整编码
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        """
        条件UNet的前向传播
        参数:
            x: 输入图像，形状为(batch_size, c_in, height, width)
            t: 时间步，形状为(batch_size,)
            y: 类别标签，形状为(batch_size,)，可以为None（无条件生成）
        返回:
            预测的噪声，形状为(batch_size, c_out, height, width)
        """
        # 处理时间步：转换类型并生成位置编码
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # 条件生成的核心：将标签信息融入时间编码
        if y is not None:
            # 将类别标签通过嵌入层映射到时间维度
            # 然后与时间编码相加，实现条件控制
            # 这种简单的加法操作让模型能够同时感知时间步和类别信息
            t += self.label_emb(y)

        # 网络前向传播过程与标准UNet完全相同
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # 瓶颈层
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # 解码器路径
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        # 输出预测的噪声
        output = self.outc(x)
        return output


# 测试代码：验证模型的基本功能
if __name__ == '__main__':
    # 创建条件UNet实例（注释掉的是标准UNet）
    # net = UNet(device="cpu")
    net = UNet_conditional(num_classes=10, device="cpu")

    # 打印模型参数总数，用于了解模型规模
    print(sum([p.numel() for p in net.parameters()]))

    # 创建测试输入
    x = torch.randn(3, 3, 64, 64)  # 3张64x64的RGB图像
    t = x.new_tensor([500] * x.shape[0]).long()  # 时间步都设为500
    y = x.new_tensor([1] * x.shape[0]).long()    # 类别标签都设为1

    # 测试前向传播，打印输出形状
    print(net(x, t, y).shape)  # 应该输出torch.Size([3, 3, 64, 64])
