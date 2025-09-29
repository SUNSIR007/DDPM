#!/usr/bin/env python3
"""命令行推理脚本，用于从训练好的DDPM条件模型采样图像"""

import argparse
import json
from pathlib import Path
from typing import List

import torch

from ddpm_conditional import Diffusion
from modules import UNet_conditional
from utils import save_images


def parse_labels(label_arg: str, num_classes: int, device: torch.device) -> torch.Tensor:
    """根据命令行参数生成标签张量"""
    if label_arg.lower() in {"all", "auto"}:
        return torch.arange(num_classes, device=device)

    try:
        values: List[int] = json.loads(label_arg)
        if isinstance(values, int):
            values = [values]
        if not isinstance(values, list):
            raise ValueError
    except json.JSONDecodeError:
        values = [int(v.strip()) for v in label_arg.split(",") if v.strip()]

    labels = torch.tensor(values, device=device, dtype=torch.long)
    if (labels < 0).any() or (labels >= num_classes).any():
        raise ValueError(f"标签必须在 [0, {num_classes - 1}] 范围内: {values}")
    return labels


def resolve_device(preferred: str) -> torch.device:
    """根据优先级选择推理设备"""
    if preferred != "auto":
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="使用条件DDPM模型进行图像采样")
    parser.add_argument("--ckpt", default="models/DDPM_conditional/ema_ckpt.pt",
                        help="模型权重路径，默认使用EMA权重")
    parser.add_argument("--device", default="auto",
                        help="推理设备，默认为自动检测，可选 cpu/cuda/mps")
    parser.add_argument("--labels", default="all",
                        help="采样标签，可用 all、逗号分隔的整数，或JSON数组")
    parser.add_argument("--cfg-scale", type=float, default=3.0,
                        help="Classifier-Free Guidance 缩放系数")
    parser.add_argument("--image-size", type=int, default=32,
                        help="训练时使用的图像尺寸")
    parser.add_argument("--num-classes", type=int, default=10,
                        help="条件类别数，默认 CIFAR-10")
    parser.add_argument("--output", default="results/DDPM_conditional/inference.jpg",
                        help="输出图片路径")

    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"🎯 使用设备: {device}")

    model = UNet_conditional(num_classes=args.num_classes, device=device.type, img_size=args.image_size).to(device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {ckpt_path}")

    print(f"📂 加载权重: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    labels = parse_labels(args.labels, args.num_classes, device)
    print(f"🏷️ 采样标签: {labels.tolist()}")

    diffusion = Diffusion(img_size=args.image_size, device=device)

    with torch.no_grad():
        samples = diffusion.sample(model, n=len(labels), labels=labels, cfg_scale=args.cfg_scale)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_images(samples, str(output_path), nrow=len(labels))
    print(f"✅ 采样完成，图像已保存至 {output_path}")


if __name__ == "__main__":
    main()
