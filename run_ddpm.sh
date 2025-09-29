#!/bin/bash

# DDPM项目运行脚本
echo "🎯 启动DDPM Conditional Training环境"
echo "=" * 50

# 检查虚拟环境是否存在
if [ ! -d "ddpm_env" ]; then
    echo "❌ 虚拟环境不存在，请先运行环境配置脚本"
    exit 1
fi

# 激活虚拟环境
source ddpm_env/bin/activate

# 显示环境信息
echo "🔧 环境信息:"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

echo ""
echo "🚀 环境已就绪！可以运行以下命令:"
echo "  1. 训练模型: python ddpm_conditional.py"
echo "  2. 查看代码: ls -la"
echo "  3. 退出环境: deactivate"
echo ""

# 保持在激活的环境中
exec bash