#!/bin/bash
# CS503 Transformer Homework - 集群环境检查脚本
# 用法: bash check_env.sh  或  srun --partition=gpu --gres=gpu:1 bash check_env.sh (在 GPU 节点上检查)

echo "========== 1. Python 版本 =========="
python3 --version

echo ""
echo "========== 2. 已加载的模块 =========="
module list 2>/dev/null || echo "(无 module 或未加载)"

echo ""
echo "========== 3. GPU 状态 (需在 GPU 节点上才有输出) =========="
nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用 (登录节点通常无 GPU，需 srun/sbatch 到 gpu 分区)"

echo ""
echo "========== 4. 依赖包检查 =========="
python3 -c "
packages = ['torch', 'torchvision', 'einops', 'matplotlib', 'tqdm']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', '?')
        print(f'  ✓ {pkg}: {ver}')
    except ImportError:
        print(f'  ✗ {pkg}: 未安装')

import sys
if 'torch' in sys.modules:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f'  CUDA 可用: {cuda_ok}')
    if cuda_ok:
        print(f'  GPU 设备: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "========== 5. 集群可用模块 (Python/CUDA) =========="
echo "  module spider python  # 查看 Python 模块"
echo "  module spider cuda   # 查看 CUDA 模块"
echo "  module load python/3.10.4  # 加载 Python"
echo "  module load cuda/11.8.0    # 加载 CUDA (安装 PyTorch 时可能需要)"
