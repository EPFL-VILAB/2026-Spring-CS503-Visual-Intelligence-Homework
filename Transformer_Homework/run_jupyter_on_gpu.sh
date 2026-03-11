#!/bin/bash
# 在计算节点上启动 Jupyter，供 Cursor 连接
# 用法：在 srun 会话中运行此脚本

set -e
cd "$(dirname "$0")"
source venv/bin/activate

# 使用 8889 避免与本地 Docker 等占用 8888 的服务冲突
PORT=8889
echo "=========================================="
echo "在计算节点 $(hostname) 上启动 Jupyter"
echo "=========================================="
echo ""
echo "在 Cursor (Remote-SSH) 中连接："
echo "  Select Kernel -> Existing Jupyter Server -> http://$(hostname):${PORT}"
echo "=========================================="
echo ""

# --ip=0.0.0.0 让 Jupyter 监听所有网卡，以便登录节点能连到计算节点
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0
