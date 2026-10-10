#!/usr/bin/env bash
# =====================================================================
# setup_env.sh — AutoDL / 任意 Linux GPU 服务器一键环境安装
# 用法： bash setup_env.sh
#
# 说明：
#   - AutoDL PyTorch 基础镜像已自带 torch+CUDA，本脚本只补装缺失依赖，
#     避免重复安装 torch 造成的版本冲突。
#   - 若在全新机器上使用，取消下方 "全新环境" 区块注释安装 torch。
# =====================================================================
set -euo pipefail

PYTHON_BIN="$(command -v python3 || command -v python)"
echo ">>> 使用 Python: ${PYTHON_BIN}"

# --- 检查 torch/CUDA 是否可用（AutoDL 镜像一般已就绪）---
if "${PYTHON_BIN}" -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
  echo ">>> 检测到已可用的 torch+CUDA，跳过 torch 安装"
  TORCH_OK=1
else
  echo ">>> 未检测到 torch，进入全新环境安装分支"
  TORCH_OK=0
fi

# --- 全新环境：安装 CUDA 版 torch（按需取消注释）---
if [ "${TORCH_OK}" = "0" ]; then
  echo ">>> 安装 torch 2.4.1 (cu121)。如机器 GPU 不同，请替换为对应版本。"
  # pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
  echo ">>> [提示] 请先 conda create -n plantdl python=3.10 && conda activate plantdl，"
  echo "         再按需安装 torch；随后继续执行本脚本剩余部分。"
fi

# --- 安装其余依赖 ---
"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -r requirements.txt

# --- 校验 ---
"${PYTHON_BIN}" - <<'EOF'
import torch
print(f"torch      : {torch.__version__}")
print(f"cuda avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu        : {torch.cuda.get_device_name(0)}")
    print(f"vram       : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
import transformers, peft, datasets, shap, modisco
print("transformers:", transformers.__version__, "| peft:", peft.__version__)
print("datasets:", datasets.__version__, "| shap:", shap.__version__)
print("modisco-lite OK")
EOF
echo ">>> 环境就绪 ✔"
