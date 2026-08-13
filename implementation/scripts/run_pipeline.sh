#!/usr/bin/env bash
# =====================================================================
# run_pipeline.sh — 端到端流水线（AutoDL 上一键运行）
# 用法:  bash scripts/run_pipeline.sh
#
# 流程: 数据下载 → 数据集构建 → CNN 训练 → AgroNT 微调
#       → 统一基准评估 → 可解释性分析
# 提示: 首次运行会自动下载 PGB/TAIR10/AgroNT 权重（需网络与 ~10GB 磁盘）。
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
echo "工作目录: $(pwd)"

echo ""
echo "===== [1/6] 下载数据 (PGB + TAIR10) ====="
python data/download_pgb.py
python data/download_tair10.py || echo "[提示] TAIR10 下载失败可手动放置基因组/GFF，见 README。"

echo ""
echo "===== [2/6] 构建自建数据集（侧翼序列 + 标签 + 防泄漏划分） ====="
python data/build_dataset.py

echo ""
echo "===== [3/6] 1D-CNN 训练（自建 + PGB） ====="
python src/train.py --data npz --task binary || echo "[跳过] 自建数据集未就绪，可先跑 PGB"
python src/train.py --data pgb --task binary

echo ""
echo "===== [4/6] AgroNT-1B LoRA 微调（首次自动下载权重 ~4GB） ====="
python src/train_agront.py --data pgb --task binary

echo ""
echo "===== [5/6] 统一基准评估（图 3） ====="
python src/evaluate.py --data pgb --task binary --models cnn agront

echo ""
echo "===== [6/6] 可解释性分析（图 4：归因 + 区域重要性 + TF-MoDISco） ====="
python src/interpret/attribution.py
python src/interpret/region_importance.py
python src/interpret/run_modisco.py || echo "[提示] modisco 需安装；归因/区域重要性已生成。"

echo ""
echo "===== 流水线完成 ====="
echo "结果:  results/          (模型权重 / 指标 CSV / 验证结果)"
echo "       figures/          (图 3-5)"
echo "下一步: 变异效应预测（模块四）见 README「模块四」一节。"
