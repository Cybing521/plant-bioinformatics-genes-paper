#!/usr/bin/env python3
"""TF-MoDISco 模体发现 + TOMTOM 与已知模体比对（模块三第二步）。

用法:
  python src/interpret/run_modisco.py --config configs/config.yaml
依赖: modisco-lite CLI（setup_env.sh 已装）
产物:
  results/interpret/modisco_results.h5
  results/interpret/modisco_report/    (含 motifs.txt, TOMTOM 结果)
说明:
  - 若 TOMTOM 未安装（MEME suite），脚本会提示手动安装:
      conda install -c bioconda meme  (或下载 meme-suite 5.x)
  - JASPAR 植物模体库需下载后放到 config 指定路径:
      wget https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_plants_non-redundant_pfm_jaspar.txt
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # -> src/

import numpy as np

from utils import setup_logging

logger = logging.getLogger("plantdl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--seqlets", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    itp = cfg["interpret"]
    out = itp["out_dir"]
    os.makedirs(out, exist_ok=True)

    ohe = np.load(os.path.join(out, "ohe.npz"))["arr_0"]            # (N,4,L)
    attr = np.load(os.path.join(out, "attributions.npz"))["arr_0"]
    seqlets = args.seqlets or itp["modisco_seqlets"]
    window = args.window or itp["modisco_window"]

    # 取归因绝对值最大的 seqlets 条序列（减轻计算量）
    score = np.abs(attr).sum(axis=1).sum(axis=1)
    top = np.argsort(score)[-seqlets:]
    ohe_top = np.ascontiguousarray(ohe[top])
    attr_top = np.ascontiguousarray(attr[top])

    ohe_path = os.path.join(out, "ohe_top.npz")
    attr_path = os.path.join(out, "attributions_top.npz")
    np.savez(ohe_path, arr_0=ohe_top)
    np.savez(attr_path, arr_0=attr_top)

    # modisco CLI 在 python 同目录（模块名 modiscolite，入口点 modisco）
    modisco_bin = os.path.join(os.path.dirname(sys.executable), "modisco")
    if not os.path.exists(modisco_bin):
        sys.exit(f"未找到 modisco CLI: {modisco_bin}\n请确认 modisco-lite 已安装")

    h5 = os.path.join(out, "modisco_results.h5")
    cmd = [modisco_bin, "motifs", "-s", ohe_path, "-a", attr_path,
           "-n", str(len(ohe_top)), "-w", str(window), "-o", h5]
    logger.info("运行: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("modisco 运行失败:\n%s\n%s", r.stdout, r.stderr)
        sys.exit(1)

    report_dir = os.path.join(out, "modisco_report")
    os.makedirs(report_dir, exist_ok=True)
    rep = [modisco_bin, "report", "-i", h5, "-o", report_dir,
           "-s", report_dir]
    # TOMTOM 需 MEME suite 的 tomtom 二进制 + JASPAR 库，缺一即跳过
    tomtom = shutil.which("tomtom") or os.path.join(os.path.dirname(sys.executable), "tomtom")
    if os.path.exists(itp["jaspar_db"]) and os.path.exists(tomtom):
        rep += ["-m", itp["jaspar_db"], "--tomtom"]
    else:
        logger.warning("跳过 TOMTOM：%s", "缺JASPAR库" if not os.path.exists(itp["jaspar_db"]) else "缺tomtom二进制")
    logger.info("运行: %s", " ".join(rep))
    r2 = subprocess.run(rep, capture_output=True, text=True)
    if r2.returncode != 0:
        logger.warning("modisco report 输出: %s", r2.stderr[-2000:])
    logger.info("完成 -> %s / %s", h5, report_dir)


if __name__ == "__main__":
    main()
