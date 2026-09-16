"""通用工具：设备、随机种子、指标计算。"""
import logging
import random
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("plantdl")


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_reg: np.ndarray | None = None,
                    y_reg_pred: np.ndarray | None = None) -> dict:
    """分类指标或回归指标。y_prob 为类别 1 概率。"""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        brier_score_loss, f1_score, matthews_corrcoef, mean_squared_error,
        r2_score, roc_auc_score,
    )
    if y_reg is not None and y_reg_pred is not None:
        y_reg = np.asarray(y_reg)
        y_reg_pred = np.asarray(y_reg_pred)
        out = {"r2": float(r2_score(y_reg, y_reg_pred)),
               "mse": float(mean_squared_error(y_reg, y_reg_pred))}
        if y_reg.ndim == 2:
            from scipy.stats import pearsonr, spearmanr
            pears, spears = [], []
            for j in range(y_reg.shape[1]):
                if np.std(y_reg[:, j]) == 0 or np.std(y_reg_pred[:, j]) == 0:
                    continue
                pears.append(pearsonr(y_reg[:, j], y_reg_pred[:, j])[0])
                spears.append(spearmanr(y_reg[:, j], y_reg_pred[:, j])[0])
            if pears:
                out["pearson_mean"] = float(np.mean(pears))
                out["spearman_mean"] = float(np.mean(spears))
        return out

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    prev = float(np.mean(y_true))
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "prevalence": prev,
        "auprc_baseline": prev,
    }
    if len(np.unique(y_true)) > 1:
        m["auroc"] = float(roc_auc_score(y_true, y_prob))
        m["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        m["auroc"] = m["auprc"] = float("nan")
    return m


def save_checkpoint(state: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("已保存模型 -> %s", path)
