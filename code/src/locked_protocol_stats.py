#!/usr/bin/env python3
"""Upgrade statistics from deposited File S1 tables (no checkpoints).

Writes FDR, rank-biserial, bootstrap median CIs, tissue-concordance CI,
and an exploratory eQTL permutation test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

P, U5, GAP, U3, T = 1000, 500, 20, 500, 1000
REGIONS = ["utr5", "utr3", "promoter", "terminator"]
RNG = np.random.default_rng(42)


def assign_region(pos: int) -> str:
    if 0 <= pos < P:
        return "promoter"
    if P <= pos < P + U5:
        return "utr5"
    if P + U5 <= pos < P + U5 + GAP:
        return "gap"
    if P + U5 + GAP <= pos < P + U5 + GAP + U3:
        return "utr3"
    if P + U5 + GAP + U3 <= pos < P + U5 + GAP + U3 + T:
        return "terminator"
    return "out"


def bootstrap_median_ci(x: np.ndarray, n_boot: int = 10000) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    meds = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        meds[i] = float(np.median(x[RNG.integers(0, n, n)]))
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(np.median(x)), float(lo), float(hi)


def iqr(x: np.ndarray) -> float:
    q1, q3 = np.percentile(x, [25, 75])
    return float(q3 - q1)


def region_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for r in REGIONS:
        x = df.loc[df["region"] == r, value_col].to_numpy(dtype=float)
        med, lo, hi = bootstrap_median_ci(x)
        rows.append(
            {
                "region": r,
                "n": int(len(x)),
                "mean": float(np.mean(x)),
                "sd": float(np.std(x, ddof=1)),
                "median": med,
                "median_ci95_lo": lo,
                "median_ci95_hi": hi,
                "iqr": iqr(x),
            }
        )
    return pd.DataFrame(rows)


def pairwise_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(REGIONS):
        xa = df.loc[df["region"] == a, value_col].to_numpy(dtype=float)
        for b in REGIONS[i + 1 :]:
            xb = df.loc[df["region"] == b, value_col].to_numpy(dtype=float)
            u, p = mannwhitneyu(xa, xb, alternative="two-sided")
            n1, n2 = len(xa), len(xb)
            r_rb = (2.0 * float(u)) / (n1 * n2) - 1.0
            rows.append(
                {
                    "region_a": a,
                    "region_b": b,
                    "n_a": n1,
                    "n_b": n2,
                    "median_a": float(np.median(xa)),
                    "median_b": float(np.median(xb)),
                    "U": float(u),
                    "p_two_sided": float(p),
                    "rank_biserial": float(r_rb),
                }
            )
    out = pd.DataFrame(rows)
    out["q_bh"] = false_discovery_control(out["p_two_sided"].to_numpy(), method="bh")
    return out


def load_cnn(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["region"] = df["flank_pos"].astype(int).map(assign_region)
    return df[df["region"].isin(REGIONS)].copy()


def load_agront(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["abs_delta"] = df["mean_abs_delta"]
    df["region"] = df["flank_pos"].astype(int).map(assign_region)
    return df[df["region"].isin(REGIONS)].copy()


def eqtl_permutation(eq: pd.DataFrame, bg: np.ndarray, n_perm: int = 10000) -> dict:
    x = eq["abs_delta"].to_numpy(dtype=float)
    obs = float(np.median(x))
    bg_med = float(np.median(bg))
    n = len(x)
    perm = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm[i] = float(np.median(bg[RNG.integers(0, len(bg), n)]))
    p_perm = (1.0 + np.sum(perm >= obs)) / (n_perm + 1.0)
    u, p_mw = mannwhitneyu(x, bg, alternative="greater")
    return {
        "n_eqtl": n,
        "median_eqtl": obs,
        "median_background": bg_med,
        "mwu_U": float(u),
        "mwu_p_greater": float(p_mw),
        "permutation_p_median": float(p_perm),
        "n_perm": n_perm,
        "note": "exploratory; n=17; do not use as confirmatory evidence",
    }


def tissue_ci(path: Path, n_boot: int = 10000) -> dict:
    conc = pd.read_csv(path)
    agree = conc["agree_with_avg_label"].to_numpy(dtype=float)
    n = len(agree)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = float(np.mean(agree[RNG.integers(0, n, n)]))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_tissues": n,
        "mean_agreement": float(np.mean(agree)),
        "sd": float(np.std(agree, ddof=1)),
        "median": float(np.median(agree)),
        "iqr": iqr(agree),
        "min": float(np.min(agree)),
        "max": float(np.max(agree)),
        "bootstrap_ci95_lo": float(lo),
        "bootstrap_ci95_hi": float(hi),
        "n_boot": n_boot,
        "bootstrap_unit": "tissues/samples (n=56), not genes",
    }


def fmt_dict(d: dict) -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, float):
            lines.append(f"{k} = {v:.6g}")
        else:
            lines.append(f"{k} = {v}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=None)
    args = ap.parse_args()
    tables = Path(args.tables) if args.tables else Path(__file__).resolve().parents[2] / "结果表"
    cnn = load_cnn(tables / "scores.tsv")
    agro = load_agront(tables / "agront_scores_by_site.tsv")
    eq = pd.read_csv(tables / "eqtl_scores.tsv", sep="\t")

    cnn_reg = region_table(cnn, "abs_delta")
    agro_reg = region_table(agro, "abs_delta")
    cnn_pair = pairwise_table(cnn, "abs_delta")
    agro_pair = pairwise_table(agro, "abs_delta")
    tissue = tissue_ci(tables / "tissue_label_concordance.csv")
    eqtl = eqtl_permutation(eq, cnn["abs_delta"].to_numpy(dtype=float))

    seeds = pd.read_csv(tables / "cnn_multiseed_pgb.csv")
    seed_note = {
        "cnn_n_seeds": int(len(seeds)),
        "cnn_auroc_mean": float(seeds["auroc"].mean()),
        "cnn_auroc_sd": float(seeds["auroc"].std(ddof=1)),
        "cnn_accuracy_mean": float(seeds["accuracy"].mean()),
        "cnn_accuracy_sd": float(seeds["accuracy"].std(ddof=1)),
        "agront_n_seeds": 1,
        "agront_seed": 42,
    }

    cnn_reg.to_csv(tables / "variant_region_descriptives.csv", index=False)
    agro_reg.to_csv(tables / "agront_variant_region_descriptives.csv", index=False)
    cnn_pair.to_csv(tables / "variant_region_pairwise_fdr.csv", index=False)
    agro_pair.to_csv(tables / "agront_variant_region_pairwise_fdr.csv", index=False)

    summary = []
    summary.append("=== CNN region descriptives ===")
    summary.append(cnn_reg.to_string(index=False))
    summary.append("\n=== CNN pairwise two-sided MW + BH-FDR ===")
    summary.append(cnn_pair.to_string(index=False))
    summary.append("\n=== AgroNT region descriptives ===")
    summary.append(agro_reg.to_string(index=False))
    summary.append("\n=== AgroNT pairwise two-sided MW + BH-FDR ===")
    summary.append(agro_pair.to_string(index=False))
    summary.append("\n=== Tissue-averaged vs single-tissue labels ===")
    summary.append(fmt_dict(tissue).rstrip())
    summary.append("\n=== eQTL exploratory ===")
    summary.append(fmt_dict(eqtl).rstrip())
    summary.append("\n=== Seeds ===")
    summary.append(fmt_dict(seed_note).rstrip())
    text = "\n".join(summary) + "\n"
    (tables / "locked_protocol_stats.txt").write_text(text)
    (tables / "tissue_concordance_ci.txt").write_text(fmt_dict(tissue))
    (tables / "eqtl_exploratory.txt").write_text(fmt_dict(eqtl))
    print(text)


if __name__ == "__main__":
    main()
