#!/usr/bin/env python3
"""Publication matplotlib helpers distilled from figures4papers (Chen Liu).

Implements the contracts in references/api.md so agents generate stable
publication figures instead of free-form rcParams each time.

Usage (in a figure script)::

    from publication_style import (
        PALETTE, FigureStyle, apply_publication_style,
        create_subplots, make_grouped_bar, finalize_figure, qa_before_save,
    )
    apply_publication_style(FigureStyle(font_size=24, axes_linewidth=3))
    fig, axes = create_subplots(1, 1, figsize=(12, 4))
    ...
    qa_before_save(fig)   # optional deterministic check
    finalize_figure(fig, "figures/comparison", formats=["png", "pdf"], dpi=300)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral"],
]

# Omics / bioinformatics composites keep the same semantic PALETTE.
# Use FigureStyle(font_size=8, axes_linewidth=0.8) for dense journal panels.
OMICS_UP = PALETTE["red_strong"]
OMICS_DOWN = PALETTE["blue_main"]
OMICS_NS = PALETTE["neutral"]
OMICS_DIVERGING_HEX = (PALETTE["blue_main"], "#FFFFFF", PALETTE["red_strong"])
OMICS_SEQUENTIAL_HEX = ("#F7F7F7", PALETTE["red_2"], PALETTE["red_strong"])
# Magnitude ramps used by Nature Sustainability–style maps (still PALETTE-rooted).
OMICS_MAGNITUDE_HEX = ("#FFFFFF", PALETTE["red_1"], PALETTE["red_2"], PALETTE["red_strong"])
OMICS_TEMPORAL_HEX = (
    PALETTE["blue_main"],
    PALETTE["blue_secondary"],
    PALETTE["teal"],
    PALETTE["green_1"],
)


def omics_diverging_cmap(name: str = "omics_rdbu"):
    """Blue–white–red diverging map from PALETTE (heatmaps, volcano, UMAP scores)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, OMICS_DIVERGING_HEX)


def omics_sequential_cmap(name: str = "omics_reds"):
    """White–red sequential map from PALETTE (NES heatmaps, abundance)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, OMICS_SEQUENTIAL_HEX)


def omics_magnitude_cmap(name: str = "omics_magnitude"):
    """White–pink–red magnitude ramp for choropleth / hotspot maps."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, OMICS_MAGNITUDE_HEX)


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    # Prefer portable stack; CJK-aware callers should prepend a CJK face.
    font_family: tuple[str, ...] = (
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "PingFang SC",
        "Heiti SC",
        "Noto Sans CJK SC",
        "sans-serif",
    )
    cjk: bool = False


def apply_publication_style(style: Optional[FigureStyle] = None) -> None:
    """Configure matplotlib rcParams once before creating figures."""
    import matplotlib as mpl

    style = style or FigureStyle()
    families = list(style.font_family)
    if style.cjk:
        # Put CJK-capable faces first when Chinese labels are expected.
        for face in ("PingFang SC", "Heiti SC", "Songti SC", "Noto Sans CJK SC", "Arial Unicode MS"):
            if face not in families:
                families.insert(0, face)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": families,
            "font.size": style.font_size,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": style.axes_linewidth,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": bool(style.use_tex),
            "axes.unicode_minus": False,
        }
    )


def create_subplots(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    """Return (fig, axes) with axes always a 1D numpy array of Axes."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    axes_arr = np.atleast_1d(axes).ravel()
    return fig, axes_arr


def make_grouped_bar(
    ax,
    categories: Sequence[str],
    series: Sequence[Sequence[float]],
    labels: Sequence[str],
    ylabel: str = "Value",
    colors: Optional[Sequence[str]] = None,
    annotate: bool = False,
    bar_width: Optional[float] = None,
    edgecolor: str = "black",
    linewidth: float = 1.5,
):
    """Grouped bars. series[i] aligns with labels[i]; each length == len(categories)."""
    cats = list(categories)
    n_cat = len(cats)
    n_ser = len(series)
    if n_ser != len(labels):
        raise ValueError("len(series) must equal len(labels)")
    arrs = [np.asarray(s, dtype=float) for s in series]
    for a in arrs:
        if a.ndim != 1 or len(a) != n_cat:
            raise ValueError("each series must be 1D with length == len(categories)")

    colors = list(colors) if colors is not None else DEFAULT_COLORS[:n_ser]
    if len(colors) < n_ser:
        raise ValueError("not enough colors for series")

    x = np.arange(n_cat, dtype=float)
    width = bar_width if bar_width is not None else min(0.8 / max(n_ser, 1), 0.35)
    offsets = (np.arange(n_ser) - (n_ser - 1) / 2.0) * width

    last_bars = None
    for i, (vals, lab, col) in enumerate(zip(arrs, labels, colors)):
        last_bars = ax.bar(
            x + offsets[i],
            vals,
            width=width * 0.95,
            label=lab,
            color=col,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        if annotate:
            annotate_bars(ax, last_bars)

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel(ylabel)
    return last_bars


def annotate_bars(ax, bars, fmt: str = "{:.2f}", fontsize: int = 10, padding: float = 3):
    """Print values above each bar in a BarContainer."""
    for rect in bars:
        height = rect.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def make_trend(
    ax,
    x,
    y_series: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Optional[Sequence[str]] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    show_shadow: bool = False,
    linewidth: float = 2.5,
):
    """Multi-line trend; optional simple rolling-std shadow when show_shadow=True."""
    x = np.asarray(x, dtype=float)
    colors = list(colors) if colors is not None else DEFAULT_COLORS[: len(y_series)]
    if len(y_series) != len(labels):
        raise ValueError("len(y_series) must equal len(labels)")
    for y, lab, col in zip(y_series, labels, colors):
        yy = np.asarray(y, dtype=float)
        if len(yy) != len(x):
            raise ValueError("each y series must match len(x)")
        ax.plot(x, yy, label=lab, color=col, linewidth=linewidth)
        if show_shadow and len(yy) >= 5:
            # Cheap visual uncertainty band (not a statistical CI).
            pad = np.nanstd(yy) * 0.15
            ax.fill_between(x, yy - pad, yy + pad, color=col, alpha=0.15, linewidth=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    return ax


def make_heatmap(
    ax,
    matrix,
    x_labels=None,
    y_labels=None,
    cmap: str = "magma",
    cbar_label: Optional[str] = None,
    annotate: bool = False,
):
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D")
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(list(x_labels), rotation=45, ha="right")
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(list(y_labels))
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    return im


def make_scatter(ax, x, y, label=None, color=None, size: float = 50, alpha: float = 0.7):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    return ax.scatter(
        x,
        y,
        s=size,
        alpha=alpha,
        label=label,
        color=color or PALETTE["blue_main"],
        edgecolors="white",
        linewidths=0.4,
    )


def make_radar(
    ax,
    categories: Sequence[str],
    series: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Optional[Sequence[str]] = None,
    fill_alpha: float = 0.08,
    rmin: float = 0.0,
    rmax: Optional[float] = None,
    linewidth: float = 2.0,
):
    """Radar / spider chart. `ax` must already be polar (figures4papers VIGIL pattern)."""
    cats = list(categories)
    n = len(cats)
    if n < 3:
        raise ValueError("radar needs at least 3 categories")
    if len(series) != len(labels):
        raise ValueError("len(series) must equal len(labels)")
    arrs = [np.asarray(s, dtype=float) for s in series]
    for a in arrs:
        if a.ndim != 1 or len(a) != n:
            raise ValueError("each series must be 1D with length == len(categories)")
    colors = list(colors) if colors is not None else DEFAULT_COLORS[: len(arrs)]
    if len(colors) < len(arrs):
        raise ValueError("not enough colors for series")

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles_c = np.append(angles, angles[0])
    ymax = float(rmax) if rmax is not None else max(float(np.nanmax(a)) for a in arrs)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(rmin, ymax)
    ax.set_xticks(angles)
    ax.set_xticklabels(list(cats))
    for vals, lab, col in zip(arrs, labels, colors):
        closed = np.append(vals, vals[0])
        ax.plot(angles_c, closed, color=col, linewidth=linewidth, label=lab)
        ax.fill(angles_c, closed, color=col, alpha=fill_alpha)
        ax.scatter(angles, vals, color=col, s=18, zorder=5, edgecolors="none")
    return ax


def make_sphere_illustration(
    ax,
    light_dir: Sequence[float] = (-0.5, 0.5, 0.8),
    resolution: int = 128,
    alpha: float = 0.6,
    cmap: str = "Blues",
):
    """Shaded 2D disk that reads as a 3D sphere (figures4papers Dispersion geometry, simplified)."""
    light = np.asarray(light_dir, dtype=float)
    nrm = np.linalg.norm(light)
    if nrm <= 0:
        raise ValueError("light_dir must be non-zero")
    light = light / nrm
    lin = np.linspace(-1.0, 1.0, int(resolution))
    xx, yy = np.meshgrid(lin, lin)
    zz2 = 1.0 - xx**2 - yy**2
    mask = zz2 >= 0
    zz = np.zeros_like(xx)
    zz[mask] = np.sqrt(zz2[mask])
    shade = np.clip(xx * light[0] + yy * light[1] + zz * light[2], 0.0, 1.0)
    img = np.full(xx.shape, np.nan, dtype=float)
    img[mask] = shade[mask]
    ax.imshow(img, origin="lower", cmap=cmap, alpha=alpha, extent=(-1, 1, -1, 1))
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def legend_only_axis(ax, handles=None, labels=None, **legend_kwargs):
    """Turn an axes into a dedicated legend panel (figures4papers pattern)."""
    ax.set_axis_off()
    if handles is not None and labels is not None:
        ax.legend(handles, labels, loc="center", **legend_kwargs)
    return ax


def finalize_figure(
    fig,
    out_path: Union[str, Path],
    formats: Optional[Iterable[str]] = None,
    dpi: int = 300,
    close: bool = True,
    pad: float = 0.05,
    tight: bool = True,
    **savefig_kwargs,
):
    """Save figure to one or more formats; create parent dirs. Returns list[Path]."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    if formats is None:
        suffix = out_path.suffix.lstrip(".").lower()
        formats = [suffix] if suffix else ["pdf", "png"]
    formats = [f.lstrip(".").lower() for f in formats]
    supported = {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}
    bad = [f for f in formats if f not in supported]
    if bad:
        raise ValueError(f"unsupported formats: {bad}")

    if tight:
        try:
            fig.tight_layout(pad=max(pad * 20, 0.5))  # pad~0.05 → layout pad ~1
        except Exception:
            pass

    saved: list[Path] = []
    stem_parent = out_path.parent
    stem = out_path.stem if out_path.suffix else out_path.name
    stem_parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        target = stem_parent / f"{stem}.{fmt}"
        fig.savefig(
            target,
            dpi=dpi,
            bbox_inches=savefig_kwargs.pop("bbox_inches", "tight"),
            pad_inches=pad,
            facecolor=savefig_kwargs.pop("facecolor", "white"),
            edgecolor=savefig_kwargs.pop("edgecolor", "none"),
            **savefig_kwargs,
        )
        saved.append(target)

    if close:
        plt.close(fig)
    return saved


def qa_before_save(fig, min_fontsize: float = 6.0, require_cjk_font: bool = False):
    """Run local figure_qa (same directory) if available; else minimal font-size scan."""
    try:
        import importlib.util
        import sys

        qa_path = Path(__file__).resolve().parent / "figure_qa.py"
        if qa_path.is_file():
            spec = importlib.util.spec_from_file_location("scientific_figures_figure_qa", qa_path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            key = "scientific_figures_figure_qa"
            if key not in sys.modules:
                spec.loader.exec_module(mod)
                sys.modules[key] = mod
            else:
                mod = sys.modules[key]
            return mod.qa_check(fig)
    except Exception as exc:  # noqa: BLE001 — soft fallback
        return {"ok": True, "defects": [], "warning": f"figure_qa unavailable: {exc}"}

    # Minimal fallback: font-size scan only
    defects = []
    for ax in fig.axes:
        for t in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            if t is None:
                continue
            try:
                fs = float(t.get_fontsize())
            except Exception:
                continue
            if fs < min_fontsize:
                defects.append(
                    {
                        "severity": "medium",
                        "location": "text",
                        "issue": f"fontsize {fs} < {min_fontsize}",
                        "fix": "increase base FigureStyle.font_size or annotation size",
                    }
                )
    return {"ok": not defects, "defects": defects}


if __name__ == "__main__":
    # Smoke: grouped bar + export to /tmp
    import matplotlib

    matplotlib.use("Agg")
    apply_publication_style(FigureStyle(font_size=14, axes_linewidth=2))
    fig, axes = create_subplots(1, 1, figsize=(8, 3.5))
    make_grouped_bar(
        axes[0],
        ["A", "B", "C"],
        [[0.9, 0.85, 0.88], [0.8, 0.82, 0.79]],
        ["Ours", "Baseline"],
        ylabel="Score",
        annotate=True,
    )
    axes[0].set_ylim(0.7, 1.0)
    paths = finalize_figure(fig, "/tmp/scientific_figures_smoke", formats=["png"], dpi=150)
    print("wrote", paths)
