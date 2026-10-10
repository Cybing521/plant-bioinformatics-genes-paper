#!/usr/bin/env python3
"""House-default flowchart / architecture drawing (locked 2026-08-12).

Default visual language for study workflows and model diagrams:
  - Black borders & arrows, white / light-gray fill only
  - Times New Roman (English) or CJK-capable stack when Chinese
  - Study flow: vertical main stages + right-hand note boxes
  - Architecture: left-to-right staged columns + optional bottom strip

Usage::

    from house_flowchart import (
        apply_flowchart_style, draw_vertical_workflow, draw_architecture_columns,
    )
    apply_flowchart_style(lang="en")  # or lang="zh"
    fig, ax = draw_vertical_workflow(stages, notes, title="...", out="Fig1.png")

This module is the reusable form of the Anatomy-Prior Fig1/Fig2 pattern and the
multi-project ``generate_workflow.py`` vertical B/W style.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

Stage = Tuple[str, str]  # (title, body)
PathLike = Union[str, Path]

LIGHT_GRAY = "#F0F0F0"
BLACK = "black"
WHITE = "white"


def apply_flowchart_style(lang: str = "en", base_size: float = 9.0) -> None:
    """Configure rcParams once. lang: 'en' | 'zh'."""
    if lang.startswith("zh"):
        family = ["Times New Roman", "Songti SC", "STSong", "PingFang SC", "Heiti SC", "serif"]
    else:
        family = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif", "serif"]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": family,
            "font.size": base_size,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _box(ax, x0, y0, x1, y1, *, fc=WHITE, lw=1.2, z=2):
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=fc,
            edgecolor=BLACK,
            lw=lw,
            zorder=z,
        )
    )


def _arrow(ax, p0, p1, *, ms=11, lw=1.05, z=3):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=BLACK,
            zorder=z,
        )
    )


def _text(ax, x, y, s, *, size=8, weight="normal", style="normal", ha="center", va="center", z=4, ls=1.35):
    """Place a (possibly multiline) text block.

    ``ls`` is the matplotlib line-spacing multiple. Dense bullet lists inside a
    fixed-height box usually need a tighter value (1.15-1.25) so that a larger
    font size still fits.
    """
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        fontstyle=style,
        linespacing=ls,
        zorder=z,
        color=BLACK,
    )


def draw_vertical_workflow(
    stages: Sequence[Stage],
    notes: Optional[Sequence[str]] = None,
    *,
    title: str = "",
    footer: str = "",
    fig_w: float = 9.6,
    fig_h: Optional[float] = None,
    out: Optional[PathLike] = None,
    dpi: int = 300,
):
    """Fig1-style: vertical numbered stages + optional right-hand notes.

    stages: list of (title, multiline body)
    notes:  same length as stages, or None to omit side column
    """
    n = len(stages)
    if notes is not None and len(notes) != n:
        raise ValueError("notes length must match stages")
    if fig_h is None:
        fig_h = max(8.0, 1.6 * n + 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    has_notes = notes is not None
    x0, x1 = 0.45, (6.35 if has_notes else fig_w - 0.45)
    cx = (x0 + x1) / 2
    nx0, nx1 = 6.85, fig_w - 0.35

    if title:
        _text(ax, fig_w / 2, fig_h - 0.28, title, size=11, weight="bold", va="top")

    top = fig_h - (0.55 if title else 0.35)
    bottom = 0.42 if footer else 0.25
    usable = top - bottom
    box_h = min(1.35, (usable - 0.15 * max(n - 1, 0)) / n)
    gap = (usable - n * box_h) / max(n - 1, 1) if n > 1 else 0.0

    y = top
    for i, (stitle, body) in enumerate(stages):
        y1 = y
        y0 = y - box_h
        _box(ax, x0, y0, x1, y1)
        _text(ax, cx, y1 - 0.18, stitle, size=9.5, weight="bold", va="top")
        _text(ax, cx, y1 - 0.42, body, size=8.0, va="top")
        if has_notes:
            _box(ax, nx0, y0 + 0.15, nx1, y1 - 0.15, lw=1.0)
            _text(ax, (nx0 + nx1) / 2, (y0 + y1) / 2, notes[i], size=7.2)
            _arrow(ax, (x1, (y0 + y1) / 2), (nx0, (y0 + y1) / 2))
        if i < n - 1:
            _arrow(ax, (cx, y0), (cx, y0 - gap))
        y = y0 - gap

    if footer:
        _text(ax, fig_w / 2, 0.14, footer, size=7.5, style="italic", va="bottom")

    if out is not None:
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=WHITE)
    return fig, ax


def draw_architecture_columns(
    columns: Sequence[dict],
    *,
    title: str = "",
    subtitle: str = "",
    bottom_strip: str = "",
    fig_w: float = 11.0,
    fig_h: float = 6.6,
    out: Optional[PathLike] = None,
    dpi: int = 300,
):
    """Fig2-style: left-to-right architecture columns.

    Each column dict::
        {
          "label": "2. Backbone",          # stage header above column
          "title": "ResNet-50",            # bold in-box title
          "body": "ImageNet init\\n...",  # multiline
          "fill": False,                  # optional light-gray panel
          "width": 1.4,                   # relative width weight
          "items": [("C2", "256-d"), ...], # optional inner boxes
        }
    """
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    if title:
        _text(ax, fig_w / 2, fig_h - 0.22, title, size=11, weight="bold", va="top")
    if subtitle:
        _text(ax, fig_w / 2, fig_h - 0.48, subtitle, size=8.5, style="italic", va="top")

    margin_l, margin_r = 0.25, 0.25
    gap = 0.20
    y_top = fig_h - (0.85 if title or subtitle else 0.35)
    y_bot = 1.15 if bottom_strip else 0.35
    usable_w = fig_w - margin_l - margin_r - gap * (len(columns) - 1)
    weights = [float(c.get("width", 1.0)) for c in columns]
    wsum = sum(weights) or 1.0
    widths = [usable_w * w / wsum for w in weights]

    x = margin_l
    centers = []
    for col, w in zip(columns, widths):
        x0, x1 = x, x + w
        centers.append((x0 + x1) / 2)
        if col.get("label"):
            _text(ax, (x0 + x1) / 2, y_top + 0.22, col["label"], size=8, weight="bold")
        fc = LIGHT_GRAY if col.get("fill") else WHITE
        _box(ax, x0, y_bot, x1, y_top, fc=fc)
        ty = y_top - 0.28
        if col.get("title"):
            _text(ax, (x0 + x1) / 2, ty, col["title"], size=9, weight="bold")
            ty -= 0.32
        if col.get("body"):
            _text(ax, (x0 + x1) / 2, (y_bot + ty) / 2, col["body"], size=7.5)
        items = col.get("items") or []
        if items:
            ih = 0.50
            total_h = len(items) * ih + (len(items) - 1) * 0.12
            yy = y_top - 0.85
            # center item stack
            start = y_bot + (y_top - y_bot - total_h) / 2 + total_h - ih
            yy = start
            for name, detail in items:
                _box(ax, x0 + 0.12, yy, x1 - 0.12, yy + ih)
                _text(ax, (x0 + x1) / 2, yy + ih / 2, f"{name}  ·  {detail}", size=7.5)
                yy -= ih + 0.12
        x = x1 + gap

    for i in range(len(centers) - 1):
        # arrow between columns at mid height
        mid_y = (y_top + y_bot) / 2
        left = margin_l + sum(widths[: i + 1]) + gap * i
        right = left + gap
        # use column boundaries more carefully
        x_right_of_i = margin_l + sum(widths[: i + 1]) + gap * i
        x_left_of_next = x_right_of_i + gap
        _arrow(ax, (x_right_of_i, mid_y), (x_left_of_next, mid_y))

    if bottom_strip:
        _box(ax, margin_l, 0.22, fig_w - margin_r, 0.95)
        _text(ax, fig_w / 2, 0.58, bottom_strip, size=7.5)

    if out is not None:
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=WHITE)
    return fig, ax


if __name__ == "__main__":
    # Minimal self-check (writes /tmp if run directly)
    apply_flowchart_style("en")
    demo_stages = [
        ("1. Data", "Public cohort\nn = example"),
        ("2. Model", "Backbone + heads"),
        ("3. Eval", "AUC · CI · tests"),
    ]
    demo_notes = ["No PHI", "Fixed split", "Report seeds"]
    out = Path("/tmp/house_flowchart_demo.png")
    draw_vertical_workflow(demo_stages, demo_notes, title="Demo workflow", out=out)
    print("wrote", out)
