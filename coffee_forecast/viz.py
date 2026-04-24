"""
Reusable plotting helpers.

Today's notebook only needs two: the raw series preview and the
forecast-origin coverage grid. More will land here as later notebooks
come online.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import get_forecast_origins
from .config import CONTEXT_LEN, FIG_DIR, HORIZON


def _style(ax: plt.Axes, alpha: float = 0.15) -> None:
    """Consistent spine / grid styling used across the paper."""
    ax.grid(True, alpha=alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_price_series(
    df: pd.DataFrame,
    title: str = "ICE Coffee C Futures - Daily Close (1994-2026)",
    save_path: Union[str, Path, None] = None,
    figsize: tuple = (14, 4),
    dpi: int = 180,
) -> plt.Figure:
    """Clean line plot of the full price series."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(df["ds"], df["y"], color="#4C72B0", linewidth=0.8, alpha=0.9)
    ax.set_title(title, fontsize=13, weight="semibold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price (cents/lb)", fontsize=11)
    _style(ax)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_forecast_origin_coverage(
    df: pd.DataFrame,
    scales: List[int],
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
    save_path: Union[str, Path, None] = None,
    figsize: tuple = (14, 6),
    dpi: int = 180,
) -> plt.Figure:
    """
    2x2 grid showing where the forecast origins sit for each scale.

    Visualizing this is important for readers of the paper: it makes
    clear that the 60-origin run spans 30+ years of regime shifts,
    not just the tail.
    """
    n = len(scales)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten() if n > 1 else [axes]

    for ax, scale in zip(axes, scales):
        origins = get_forecast_origins(df, scale, context_len, horizon)

        ax.plot(df["ds"], df["y"],
                color="#6BAED6", alpha=0.8, linewidth=0.8,
                label="Coffee C Futures Price")

        for i, a in enumerate(origins, start=1):
            start_ds = df["ds"].iloc[a]
            end_ds   = df["ds"].iloc[a + horizon - 1]
            y_level  = df["y"].iloc[a]
            ax.plot([start_ds, end_ds], [y_level, y_level],
                    color="#FFA500", linewidth=2.5,
                    label="Forecast Horizon" if i == 1 else None)
            ax.scatter([start_ds], [y_level],
                       color="#4C72B0", s=15, zorder=5,
                       label="Forecast Origin" if i == 1 else None)

        ax.set_title(
            f"{scale} Forecast Origin{'s' if scale > 1 else ''} "
            f"with {horizon}-Day Horizon",
            fontsize=11, weight="semibold",
        )
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Price (cents/lb)", fontsize=9)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        _style(ax, alpha=0.12)

    # Hide any unused axes (when len(scales) is odd).
    for ax in axes[len(scales):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Rolling-Window Backtest Design: {', '.join(str(s) for s in scales)} "
        f"Forecast Origins",
        fontsize=13, weight="semibold", y=1.01,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
