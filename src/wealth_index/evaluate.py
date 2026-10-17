"""Metrics and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: plots are written to files, never shown

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = root_mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rrmse_pct": float(rmse / np.mean(y_true) * 100),
    }


def plot_actual_vs_predicted(y_true, y_pred, path: Path, title: str) -> None:
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=12)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlabel="Actual wealth index", ylabel="Predicted wealth index", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(names, importances, path: Path, top_n: int = 15) -> None:
    order = np.argsort(importances)[::-1][:top_n][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(order) + 1.5))
    ax.barh(np.asarray(names)[order], np.asarray(importances)[order])
    ax.set(xlabel="Permutation importance (test set)", title=f"Top {len(order)} features")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
