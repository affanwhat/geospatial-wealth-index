"""Prediction maps: join `wi predict` output to the village polygons and draw a choropleth."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: maps are written to files, never shown

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger(__name__)


def join_predictions(
    villages: gpd.GeoDataFrame, predictions: pd.DataFrame, *, id_col: str, value_col: str
) -> gpd.GeoDataFrame:
    """Villages (indexed by id) with the prediction column; villages without one get NaN."""
    if value_col not in predictions:
        raise ValueError(f"predictions have no column '{value_col}'")
    values = predictions.set_index(predictions[id_col].astype(str).str.strip())[value_col]
    dupes = values.index[values.index.duplicated()].unique()
    if len(dupes):
        raise ValueError(f"duplicate prediction ids: {list(dupes[:10])}")
    unknown = values.index.difference(villages.index.astype(str))
    if len(unknown):
        log.warning("%d prediction ids not in the boundaries: %s", len(unknown), list(unknown[:10]))
    out = villages.copy()
    out[value_col] = values.reindex(out.index.astype(str)).to_numpy()
    if out[value_col].isna().all():
        raise ValueError("no prediction id matches a village id")
    return out


def plot_choropleth(
    gdf: gpd.GeoDataFrame, value_col: str, path: Path, title: str, cmap: str = "viridis"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(
        column=value_col,
        ax=ax,
        cmap=cmap,
        linewidth=0.1,
        edgecolor="white",
        legend=True,
        legend_kwds={"label": "Predicted wealth index", "shrink": 0.6},
        missing_kwds={"color": "lightgrey", "label": "No prediction"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_map(
    villages: gpd.GeoDataFrame,
    predictions_path: Path,
    output: Path,
    *,
    id_col: str = "id",
    value_col: str = "wealth_index_pred",
    vector: Path | None = None,
) -> gpd.GeoDataFrame:
    """Write the choropleth PNG (and optionally a GeoPackage of the joined polygons)."""
    predictions = pd.read_csv(predictions_path, dtype={id_col: str})
    gdf = join_predictions(villages, predictions, id_col=id_col, value_col=value_col)
    output.parent.mkdir(parents=True, exist_ok=True)
    n = int(gdf[value_col].notna().sum())
    plot_choropleth(gdf, value_col, output, f"Predicted wealth index, {n} villages")
    if vector is not None:
        vector.parent.mkdir(parents=True, exist_ok=True)
        gdf.reset_index().to_file(vector, driver="GPKG")
    log.info("Mapped %d of %d villages", n, len(gdf))
    return gdf
