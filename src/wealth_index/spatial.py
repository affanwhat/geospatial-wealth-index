"""Spatial blocking for honest validation of spatially autocorrelated data.

Neighbouring villages share similar wealth, so a random split leaks near-duplicates
into the test set. Grouping villages into square blocks and splitting by block keeps
each block entirely on one side of a split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQUATOR = 111.320


def spatial_blocks(coords: pd.DataFrame, block_size_km: float) -> np.ndarray:
    """Assign each (lon, lat) point to a square block of `block_size_km` and return block ids."""
    lon = coords.iloc[:, 0].to_numpy(dtype=float)
    lat = coords.iloc[:, 1].to_numpy(dtype=float)
    km_per_deg_lon = KM_PER_DEG_LON_EQUATOR * np.cos(np.deg2rad(lat.mean()))
    col = np.floor(lon * km_per_deg_lon / block_size_km).astype(int)
    row = np.floor(lat * KM_PER_DEG_LAT / block_size_km).astype(int)
    _, groups = np.unique(np.stack([col, row], axis=1), axis=0, return_inverse=True)
    return groups.ravel()


def make_cv(scheme: str, n_splits: int, seed: int) -> KFold | GroupKFold:
    if scheme == "random":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if scheme == "spatial":
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"Unknown split scheme: {scheme!r}")


def holdout_split(
    n: int, scheme: str, groups: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx). For the spatial scheme `test_size` is a fraction of blocks."""
    idx = np.arange(n)
    if scheme == "random":
        return train_test_split(idx, test_size=test_size, random_state=seed)
    if scheme == "spatial":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(splitter.split(idx, groups=groups))
    raise ValueError(f"Unknown split scheme: {scheme!r}")
