"""Typed training configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    path: str = "data/processed/dataset.csv"
    sep: str = ","
    id_col: str = "id"
    target_col: str = "wealth_index"
    # Village centroid coordinates (WGS84). Used for spatial blocking, never as model features.
    coord_cols: list[str] = field(default_factory=lambda: ["lon", "lat"])


@dataclass
class SplitConfig:
    scheme: str = "spatial"  # "spatial" (block-grouped) or "random"
    test_size: float = 0.3
    block_size_km: float = 5.0
    cv_folds: int = 5


@dataclass
class FeatureConfig:
    # Share of texture (TEX_*) variance kept by a PCA step; None leaves the texture columns as is.
    texture_pca: float | None = None
    corr_threshold: float | None = 0.95  # None disables the correlation filter
    top_k: int = 25
    pfi_repeats: int = 5
    pfi_estimators: int = 100


@dataclass
class TpotConfig:
    # "rf": RandomForest only, over the `model__*` entries of param_distributions (thesis setup).
    # Anything else is passed to TPOT as a preset: "linear-light", "linear", "graph-light", "graph".
    search_space: str = "rf"
    generations: int = 10
    population_size: int = 50
    early_stop: int | None = None
    max_time_mins: float | None = 60


@dataclass
class SearchConfig:
    method: str = "random"  # "random" (RandomizedSearchCV) or "tpot" (needs the tpot extra)
    n_iter: int = 30
    scoring: str = "neg_root_mean_squared_error"
    param_distributions: dict[str, list[Any]] = field(default_factory=dict)
    tpot: TpotConfig = field(default_factory=TpotConfig)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    seed: int = 42
    n_jobs: int = -1
    output_dir: str = "artifacts"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    search = dict(raw.get("search", {}))
    search["tpot"] = TpotConfig(**search.get("tpot", {}))
    return Config(
        data=DataConfig(**raw.get("data", {})),
        split=SplitConfig(**raw.get("split", {})),
        features=FeatureConfig(**raw.get("features", {})),
        search=SearchConfig(**search),
        **{k: v for k, v in raw.items() if k not in {"data", "split", "features", "search"}},
    )
