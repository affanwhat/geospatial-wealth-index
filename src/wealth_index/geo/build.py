"""Stage runner with manifests and caching, and the final join into features.csv."""

from __future__ import annotations

import glob
import importlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from wealth_index.data import file_sha256
from wealth_index.geo.boundaries import read_villages
from wealth_index.geo.catalogue import catalogue
from wealth_index.geo.config import FeatureBuildConfig
from wealth_index.train import _git_sha

log = logging.getLogger(__name__)

STAGES = ["boundaries", "osm", "s2", "texture", "ntl", "green"]


class MissingInputError(FileNotFoundError):
    """A stage's raw inputs (or an earlier stage's outputs) are not there yet."""


def _globbed(pattern: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(pattern, recursive=True))


def stage_inputs(cfg: FeatureBuildConfig, stage: str) -> list[Path]:
    villages = cfg.interim / "boundaries.parquet"
    rasters = cfg.interim / "rasters"
    match stage:
        case "boundaries":
            return [Path(cfg.paths.boundaries)]
        case "osm":
            return [Path(cfg.paths.osm_pbf), villages]
        case "s2" | "texture":
            return [*_globbed(cfg.paths.s2_composite), villages]
        case "ntl":
            return [*_globbed(cfg.paths.sdgsat), villages]
        case "green":
            ntl = [rasters / "ntl.tif"] if cfg.green.settlement == "ntl" else []
            return [rasters / "s2_indices.tif", *ntl, villages]
    raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")


def stage_params(cfg: FeatureBuildConfig, stage: str) -> dict:
    c = cfg.to_dict()
    common = {"crs": cfg.crs, "id_col": cfg.id_col, "feature_set": cfg.feature_set}
    s2 = {k: v for k, v in c["s2"].items() if k != "gee"}  # export settings live with the export
    return (
        common
        | {
            "boundaries": {},
            "osm": {"osm": c["osm"], "grid": c["grid"]},
            "s2": {"s2": s2},
            "texture": {"texture": c["texture"], "s2": s2},
            "ntl": {"ntl": c["ntl"], "grid": c["grid"]},
            "green": {"green": c["green"]},
        }[stage]
    )


def _check_inputs(cfg: FeatureBuildConfig, stage: str) -> list[Path]:
    inputs = stage_inputs(cfg, stage)
    missing = [p for p in inputs if not p.exists()]
    raw = {"s2": cfg.paths.s2_composite, "texture": cfg.paths.s2_composite, "ntl": cfg.paths.sdgsat}
    if stage in raw and not _globbed(raw[stage]):
        missing.append(Path(raw[stage]))
    if missing:
        raise MissingInputError(f"Stage '{stage}' is missing inputs: {[str(p) for p in missing]}")
    return inputs


def stage_output(cfg: FeatureBuildConfig, stage: str) -> Path:
    return cfg.interim / f"{stage}.parquet"


def run_stage(cfg: FeatureBuildConfig, stage: str, force: bool = False) -> Path:
    """Run one stage unless its inputs and parameters match the last run's manifest."""
    inputs = _check_inputs(cfg, stage)
    out, manifest_path = stage_output(cfg, stage), cfg.interim / f"{stage}.manifest.json"
    params = stage_params(cfg, stage)
    log.info("Stage %s: hashing %d input file(s)", stage, len(inputs))
    files = [{"path": str(p), "size": p.stat().st_size, "sha256": file_sha256(p)} for p in inputs]
    if not force and out.exists() and manifest_path.exists():
        last = json.loads(manifest_path.read_text())
        if last["inputs"] == files and last["params"] == json.loads(json.dumps(params)):
            log.info("Stage %s is up to date (use --force to rerun)", stage)
            return out

    started = time.time()
    module = importlib.import_module(f"wealth_index.geo.{stage}")
    if stage == "boundaries":
        df = module.run(cfg).drop(columns="geometry")  # boundaries.run writes its own GeoParquet
    else:
        df = module.run(cfg, read_villages(cfg))
        df.to_parquet(out)
    manifest = {
        "stage": stage,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - started, 1),
        "git_sha": _git_sha(),
        "inputs": files,
        "params": params,
        "rows": len(df),
        "columns": len(df.columns),
        "nan_share": {c: round(float(v), 4) for c, v in df.isna().mean().items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Stage %s done in %.0fs: %d rows x %d columns", stage, manifest["seconds"], *df.shape)
    return out


class FeatureValidationError(ValueError):
    """The joined feature table breaks the output contract (spec section 4.6)."""


def validate_features(
    df: pd.DataFrame, villages_index: pd.Index, expected: list[str], max_nan: float
) -> pd.Series:
    """Raise on contract violations; return the NaN share per column."""
    problems = []
    if not df.index.is_unique:
        problems.append("duplicate ids")
    if set(df.index) != set(villages_index):
        problems.append(
            f"{len(set(villages_index) - set(df.index))} villages missing, "
            f"{len(set(df.index) - set(villages_index))} unknown ids"
        )
    if list(df.columns) != expected:
        missing = [c for c in expected if c not in df]
        extra = [c for c in df if c not in expected]
        problems.append(f"columns differ from the catalogue: missing {missing}, unexpected {extra}")
    non_numeric = [c for c in df if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"non-numeric columns {non_numeric}")
    numeric = df.select_dtypes("number")
    infinite = numeric.columns[np.isinf(numeric.to_numpy()).any(axis=0)].tolist()
    if infinite:
        problems.append(f"infinite values in {infinite}")
    nan_share = df.isna().mean()
    too_sparse = nan_share[nan_share > max_nan]
    if len(too_sparse):
        problems.append(f"over {max_nan:.0%} NaN: {too_sparse.round(3).to_dict()}")
    if problems:
        raise FeatureValidationError("; ".join(problems))
    return nan_share


def build(cfg: FeatureBuildConfig, partial: bool = False) -> Path:
    """Join the stage outputs into processed/features.csv (id, lon, lat, features)."""
    villages = read_villages(cfg)
    groups = catalogue(cfg)
    missing = [s for s in groups if not stage_output(cfg, s).exists()]
    if missing and not partial:
        raise MissingInputError(f"Stages not run yet: {missing} (or pass --partial)")
    frames = [pd.read_parquet(stage_output(cfg, s)) for s in groups if s not in missing]
    expected = [c for s, cols in groups.items() if s not in missing for c in cols]
    features = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=villages.index)
    nan_share = validate_features(features, villages.index, expected, cfg.build.max_nan_share)

    out = villages[["lon", "lat"]].join(features)
    out.index.name = "id"
    path = cfg.processed / "features.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stages": [s for s in groups if s not in missing],
        "missing_stages": missing,
        "rows": len(out),
        "features": len(expected),
        "nan_share": {c: round(float(v), 4) for c, v in nan_share.items() if v > 0},
    }
    (cfg.processed / "features.manifest.json").write_text(json.dumps(report, indent=2))
    if missing:
        log.warning("Partial build without %s", missing)
    log.info("Wrote %s: %d villages x %d features", path, len(out), len(expected))
    return path


def run_all(cfg: FeatureBuildConfig, force: bool = False) -> Path:
    """Every stage whose inputs exist, then the build (partial if any stage was skipped)."""
    skipped = []
    for stage in STAGES:
        try:
            run_stage(cfg, stage, force=force)
        except MissingInputError as e:
            log.warning("Skipping: %s", e)
            skipped.append(stage)
    return build(cfg, partial=bool(skipped))
