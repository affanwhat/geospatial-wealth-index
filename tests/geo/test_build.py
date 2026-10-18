"""End-to-end run of the raster stages and the build on a two-village test area."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("exactextract")
pytest.importorskip("skimage")

import geopandas as gpd  # noqa: E402
from geo_helpers import UTM, write_raster  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.geometry import box  # noqa: E402

from wealth_index.geo.build import (  # noqa: E402
    FeatureValidationError,
    MissingInputError,
    build,
    run_stage,
    validate_features,
)
from wealth_index.geo.catalogue import catalogue  # noqa: E402
from wealth_index.geo.config import FeatureBuildConfig, PathsConfig, load_feature_config  # noqa: E402

X0, Y0 = 700_000, 9_320_000
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def area(tmp_path):
    """Two 1 km villages, a 10-band composite and one NTL scene (in lon/lat, no nodata value)."""
    gpd.GeoDataFrame(
        {"KDEPUM": ["31.71.01.1001", "31.71.01.1002"], "NAMOBJ": ["A", "B"]},
        geometry=[box(X0, Y0, X0 + 1000, Y0 + 1000), box(X0 + 1000, Y0, X0 + 2000, Y0 + 1000)],
        crs=UTM,
    ).to_crs("EPSG:4326").to_file(tmp_path / "villages.geojson")

    rng = np.random.default_rng(0)
    left, top, size = X0 - 3000, Y0 + 4000, 800  # 8 x 8 km at 10 m around the villages
    composite = rng.integers(300, 3000, size=(10, size, size)).astype("uint16")
    composite[6, :, :400] = rng.integers(
        1500, 4500, size=(size, 400)
    )  # greener west: higher NIR (B8)
    write_raster(tmp_path / "s2" / "s2_l2a_2024_median.tif", composite, left, top, 10, nodata=0)

    # Scene centred on the shared village edge, so its bright east half covers only village B.
    lon, lat = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform(X0 + 1000, Y0 + 500)
    ntl = rng.gamma(2.0, 3.0, size=(600, 600)).astype("float32")
    ntl[:, 300:] += 30
    ntl[:20], ntl[-20:], ntl[:, :20], ntl[:, -20:] = 0, 0, 0, 0  # fill border around the scene
    write_raster(
        tmp_path / "sdgsat" / "scene.tif",
        ntl,
        lon - 0.03,
        lat + 0.03,
        0.0001,
        crs="EPSG:4326",
        nodata=None,
    )

    cfg = FeatureBuildConfig(
        paths=PathsConfig(
            boundaries=str(tmp_path / "villages.geojson"),
            osm_pbf=str(tmp_path / "missing.osm.pbf"),
            s2_composite=str(tmp_path / "s2" / "*.tif"),
            sdgsat=str(tmp_path / "sdgsat" / "*.tif"),
            interim=str(tmp_path / "interim"),
            processed=str(tmp_path / "processed"),
        )
    )
    return cfg


def test_raster_stages_and_partial_build(area):
    cfg = area
    for stage in ["boundaries", "s2", "texture", "ntl", "green"]:
        run_stage(cfg, stage)
    with pytest.raises(MissingInputError, match="osm"):
        run_stage(cfg, "osm")
    with pytest.raises(MissingInputError, match="osm"):
        build(cfg)

    out = pd.read_csv(build(cfg, partial=True), index_col="id")
    groups = catalogue(cfg)
    assert list(out.columns) == ["lon", "lat"] + [
        c for s in ["s2", "green", "texture", "ntl"] for c in groups[s]
    ]
    assert list(out.index) == ["31.71.01.1001", "31.71.01.1002"]
    assert out.notna().all().all()
    west, east = out.iloc[0], out.iloc[1]
    assert west["NDVI_MEAN"] > east["NDVI_MEAN"]
    assert east["NTL_MEAN"] > west["NTL_MEAN"] + 20
    assert 0 <= west["NTLBin_Ratio"] <= 1 and east["NTLPos_MEAN"] >= east["NTL_MEAN"]
    manifest = json.loads((Path(cfg.paths.processed) / "features.manifest.json").read_text())
    assert manifest["missing_stages"] == ["osm"]


def test_unchanged_stage_is_skipped_and_force_reruns(area):
    cfg = area
    run_stage(cfg, "boundaries")
    manifest = Path(cfg.paths.interim) / "boundaries.manifest.json"
    first = manifest.read_text()
    run_stage(cfg, "boundaries")
    assert manifest.read_text() == first
    cfg.green.radius_m = 100  # other stages' parameters don't invalidate this one
    run_stage(cfg, "boundaries")
    assert manifest.read_text() == first
    run_stage(cfg, "boundaries", force=True)
    assert json.loads(manifest.read_text())["seconds"] >= 0


def test_validation_rejects_broken_tables():
    idx = pd.Index(["a", "b"])
    good = pd.DataFrame({"x": [1.0, 2.0], "y": [0.5, np.nan]}, index=idx)
    validate_features(good, idx, ["x", "y"], max_nan=0.5)
    with pytest.raises(FeatureValidationError, match="missing \\['z'\\]"):
        validate_features(good, idx, ["x", "y", "z"], max_nan=0.5)
    with pytest.raises(FeatureValidationError, match="NaN"):
        validate_features(good, idx, ["x", "y"], max_nan=0.2)
    with pytest.raises(FeatureValidationError, match="infinite"):
        validate_features(good.assign(x=[1.0, np.inf]), idx, ["x", "y"], max_nan=0.5)
    with pytest.raises(FeatureValidationError, match="villages missing"):
        validate_features(good.iloc[:1], idx, ["x", "y"], max_nan=0.5)


def test_features_yaml_matches_the_config_defaults():
    assert load_feature_config(ROOT / "configs" / "features.yaml") == FeatureBuildConfig()
