"""Synthetic stand-in dataset with the same schema as the real one.

The real wealth index is derived from BPS Podes 2024 microdata, which cannot be
redistributed. This generator produces villages scattered over Jabodetabek with the
feature names of the Phase 2 catalogue (`wealth_index.geo.catalogue`), so the pipeline,
tests, and README quickstart run without private data. Values are not meant to be realistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wealth_index.geo.catalogue import catalogue
from wealth_index.geo.config import FeatureBuildConfig

BBOX = {"lon": (106.40, 107.30), "lat": (-6.80, -5.95)}  # approx. Jabodetabek
JAKARTA_CENTER = (106.827, -6.175)  # Monas


def _dist_km(lon, lat, lon0, lat0):
    dx = (lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0))
    dy = (lat - lat0) * 110.574
    return np.hypot(dx, dy)


def generate_sample(n: int = 1200, seed: int = 42, feature_set: str = "curated") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(*BBOX["lon"], n)
    lat = rng.uniform(*BBOX["lat"], n)

    # Latent wealth: urban-core gradient plus a smooth spatial field (autocorrelation).
    urban = np.exp(-_dist_km(lon, lat, *JAKARTA_CENTER) / 20)
    bump_lon = rng.uniform(*BBOX["lon"], 30)
    bump_lat = rng.uniform(*BBOX["lat"], 30)
    bump_amp = rng.normal(0, 1, 30)
    field = sum(
        a * np.exp(-(_dist_km(lon, lat, bx, by) ** 2) / (2 * 8**2))
        for a, bx, by in zip(bump_amp, bump_lon, bump_lat, strict=True)
    )
    latent = 2.0 * urban + 0.6 * field + rng.normal(0, 0.35, n)

    def noisy(x, scale):
        return x + rng.normal(0, scale, n)

    area_km2 = rng.lognormal(1.2, 0.8, n)
    ndvi = np.clip(noisy(0.65 - 0.35 * urban, 0.06), -0.2, 1)
    ndbi = noisy(-0.1 + 0.3 * urban, 0.05)
    ntl = np.exp(0.8 * latent) + rng.gamma(1.0, 0.3, n)
    # Per-index village means; EVI and SAVI are near-duplicates of NDVI on purpose.
    index_mean = {
        "NDVI": ndvi,
        "EVI": noisy(1.1 * ndvi, 0.02),
        "SAVI": noisy(0.75 * ndvi, 0.02),
        "MSAVI": noisy(0.9 * ndvi, 0.03),
        "TVI": np.sqrt(ndvi + 0.5),
        "GR": (1 + ndvi) / (1 - ndvi + 1e-3),
        "NDBI": ndbi,
    }

    def s2(name):
        index, stat = name.split("_")
        if index == "NDBIPos":
            return np.abs(noisy(0.05 + 0.15 * urban, 0.03))
        if index == "NDBIBin":
            return np.clip(noisy(0.2 + 0.6 * urban, 0.08), 0, 1)
        m = index_mean[index]
        spread = np.abs(noisy(0.15 - 0.05 * urban, 0.03))
        return {
            "MEAN": m,
            "STD": spread,
            "MIN": m - 2 * spread,
            "MAX": m + 2 * spread,
            "RANGE": 4 * spread,
            "SUM": m * area_km2 * 1e4,
        }[stat]

    def green(name):
        base = ndvi - 0.1 if name.startswith("Green_") else 0.4 - 0.2 * urban + 0.05 * field
        return {
            "MEAN": noisy(base, 0.05),
            "STD": np.abs(noisy(0.1, 0.03)),
            "PCT90": noisy(base + 0.15, 0.05),
        }[name.split("_")[1]]

    def texture(name):
        # Mostly noise; contrast and homogeneity carry a weak built-up signal.
        signal = {"CONTRAST": 0.8, "HOMOGENEITY": -0.6}.get(name.split("_")[-1], 0.0)
        return noisy(signal * latent, 1.0)

    def ntl_col(name):
        return {
            "NTL_MEAN": ntl,
            "NTL_STD": noisy(0.5 * ntl, 0.3),
            "NTL_MEDIAN": noisy(0.9 * ntl, 0.2),
            "NTL_PCT90": noisy(1.6 * ntl, 0.4),
            "NTL_MAX": noisy(3 * ntl, 1.0),
            "NTL_SUM": ntl * area_km2 * 1e4,
            "NTLPos_MEAN": noisy(1.2 * ntl, 0.3),
            "NTLBin_Ratio": np.clip(noisy(0.3 + 0.6 * urban, 0.1), 0, 1),
        }[name]

    def osm(name):
        if name.endswith("_Dens") and name.startswith("POI"):
            return rng.gamma(2.0, np.exp(0.6 * latent))
        if name == "POI_Mean":
            return rng.gamma(4.0, 2 * np.exp(0.6 * latent))
        if name.startswith("POI_Count"):
            km = float(name.removeprefix("POI_Count").removesuffix("KM"))
            return rng.poisson(5 * km**2 * np.exp(0.5 * latent)).astype(float)
        if name.endswith(("_Prox", "_MeanDist")):
            d = rng.exponential(200 + 3000 * (1 - urban))
            return d if name.endswith("_Prox") else d + rng.exponential(300 + 1500 * (1 - urban))
        if name in ("Road_Dens", "RoadPri_Dens", "RoadSec_Dens", "RoadTer_Dens"):
            return np.abs(noisy(2 + 10 * urban, 1.0)) * (1 if name == "Road_Dens" else 0.2)
        if name == "Road_Length":
            return np.abs(noisy(2 + 10 * urban, 1.0)) * area_km2
        if name == "Road_RLength":
            return np.abs(noisy(5 + 15 * urban, 2.0))
        raise KeyError(name)

    makers = {"s2": s2, "green": green, "texture": texture, "ntl": ntl_col, "osm": osm}
    columns = {"id": [f"SYN{i:05d}" for i in range(n)], "lon": lon, "lat": lat}
    for stage, names in catalogue(FeatureBuildConfig(feature_set=feature_set)).items():
        columns |= {name: makers[stage](name) for name in names}
    df = pd.DataFrame(columns)

    z = (latent - latent.mean()) / latent.std()
    df["wealth_index"] = np.clip(37.5 + 18 * z, 0, 100)
    return df
