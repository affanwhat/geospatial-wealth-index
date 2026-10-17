"""Synthetic stand-in dataset with the same schema as the real one.

The real wealth index is derived from BPS Podes 2024 microdata, which cannot be
redistributed. This generator produces villages scattered over Jabodetabek with
feature names taken from the thesis feature set, so the pipeline, tests, and
README quickstart run without private data. Values are not meant to be realistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BBOX = {"lon": (106.40, 107.30), "lat": (-6.80, -5.95)}  # approx. Jabodetabek
JAKARTA_CENTER = (106.827, -6.175)  # Monas


def _dist_km(lon, lat, lon0, lat0):
    dx = (lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0))
    dy = (lat - lat0) * 110.574
    return np.hypot(dx, dy)


def generate_sample(n: int = 1200, seed: int = 42) -> pd.DataFrame:
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

    area_km2 = rng.lognormal(0, 0.8, n)
    ntl_mean = np.exp(0.8 * latent) + rng.gamma(1.0, 0.3, n)
    ndvi = np.clip(noisy(0.65 - 0.35 * urban, 0.06), -0.2, 1)

    df = pd.DataFrame(
        {
            "id": [f"SYN{i:05d}" for i in range(n)],
            "lon": lon,
            "lat": lat,
            # SDGSAT-1 night-time light statistics
            "NTL_MEAN": ntl_mean,
            "NTL_STD": noisy(0.5 * ntl_mean, 0.3),
            "NTL_SUM": ntl_mean * area_km2 * 1e4,
            "NTL_MEDIAN": noisy(0.9 * ntl_mean, 0.2),
            "NTLBin_RCount": np.clip(noisy(0.3 + 0.6 * urban, 0.1), 0, 1),
            # Sentinel-2 spectral indices
            "NDVI_MEAN1": ndvi,
            "EVI_MEAN1": noisy(1.1 * ndvi, 0.02),  # near-duplicate of NDVI on purpose
            "MSAVI_MEAN1": noisy(0.9 * ndvi, 0.03),
            "NDVI_DEVST1": np.abs(noisy(0.15 - 0.05 * urban, 0.03)),
            "NDBI_MEAN1": noisy(-0.1 + 0.3 * urban, 0.05),
            "Green200M_MEAN": noisy(0.4 - 0.2 * urban + 0.05 * field, 0.06),
            # OpenStreetMap accessibility
            "Road_Dens": noisy(2 + 10 * urban, 1.0),
            "Road_RLength": noisy(5 + 15 * urban, 2.0),
            "RoadPri_Prox": rng.exponential(200 + 3000 * (1 - urban)),
            "POIEdu_Dens": rng.poisson(np.exp(0.6 * latent)).astype(float),
            "POIHC_Dens": rng.poisson(np.exp(0.5 * latent)).astype(float),
            "POICmer_Dens": rng.poisson(np.exp(0.8 * latent)).astype(float),
            "POIFin_Dens": rng.poisson(np.exp(0.7 * latent)).astype(float),
            "POI_Count1KM": rng.poisson(5 * np.exp(0.5 * latent)).astype(float),
        }
    )
    # Building texture principal components: mostly noise, weak signal on PC1.
    for band in ["B2", "B3", "B4", "B8", "B11", "B12"]:
        df[f"{band}_PC1"] = noisy(0.8 * latent, 1.0)
        for k in (2, 3):
            df[f"{band}_PC{k}"] = rng.normal(0, 1, n)

    z = (latent - latent.mean()) / latent.std()
    df["wealth_index"] = np.clip(37.5 + 18 * z, 0, 100)
    return df
