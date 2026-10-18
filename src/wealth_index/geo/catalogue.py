"""Feature names per group (spec section 5). Pure Python, so the synthetic generator can use it."""

from __future__ import annotations

from wealth_index.geo.config import FeatureBuildConfig

INDICES = ["NDVI", "EVI", "SAVI", "MSAVI", "TVI", "GR", "NDBI"]
TEXTURE_PROPS = ["CONTRAST", "DISSIMILARITY", "HOMOGENEITY", "ENERGY", "CORRELATION", "ASM"]
TEXTURE_MOMENTS = ["SKEW", "KURT"]
ROAD_CLASSES = ["Pri", "Sec", "Ter"]


def s2_columns(cfg: FeatureBuildConfig) -> list[str]:
    if cfg.feature_set == "superset":
        cols = [f"{i}_{s}" for i in INDICES for s in ("MEAN", "STD", "MIN", "MAX", "RANGE", "SUM")]
    else:
        cols = [f"{i}_{s}" for i in INDICES[:-1] for s in ("MEAN", "STD")]
        cols += ["NDVI_MIN", "NDVI_MAX", "NDBI_MEAN", "NDBI_STD"]
    return cols + ["NDBIPos_MEAN", "NDBIBin_Ratio"]


def texture_bands(cfg: FeatureBuildConfig) -> list[str]:
    return cfg.s2.bands if cfg.feature_set == "superset" else cfg.texture.bands


def texture_columns(cfg: FeatureBuildConfig) -> list[str]:
    props = TEXTURE_PROPS + TEXTURE_MOMENTS
    return [f"TEX_{b}_{p}" for b in texture_bands(cfg) for p in props]


def green_columns(cfg: FeatureBuildConfig) -> list[str]:
    return ["Green_MEAN", "Green_STD", "Green200M_MEAN", "Green200M_STD", "Green200M_PCT90"]


def ntl_columns(cfg: FeatureBuildConfig) -> list[str]:
    stats = ["MEAN", "STD", "MEDIAN", "PCT90", "MAX", "SUM"]
    return [f"NTL_{s}" for s in stats] + ["NTLPos_MEAN", "NTLBin_Ratio"]


def osm_columns(cfg: FeatureBuildConfig) -> list[str]:
    cats = list(cfg.osm.poi_categories)
    cols = [f"POI{c}_Dens" for c in cats] + [f"POI{c}_Prox" for c in cats]
    cols += [f"POI_Count{r:g}KM" for r in cfg.osm.count_radii_km] + ["POI_Mean"]
    cols += ["Road_Length", "Road_RLength", "Road_Dens"]
    cols += [f"Road{c}_{s}" for s in ("Dens", "Prox", "MeanDist") for c in ROAD_CLASSES]
    if cfg.osm.buildings:
        cols += ["Bldg_Count", "Bldg_Dens", "Bldg_AreaShare", "Bldg_MeanArea"]
    return cols


# Stage name -> its columns. Order is the column order of features.csv.
GROUPS = {
    "s2": s2_columns,
    "green": green_columns,
    "texture": texture_columns,
    "ntl": ntl_columns,
    "osm": osm_columns,
}


def catalogue(cfg: FeatureBuildConfig) -> dict[str, list[str]]:
    return {stage: fn(cfg) for stage, fn in GROUPS.items()}
