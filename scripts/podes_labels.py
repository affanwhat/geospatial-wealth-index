"""Turn the thesis' WI.csv (keyed by the Jabodetabekpunjur shapefile's OBJECTID) into a labels
file keyed by KDEPUM, ready for `wi labels join`.

Codes are looked up through OBJECTID. A village whose code differs from our boundaries (e.g. a
desa that became a kelurahan, 2xxx -> 1xxx) is matched by location instead: the boundary
village containing its representative point.

    python scripts/podes_labels.py
    wi labels join --labels data_inside/labels_podes.csv

Inputs and output live in the git-ignored data_inside/ (the thesis files are not in the repo).
The output is Podes-derived: personal use only, never commit or publish.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "data_inside"
WI = HERE / "WI.csv"
SHP = HERE / "Jabodetabekpunjur" / "Jabodetabekpunjur.shp"
BOUNDARIES = ROOT / "data" / "spatial_data" / "subdistricts.json"
THESIS_MAP = HERE / "labels_from_thesis_map.csv"
OUT = HERE / "labels_podes.csv"


def main() -> None:
    wi = pd.read_csv(WI, sep=";")
    shp = gpd.read_file(SHP)[["OBJECTID", "KDEPUM", "NAMOBJ", "geometry"]]
    shp["OBJECTID"] = shp["OBJECTID"].astype(int)
    lab = wi.merge(shp, left_on="id", right_on="OBJECTID", how="left", validate="1:1")
    if lab["KDEPUM"].isna().any():
        raise SystemExit(f"{int(lab['KDEPUM'].isna().sum())} WI ids are not in {SHP.name}")
    lab = gpd.GeoDataFrame(lab, geometry="geometry", crs=shp.crs)

    ours = gpd.read_file(BOUNDARIES)[["KDEPUM", "geometry"]].to_crs(shp.crs)
    lab["match"] = np.where(lab["KDEPUM"].isin(ours["KDEPUM"]), "code", "")
    todo = lab["match"] == ""
    if todo.any():
        pts = lab.loc[todo, ["geometry"]].copy()
        pts["geometry"] = pts.representative_point()
        hit = gpd.sjoin(pts, ours, predicate="within", how="left")
        hit = hit[~hit.index.duplicated()]
        found = hit["KDEPUM"].notna()
        for i, row in hit[found].iterrows():
            print(
                f"  {lab.at[i, 'NAMOBJ']}: {lab.at[i, 'KDEPUM']} -> {row['KDEPUM']} (by location)"
            )
        lab.loc[hit.index[found], "KDEPUM"] = hit.loc[found, "KDEPUM"]
        lab.loc[hit.index[found], "match"] = "location"
        for i in hit.index[~found]:
            print(
                f"  {lab.at[i, 'NAMOBJ']} ({lab.at[i, 'KDEPUM']}): no boundary village, kept as is"
            )
    if lab["KDEPUM"].duplicated().any():
        raise SystemExit(
            "duplicate KDEPUM after matching: "
            f"{lab.loc[lab.KDEPUM.duplicated(), 'KDEPUM'].tolist()}"
        )

    out = lab.rename(
        columns={"id": "OBJECTID_wi", "wealth_index": "wi", "wealth_quintile": "quintile"}
    )
    out = out[["KDEPUM", "NAMOBJ", "wi", "quintile", "OBJECTID_wi", "match"]]
    out.to_csv(OUT, index=False)
    print(
        f"Wrote {OUT}: {len(out)} villages ({(out.match == 'location').sum()} matched by location)"
    )
    print(out["wi"].describe().round(2).to_string())

    if THESIS_MAP.exists():
        t = pd.read_csv(THESIS_MAP, dtype={"KDEPUM": str})[["KDEPUM", "wi", "quintile_map"]]
        c = out.merge(t, on="KDEPUM", suffixes=("", "_map"))
        d = c["wi_map"] - c["wi"]
        print(
            f"vs thesis-map labels (n={len(c)}): r={np.corrcoef(c.wi, c.wi_map)[0, 1]:.4f} "
            f"MAE={d.abs().mean():.2f} RMSE={np.sqrt((d**2).mean()):.2f} "
            f"quintiles agree {(c.quintile == c.quintile_map).mean():.1%}"
        )


if __name__ == "__main__":
    main()
