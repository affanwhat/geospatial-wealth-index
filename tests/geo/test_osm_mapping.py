import numpy as np
import pytest

pytest.importorskip("rasterio")

import geopandas as gpd  # noqa: E402
from geo_helpers import UTM  # noqa: E402
from shapely.geometry import LineString, Point, Polygon, box  # noqa: E402

from wealth_index.geo.boundaries import clean_villages  # noqa: E402
from wealth_index.geo.catalogue import osm_columns  # noqa: E402
from wealth_index.geo.config import FeatureBuildConfig  # noqa: E402
from wealth_index.geo.osm import clipped_length, osm_features, poi_points, road_lines  # noqa: E402

X0, Y0 = 700_000, 9_320_000  # central Jakarta in UTM 48S


def quackosm_frame(rows):
    """Rows of (feature_id, geometry, {group: 'key=value'}) in quackosm's grouped output layout."""
    groups = sorted({g for _, _, tags in rows for g in tags})
    data = {"feature_id": [r[0] for r in rows], "geometry": [r[1] for r in rows]}
    data |= {g: [r[2].get(g) for r in rows] for g in groups}
    return gpd.GeoDataFrame(data, crs=UTM)


def test_node_inside_area_of_same_category_counts_once():
    bank_building = box(X0, Y0, X0 + 40, Y0 + 40)
    osm = quackosm_frame(
        [
            ("way/1", bank_building, {"Fin": "building=retail", "Cmer": "building=retail"}),
            ("node/2", Point(X0 + 20, Y0 + 20), {"Fin": "amenity=bank"}),
            ("way/3", box(X0 + 500, Y0, X0 + 600, Y0 + 100), {"Edu": "amenity=school"}),
            ("node/4", Point(X0 + 900, Y0), {"Fin": "amenity=atm", "Cmer": "shop=convenience"}),
        ]
    )
    pois = poi_points(osm, ["Fin", "Cmer", "Edu"])
    counts = pois.groupby("category").size().to_dict()
    assert counts == {"Fin": 2, "Cmer": 2, "Edu": 1}  # way/1 dropped from Fin, kept for Cmer
    assert set(pois.geom_type) == {"Point"}
    assert (
        pois.loc[pois.category == "Edu", "geometry"]
        .iloc[0]
        .within(box(X0 + 500, Y0, X0 + 600, Y0 + 100))
    )


def test_road_classes_and_lines_only():
    osm = quackosm_frame(
        [
            ("way/1", LineString([(0, 0), (10, 0)]), {"Road": "highway=primary_link"}),
            ("way/2", LineString([(0, 0), (0, 10)]), {"Road": "highway=residential"}),
            (
                "way/3",
                box(0, 0, 5, 5),
                {"Road": "highway=service"},
            ),  # a pedestrian area, not a line
        ]
    )
    roads = road_lines(osm, FeatureBuildConfig().osm.road_classes)
    assert list(roads.feature_id) == ["way/1", "way/2"]
    assert roads.cls.iloc[0] == "Pri" and roads.cls.isna().iloc[1]


def test_road_length_is_split_between_villages():
    villages = gpd.GeoDataFrame(
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], index=["a", "b"], crs=UTM
    )
    roads = gpd.GeoSeries(
        [LineString([(50, 50), (150, 50)]), LineString([(10, 10), (20, 10)])], crs=UTM
    )
    assert clipped_length(villages, roads).to_dict() == pytest.approx({"a": 60, "b": 50})


def test_osm_features_produce_the_catalogue(tmp_path):
    cfg = FeatureBuildConfig()
    cfg.paths.interim = str(tmp_path)
    raw = gpd.GeoDataFrame(
        {"KDEPUM": ["v1", "v2"]},
        geometry=[
            box(X0, Y0, X0 + 1000, Y0 + 1000),
            Polygon(
                [(X0 + 1000, Y0), (X0 + 3000, Y0), (X0 + 3000, Y0 + 1000), (X0 + 1000, Y0 + 1000)]
            ),
        ],
        crs=UTM,
    ).to_crs("EPSG:4326")
    villages = clean_villages(raw, cfg)
    osm = quackosm_frame(
        [
            ("node/1", Point(X0 + 500, Y0 + 500), {"Edu": "amenity=school"}),
            ("node/2", Point(X0 + 2500, Y0 + 500), {"HC": "amenity=clinic"}),
            (
                "way/3",
                LineString([(X0, Y0 + 500), (X0 + 3000, Y0 + 500)]),
                {"Road": "highway=primary"},
            ),
            (
                "way/4",
                LineString([(X0 + 500, Y0), (X0 + 500, Y0 + 1000)]),
                {"Road": "highway=residential"},
            ),
        ]
    ).to_crs("EPSG:4326")
    feats = osm_features(cfg, villages, osm)

    assert list(feats.columns) == osm_columns(cfg)
    assert list(feats.index) == ["v1", "v2"]
    assert feats.loc["v1", "POIEdu_Prox"] == 0 and feats.loc["v2", "POIEdu_Prox"] == pytest.approx(
        500, abs=1
    )
    assert feats.loc["v1", "Road_Length"] == pytest.approx(
        2.0, rel=0.01
    )  # 1 km primary + 1 km residential
    assert feats.loc["v2", "Road_RLength"] == pytest.approx(1.0, rel=0.01)  # 2 km in 2 km2
    assert feats.loc["v1", "RoadPri_Prox"] == 0 and feats.loc["v1", "RoadPri_MeanDist"] < 300
    assert feats["POIEdu_Dens"].loc["v1"] > feats["POIEdu_Dens"].loc["v2"]
    assert feats.loc["v1", "POI_Count1KM"] == 1 and feats.loc["v1", "POI_Count5KM"] == 2
    assert np.isnan(feats["POIFin_Prox"]).all()  # no banks mapped at all


def test_clip_pbf_keeps_features_touching_the_box(tmp_path):
    osmium = pytest.importorskip("osmium")
    from osmium.osm.mutable import Node, Relation, Way

    from wealth_index.geo.osm import clip_pbf

    src = tmp_path / "region.osm.pbf"
    with osmium.SimpleWriter(str(src)) as w:
        for i, lon in enumerate([0.5, 1.5, 5.0, 6.0], start=1):  # nodes 1, 2 inside the box
            w.add_node(Node(id=i, location=(lon, 0.5)))
        w.add_way(Way(id=10, nodes=[2, 3], tags={"highway": "primary"}))  # crosses the edge
        w.add_way(Way(id=11, nodes=[3, 4], tags={"highway": "primary"}))  # wholly outside
        w.add_relation(Relation(id=20, members=[("w", 10, "outer")], tags={"type": "route"}))

    out = clip_pbf(src, (0.0, 0.0, 2.0, 1.0), tmp_path / "clip.osm.pbf")
    kept = {(o.type_str(), o.id) for o in osmium.FileProcessor(str(out))}
    # Node 3 lies outside but is kept so way 10 stays complete.
    assert kept == {("n", 1), ("n", 2), ("n", 3), ("w", 10), ("r", 20)}
