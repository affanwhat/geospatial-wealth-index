# Geospatial Wealth Index — Jakarta Metropolitan Area

Predicting village-level (kelurahan/desa) wealth across Jabodetabek from open geospatial data:
Sentinel-2 imagery, SDGSAT-1 night-time lights, and OpenStreetMap. The target is a DHS-style
wealth index built by PCA over BPS Potensi Desa (Podes) 2024 indicators.

Based on my undergraduate thesis (Universitas Indonesia, 2025), rebuilt as a reproducible pipeline.

> **Status:** The pipeline has run end to end on real data: features extracted, model trained,
> all 1,502 villages scored and mapped. Labels are the wealth index from the thesis, so
> rebuilding it from Podes was skipped. See [MODEL_CARD.md](MODEL_CARD.md) for results.

## What the pipeline does

- **Leak-free feature selection.** Correlation filtering and permutation-importance selection are
  scikit-learn transformers inside the `Pipeline`, so they are refit on every training fold.
- **Spatial validation.** Villages are grouped into square blocks; the test set and CV folds are
  split by block. Each run also reports random vs. spatial CV error to show how much a naive
  split overstates accuracy.
- **Reproducible runs.** Every run saves the fitted pipeline, metrics, test predictions, plots,
  config, data hash, git SHA, and library versions under `artifacts/<timestamp>/`.
- **Two tuners.** `search.method: random` (scikit-learn `RandomizedSearchCV`) or `tpot`
  (genetic programming, as used in the thesis; `pip install -e ".[tpot]"`). TPOT can search
  RandomForest only over the thesis grid, or any TPOT preset (`linear-light`, `graph`, ...).
- **Batch scoring.** `wi predict` validates the input schema against the trained model and writes
  one prediction per village.
- **Prediction maps.** `wi map` joins the predictions to the village polygons and draws a
  choropleth (PNG), optionally writing the joined polygons as a GeoPackage for QGIS.
- **Model card.** `wi model-card` writes `MODEL_CARD.md` from a run's saved metrics, metadata and
  feature importances, plus the predictions and map when they exist.
- **Feature extraction (`wi features`).** Village polygons, OSM, Sentinel-2 and SDGSAT-1 become
  89 features per village (`configs/features.yaml`): POI kernel density, proximity and counts;
  road length, density and distance; spectral index statistics; GLCM texture; green exposure
  around settlements; destriped night-time light statistics. Each stage caches its output with
  a manifest (input hashes, parameters, NaN shares) and reruns only when those change.

## Quickstart

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"

wi make-sample                                   # synthetic data with the real schema
wi train --config configs/sample.yaml
wi train --config configs/sample-tpot.yaml       # optional, needs the tpot extra
wi predict --input data/sample/dataset.csv
wi report                                        # dashboard of all runs: artifacts/report/index.html
wi export-runs                                   # same run tables as CSV files
pytest
```

Feature extraction (`pip install -e ".[geo]"`):

```bash
wi features boundaries     # data/spatial_data/subdistricts.json -> data/processed/villages.csv
wi features osm            # needs data/raw/osm/java-250101.osm.pbf (Geofabrik)
wi features s2-export      # submits the cloud-masked 2024 composite on Google Earth Engine
wi features s2             # after downloading the export into data/raw/s2/
wi features texture
wi features ntl            # SDGSAT-1 GIU scenes in data/raw/sdgsat/; without them it offers
                           # VIIRS (~460 m) from GEE instead (or set ntl.source: viirs)
wi features green
wi features build          # join + validate -> data/processed/features.csv
wi features all            # every stage whose inputs exist, then build
```

Real data, from features to map (also needs the `geo` extra):

```bash
wi labels join --labels data_inside/labels_podes.csv   # features + labels -> data/processed/dataset.csv
wi train --config configs/train.yaml
wi predict --input data/processed/features.csv         # -> data/predictions/wealth_index.csv
wi map --vector data/predictions/wealth_index_map.gpkg # -> data/predictions/wealth_index_map.png
wi model-card                                          # latest run -> MODEL_CARD.md
```

## Layout

```
configs/            training configs (real data, synthetic sample)
src/wealth_index/   data validation, spatial blocking, selectors, training, prediction, CLI
src/wealth_index/geo/  feature extraction (boundaries, OSM, Sentinel-2, texture, NTL, build), prediction maps
tests/              unit + end-to-end tests on synthetic data (tests/geo: small known-answer fixtures)
notebooks/          original thesis notebook (exploration record)
```

## Data

BPS Podes microdata is licensed for private use and is never committed. `data/` is git-ignored;
the synthetic generator (`wi make-sample`) mirrors the real schema so everything runs without it.

OpenStreetMap data © OpenStreetMap contributors, ODbL. Sentinel-2 data: Copernicus (ESA).

## Roadmap

1. ~~Modeling pipeline~~
2. ~~Feature extraction~~ (Sentinel-2 indices and texture, SDGSAT-1 NTL statistics, OSM POI/road features, zonal stats per village)
3. ~~Wealth index construction from Podes 2024~~ (skipped: labels are the thesis's wealth index)
4. ~~Training on real data, predictions, prediction maps, model card~~
