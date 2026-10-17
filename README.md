# Geospatial Wealth Index — Jakarta Metropolitan Area

Predicting village-level (kelurahan/desa) wealth across Jabodetabek from open geospatial data:
Sentinel-2 imagery, SDGSAT-1 night-time lights, and OpenStreetMap. The target is a DHS-style
wealth index built by PCA over BPS Potensi Desa (Podes) 2024 indicators.

Based on my undergraduate thesis (Universitas Indonesia, 2025), rebuilt as a reproducible pipeline.

> **Status:** Phase 1 (modeling pipeline) is in place and runs on a synthetic dataset.
> Feature extraction from raw imagery/OSM and the Podes wealth index are next.

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

## Quickstart

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"

wi make-sample                                   # synthetic data with the real schema
wi train --config configs/sample.yaml
wi train --config configs/sample-tpot.yaml       # optional, needs the tpot extra
wi predict --input data/sample/dataset.csv
wi export-runs                                   # all runs as CSV tables in artifacts/report/
pytest
```

## Layout

```
configs/            training configs (real data, synthetic sample)
src/wealth_index/   data validation, spatial blocking, selectors, training, prediction, CLI
tests/              unit + end-to-end tests on synthetic data
notebooks/          original thesis notebook (exploration record)
```

## Data

BPS Podes microdata is licensed for private use and is never committed. `data/` is git-ignored;
the synthetic generator (`wi make-sample`) mirrors the real schema so everything runs without it.

## Roadmap

1. ~~Modeling pipeline~~
2. Feature extraction: Sentinel-2 indices and texture, SDGSAT-1 NTL statistics, OSM POI/road features, zonal stats per village
3. Wealth index construction from Podes 2024 (PCA, DHS method)
4. Training on real data, prediction maps, model card
