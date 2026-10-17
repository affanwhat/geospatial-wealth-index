import numpy as np
import pandas as pd

from wealth_index.spatial import holdout_split, make_cv, spatial_blocks


def test_nearby_points_share_a_block_and_distant_points_do_not():
    coords = pd.DataFrame({"lon": [106.800, 106.801, 107.200], "lat": [-6.200, -6.201, -6.600]})
    groups = spatial_blocks(coords, block_size_km=5)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_spatial_holdout_has_no_block_overlap():
    rng = np.random.default_rng(0)
    coords = pd.DataFrame(
        {"lon": rng.uniform(106.4, 107.3, 500), "lat": rng.uniform(-6.8, -5.95, 500)}
    )
    groups = spatial_blocks(coords, block_size_km=5)
    tr, te = holdout_split(len(groups), "spatial", groups, test_size=0.3, seed=0)
    assert not set(groups[tr]) & set(groups[te])
    assert len(tr) + len(te) == 500


def test_spatial_cv_folds_have_no_block_overlap():
    rng = np.random.default_rng(1)
    coords = pd.DataFrame(
        {"lon": rng.uniform(106.4, 107.3, 300), "lat": rng.uniform(-6.8, -5.95, 300)}
    )
    groups = spatial_blocks(coords, block_size_km=10)
    for tr, va in make_cv("spatial", 3, seed=0).split(np.zeros(300), groups=groups):
        assert not set(groups[tr]) & set(groups[va])
