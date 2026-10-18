import numpy as np
import pytest

pytest.importorskip("rasterio")

from geo_helpers import UTM  # noqa: E402, F401

from wealth_index.geo.config import StripeConfig  # noqa: E402
from wealth_index.geo.ntl import combine, destripe, stripe_direction  # noqa: E402


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    clean = (5 + rng.normal(0, 0.5, size=(120, 120))).astype("float32")
    clean[60:63, 40:43] = 200  # a bright 3x3 light that must survive
    striped = clean.copy()
    striped[:, 80] += 40  # a full-height stripe
    return clean, striped


def test_stripe_direction_is_detected(scene):
    _, striped = scene
    assert stripe_direction(striped) == "vertical"
    assert stripe_direction(striped.T) == "horizontal"


def test_destripe_removes_stripe_and_keeps_lights(scene):
    clean, striped = scene
    fixed = destripe(striped, StripeConfig(direction="auto", k=3, min_run=10, filter_width=5))
    assert np.abs(fixed[:, 80] - clean[:, 80]).mean() < 1.0  # stripe back to background
    assert np.allclose(fixed[60:63, 40:43], 200)  # light untouched
    assert np.allclose(fixed[:, :78], clean[:, :78])  # nothing else changed


def test_destripe_leaves_nan_alone():
    img = np.full((30, 30), 5.0, dtype="float32")
    img[:, :5] = np.nan
    fixed = destripe(img, StripeConfig(direction="vertical"))
    assert np.isnan(fixed[:, :5]).all() and np.allclose(fixed[:, 5:], 5)


def test_combine_overlapping_scenes():
    stack = np.array([[[1.0, np.nan, np.nan]], [[3.0, 4.0, np.nan]], [[5.0, np.nan, np.nan]]])
    assert np.allclose(combine(stack, "median")[0, :2], [3, 4]) and np.isnan(
        combine(stack, "median")[0, 2]
    )
    assert np.allclose(combine(stack, "first")[0, :2], [1, 4])
    assert np.allclose(combine(stack, "latest")[0, :2], [5, 4])
