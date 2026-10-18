import numpy as np
import pytest

pytest.importorskip("rasterio")

from geo_helpers import UTM  # noqa: E402, F401

from wealth_index.geo.s2 import compute_indices  # noqa: E402


def bands(b2, b4, b8, b11):
    return {k: np.array([v]) for k, v in {"B2": b2, "B4": b4, "B8": b8, "B11": b11}.items()}


def test_indices_match_hand_values():
    out = compute_indices(bands(b2=0.05, b4=0.1, b8=0.3, b11=0.2))
    assert out["NDVI"][0] == pytest.approx(0.5)
    assert out["EVI"][0] == pytest.approx(2.5 * 0.2 / (0.3 + 0.6 - 0.375 + 1))
    assert out["SAVI"][0] == pytest.approx(1.5 * 0.2 / 0.9)
    assert out["MSAVI"][0] == pytest.approx((1.6 - np.sqrt(1.6**2 - 1.6)) / 2)
    assert out["TVI"][0] == pytest.approx(1.0)
    assert out["GR"][0] == pytest.approx(3.0)
    assert out["NDBI"][0] == pytest.approx(-0.2)
    assert np.isnan(out["NDBI_POS"][0]) and out["NDBI_BIN"][0] == 0


def test_built_up_pixel_counts_as_positive():
    out = compute_indices(bands(b2=0.1, b4=0.15, b8=0.2, b11=0.3))
    assert out["NDBI_POS"][0] == pytest.approx(0.2) and out["NDBI_BIN"][0] == 1


def test_division_by_zero_gives_nan():
    out = compute_indices(bands(b2=0.0, b4=0.0, b8=0.0, b11=0.0))
    assert np.isnan(out["NDVI"][0]) and np.isnan(out["GR"][0]) and np.isnan(out["NDBI"][0])
    assert np.isnan(out["NDBI_BIN"][0])  # no data stays no data, not "not built-up"


def test_evi_blow_up_on_blue_bright_pixel_is_nan():
    # Denominator 0.06 + 6*0.05 - 7.5*0.18 + 1 = 0.01, so raw EVI would be 2.5.
    out = compute_indices(bands(b2=0.18, b4=0.05, b8=0.06, b11=0.1))
    assert np.isnan(out["EVI"][0]) and out["NDVI"][0] == pytest.approx(0.01 / 0.11)
