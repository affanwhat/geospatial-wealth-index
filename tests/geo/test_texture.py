import numpy as np
import pytest

pytest.importorskip("skimage")

from geo_helpers import UTM  # noqa: E402, F401

from wealth_index.geo.texture import masked_glcm, quantize  # noqa: E402


def test_quantize_uses_levels_one_to_n():
    q = quantize(np.array([-5.0, 0.0, 0.5, 1.0, 9.0, np.nan]), 0.0, 1.0, levels=4)
    assert q.min() == 1 and q.max() == 4


def test_pixels_outside_the_village_do_not_change_texture():
    rng = np.random.default_rng(0)
    inside = np.zeros((40, 40), dtype=bool)
    inside[10:30, 5:25] = True
    q = rng.integers(1, 33, size=(40, 40)).astype(np.uint8)
    noisier = q.copy()
    noisier[~inside] = rng.integers(1, 33, size=(~inside).sum())  # different noise outside

    a = masked_glcm(q, inside, levels=32, distances=[1, 2])
    b = masked_glcm(noisier, inside, levels=32, distances=[1, 2])
    assert a == pytest.approx(b)
    assert all(np.isfinite(v) for v in a.values())


def test_village_without_pixel_pairs_gives_nan():
    inside = np.zeros((5, 5), dtype=bool)
    inside[2, 2] = True
    props = masked_glcm(np.full((5, 5), 3, np.uint8), inside, levels=32, distances=[1])
    assert all(np.isnan(v) for v in props.values())
