"""Pruebas de segmentación clásica (F2)."""

import numpy as np
import pytest

from muni.core.volume import Volume3D
from muni.segment.classical import ClassicalSegmenter
from muni.segment.connected import connected_components, largest_component
from muni.segment.morphology import binary_close, binary_dilate, binary_erode, binary_open
from muni.segment.threshold import adaptive_mean_threshold, otsu_threshold


def test_otsu_separates_bimodal():
    rng = np.random.default_rng(1)
    dark = rng.normal(40, 6, 4000)
    light = rng.normal(200, 6, 4000)
    img = np.concatenate([dark, light])
    t = otsu_threshold(img)
    assert 40 < t < 200


def test_morphology_2d():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    d = binary_dilate(mask, structure="cross")
    assert d.sum() == 5  # centro + 4 vecinos
    assert binary_erode(mask, structure="cross").sum() == 0  # píxel aislado se elimina

    blob = np.zeros((7, 7), dtype=bool)
    blob[3, 3] = True
    closed = binary_close(binary_open(blob, iterations=1), iterations=1)
    assert closed.shape == blob.shape


def test_connected_components():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True  # componente de 4
    mask[6:9, 6:9] = True  # componente de 9
    labels, n = connected_components(mask)
    assert n == 2
    big = largest_component(mask)
    assert big.sum() == 9


def test_adaptive_mean_threshold_dark_foreground():
    # Fondo claro con una franja oscura central.
    img = np.full((20, 20), 200.0)
    img[:, 8:12] = 20.0
    m = adaptive_mean_threshold(img, radius=3, C=10.0)
    assert m[:, 9].all()
    assert not m[:, 0].any()


def test_classical_segmenter_end_to_end():
    # Volumen sintético: cilindro oscuro vertical sobre fondo claro.
    z, y, x = 12, 48, 48
    yy, xx = np.mgrid[0:y, 0:x]
    disk = (yy - 24) ** 2 + (xx - 24) ** 2 < 10 ** 2
    bg = np.full((z, y, x), 220.0, dtype=np.float32)
    bg[:, disk] = 30.0
    vol = Volume3D(bg)

    seg = ClassicalSegmenter(denoise=False, keep_largest=True)
    result = seg.segment(vol)
    assert result.probability.shape == vol.shape
    assert result.probability.min() >= 0 and result.probability.max() <= 1

    # El cilindro debe capturarse con un IoU alto.
    inter = (result.mask & (bg < 100)).sum()
    union = (result.mask | (bg < 100)).sum()
    assert inter / union > 0.85
