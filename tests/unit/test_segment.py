"""Pruebas de segmentación (clásica y enfoque+Frangi)."""

import numpy as np

from muni.core.volume import Volume3D
from muni.segment.classical import ClassicalSegmenter
from muni.segment.connected import connected_components, largest_component
from muni.segment.focused import FocusedSegmenter
from muni.segment.meijering import MeijeringSegmenter
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
    _, n = connected_components(mask)
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


def test_focused_segmenter_end_to_end():
    rng = np.random.default_rng(42)
    z, y, x = 12, 64, 64
    yy, xx = np.mgrid[0:y, 0:x]
    bg = np.full((z, y, x), 200.0, dtype=np.float32)
    for zi in range(z):
        r = 10 + zi
        disk = (yy - 32) ** 2 + (xx - 32) ** 2 < r**2
        bg[zi, disk] = 50.0
    bg += rng.normal(0, 8, bg.shape).astype(np.float32)
    vol = Volume3D(bg)

    seg = FocusedSegmenter(denoise=False, keep_largest=True)
    result = seg.segment(vol)
    assert result.method == "focused"
    assert result.probability.shape == vol.shape
    assert result.probability.min() >= 0 and result.probability.max() <= 1
    assert result.mask.shape == vol.shape
    assert result.mask.any()


def test_focused_segmenter_with_denoise():
    rng = np.random.default_rng(99)
    z, y, x = 8, 48, 48
    yy, xx = np.mgrid[0:y, 0:x]
    bg = np.full((z, y, x), 200.0, dtype=np.float32)
    for zi in range(z):
        r = 6 + zi
        disk = (yy - 24) ** 2 + (xx - 24) ** 2 < r**2
        bg[zi, disk] = 50.0
    bg += rng.normal(0, 8, bg.shape).astype(np.float32)
    vol = Volume3D(bg)

    seg = FocusedSegmenter(denoise=True, sharpness_radius=3, keep_largest=False)
    result = seg.segment(vol)
    assert result.method == "focused"
    assert result.mask.shape == vol.shape
    assert result.mask.any()


def test_meijering_segmenter_end_to_end():
    rng = np.random.default_rng(7)
    z, y, x = 8, 48, 48
    yy, xx = np.mgrid[0:y, 0:x]
    bg = np.full((z, y, x), 200.0, dtype=np.float32)
    for zi in range(z):
        r = 8 + zi
        disk = (yy - 24) ** 2 + (xx - 24) ** 2 < r**2
        bg[zi, disk] = 50.0
    bg += rng.normal(0, 8, bg.shape).astype(np.float32)
    vol = Volume3D(bg)

    seg = MeijeringSegmenter()
    result = seg.segment(vol)
    assert result.method == "meijering"
    assert result.probability.shape == vol.shape
    assert result.probability.min() >= 0 and result.probability.max() <= 1
    assert result.mask.shape == vol.shape
    assert result.mask.any()


def test_meijering_segmenter_flat_slice_is_empty():
    vol = Volume3D(np.full((4, 16, 16), 128.0, dtype=np.float32))
    result = MeijeringSegmenter().segment(vol)
    assert result.mask.sum() == 0
    assert result.probability.max() <= 1
