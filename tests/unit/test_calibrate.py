"""Pruebas de calibración (F4)."""

import numpy as np

from muni.calibrate.isolevel import otsu_gray_threshold
from muni.calibrate.zcalib import measure_on_grayscale, measure_on_mask
from muni.core.volume import Volume3D


def _cylinder_volume():
    z, y, x = 12, 40, 40
    vol = np.full((z, y, x), 200.0, dtype=np.float32)
    yy, xx = np.mgrid[0:y, 0:x]
    disk = (yy - 20) ** 2 + (xx - 20) ** 2 < 25  # radio 5 → 9 px de ancho en y=20
    for p in range(2, 10):
        vol[p][disk] = 20.0
    return Volume3D(vol)


def test_measure_on_grayscale_cylinder():
    vol = _cylinder_volume()
    cal = measure_on_grayscale(vol, isolevel=100.0, seed_plane=5, line=(0, 20, 39, 20))
    assert cal.z_extent_planes == 8  # planos 2..9
    assert cal.max_diameter_px == 9  # ancho del disco en y=20
    # Resoluciones asumiendo dendrita de 4 µm de diámetro.
    assert cal.spacing_xy_um(4.0) == 4.0 / 9
    assert cal.spacing_z_um(4.0) == 4.0 / 8


def test_measure_on_mask_cylinder():
    mask = (_cylinder_volume().data <= 100.0)
    cal = measure_on_mask(mask, seed_plane=5, line=(0, 20, 39, 20))
    assert cal.z_extent_planes == 8
    assert cal.max_diameter_px == 9


def test_otsu_gray_threshold_bimodal():
    # Histograma multinivel (como una imagen real), no perfectamente binario.
    z, y, x = 4, 30, 30
    xx = np.arange(x, dtype=np.float32)
    vol = np.full((z, y, x), 200.0, dtype=np.float32)
    vol += (xx % 40)[None, None, :]  # fondo con variación suave
    vol[:, 10:20, 10:20] = 40.0 + (xx[None, None, 10:20] % 20)  # primer plano
    t = otsu_gray_threshold(Volume3D(vol), denoise=False)
    assert 40 < t < 220
