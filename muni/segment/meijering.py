"""Segmentación por filtro de neuritas de Meijering + energía de foco.

Porte del pipeline de ``tests/segmentation/fmtest2.py``: combina la energía de
foco (Laplaciano suavizado) con la respuesta estrutural del filtro de
Meijering para resaltar dendritas sobre el fondo claro, y binariza con Otsu
por corte. Asume neuronas oscuras (Golgi-Cox) sobre fondo claro.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import meijering, threshold_otsu
from skimage.morphology import closing, disk, remove_small_objects

from muni.core.volume import Volume3D
from muni.segment.base import SegmentationResult, Segmenter


class MeijeringSegmenter(Segmenter):
    """Segmentador basado en Meijering (rige-ness) + Laplaciano por corte."""

    name = "meijering"

    def segment(self, volume: Volume3D) -> SegmentationResult:
        data = np.asarray(volume.data, dtype=np.float64)
        prob = np.zeros(data.shape, dtype=np.float32)
        mask = np.zeros(data.shape, dtype=bool)

        for z in range(data.shape[0]):
            img = data[z]

            # 1. Invertir: las dendritas oscuras pasan a ser crestas brillantes.
            imax = img.max()
            if imax <= 0:
                continue
            inverted = 1.0 - (img / imax)

            # 2. Energía de foco: Laplaciano absoluto suavizado.
            energy = np.abs(ndimage.laplace(inverted))
            energy = ndimage.gaussian_filter(energy, sigma=1.5)

            # 3. Energía estrutural: filtro de neuritas de Meijering.
            neurite = meijering(inverted, sigmas=range(1, 6), black_ridges=False)
            ne_max = neurite.max()
            if ne_max > 0:
                neurite = neurite / ne_max

            # 4. Combinar y normalizar (per-corte → campo de probabilidad).
            combined = energy * neurite
            c_max = combined.max()
            if c_max <= 0:
                continue
            prob[z] = combined / c_max

            # 5. Otsu por corte + limpieza morfológica.
            binarized = threshold_otsu(prob[z])
            cleaned = remove_small_objects(
                closing(prob[z] > binarized, disk(2)), min_size=50
            )
            mask[z] = cleaned

        return SegmentationResult(
            probability=prob,
            mask=mask,
            threshold=0.5,
            method=self.name,
        )