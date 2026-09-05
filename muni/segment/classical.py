"""Segmentación clásica (sin modelos entrenados), escrita desde cero.

Pipeline por defecto, siempre disponible y reproducible:

1. Normalización y difusión anisotrópica (opcional) para reducir ruido.
2. Umbral global de **Otsu** (la neurona teñida es oscura).
3. Probabilidad suave por sigmoide alrededor del umbral.
4. Limpieza morfológica (apertura/cierre) de la máscara.
5. Conservación del componente conexo principal (opcional).

Sirve además para generar pseudo-etiquetas con las que entrenar el motor U-Net.
"""

from __future__ import annotations

import numpy as np

from muni.core.volume import Volume3D
from muni.preprocess.denoise import anisotropic_diffusion, normalize
from muni.segment.base import Segmenter, SegmentationResult
from muni.segment.connected import largest_component
from muni.segment.morphology import binary_close, binary_open
from muni.segment.threshold import otsu_threshold, sigmoid_probability


class ClassicalSegmenter(Segmenter):
    """Segmentador clásico basado en umbral de Otsu + morfología."""

    name = "classical"

    def __init__(
        self,
        *,
        denoise: bool = True,
        denoise_iterations: int = 5,
        denoise_kappa: float = 15.0,
        threshold_scale: float | None = None,
        open_iterations: int = 1,
        close_iterations: int = 1,
        keep_largest: bool = True,
        binary_threshold: float = 0.5,
    ) -> None:
        self.denoise = denoise
        self.denoise_iterations = denoise_iterations
        self.denoise_kappa = denoise_kappa
        self.threshold_scale = threshold_scale
        self.open_iterations = open_iterations
        self.close_iterations = close_iterations
        self.keep_largest = keep_largest
        self.binary_threshold = binary_threshold

    def segment(self, volume: Volume3D) -> SegmentationResult:
        data = normalize(volume.data, (0.0, 255.0))
        if self.denoise:
            data = anisotropic_diffusion(
                data,
                iterations=self.denoise_iterations,
                kappa=self.denoise_kappa,
                gamma=0.2,
            )

        # Umbral global de Otsu sobre el histograma del volumen completo.
        t = otsu_threshold(data)

        prob = sigmoid_probability(data, t, self.threshold_scale)
        mask = prob >= self.binary_threshold

        # Limpieza morfológica.
        if self.open_iterations:
            mask = binary_open(mask, structure="cross", iterations=self.open_iterations)
        if self.close_iterations:
            mask = binary_close(mask, structure="cross", iterations=self.close_iterations)

        if self.keep_largest:
            mask = largest_component(mask)

        return SegmentationResult(
            probability=prob.astype(np.float32),
            mask=mask,
            threshold=self.binary_threshold,
            method=self.name,
        )
