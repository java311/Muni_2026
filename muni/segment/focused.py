"""Segmentación por contraste local (nitidez) + Otsu.

Método que combina umbral de intensidad (Otsu) con filtro de nitidez para
eliminar regiones desenfocadas del stack axial.

Pipeline:

1. Normalización y difusión anisotrópica (opcional).
2. Umbral de Otsu → máscara binaria base (neurona oscura en Golgi-Cox).
3. **Desviación estándar local** por píxel: mide contraste local (alto = enfocado,
   bajo = desenfoque). Mejor métrica para brightfield axial stacks (Zhang et al. 2023).
4. Filtrado: elimina de la máscara los píxeles con nitidez baja.
5. Limpieza morfológica + componente mayor.

Ventaja sobre Otsu puro: elimina regiones desenfocadas donde la neurona tiene
bajo contraste local, reduciendo falsos positivos del desenfoque axial.
"""

from __future__ import annotations

import numpy as np

from muni.core.volume import Volume3D
from muni.preprocess.denoise import anisotropic_diffusion, normalize
from muni.segment.base import SegmentationResult, Segmenter
from muni.segment.connected import largest_component
from muni.segment.morphology import binary_close, binary_open
from muni.segment.threshold import _box_mean, otsu_threshold, sigmoid_probability


def _local_std(image: np.ndarray, radius: int = 3) -> np.ndarray:
    """Desviación estándar local: sqrt(E[X^2] - E[X]^2)."""
    img = np.asarray(image, dtype=np.float64)
    mean = _box_mean(img, radius)
    mean_sq = _box_mean(img * img, radius)
    variance = np.clip(mean_sq - mean * mean, 0, None)
    return np.sqrt(variance)


class FocusedSegmenter(Segmenter):
    """Segmentador que filtra por contraste local después de Otsu."""

    name = "focused"

    def __init__(
        self,
        *,
        denoise: bool = True,
        denoise_iterations: int = 5,
        denoise_kappa: float = 15.0,
        sharpness_radius: int = 3,
        open_iterations: int = 1,
        close_iterations: int = 1,
        keep_largest: bool = True,
    ) -> None:
        self.denoise = denoise
        self.denoise_iterations = denoise_iterations
        self.denoise_kappa = denoise_kappa
        self.sharpness_radius = sharpness_radius
        self.open_iterations = open_iterations
        self.close_iterations = close_iterations
        self.keep_largest = keep_largest

    def segment(self, volume: Volume3D) -> SegmentationResult:
        data = normalize(volume.data, (0.0, 255.0))
        if self.denoise:
            data = anisotropic_diffusion(
                data,
                iterations=self.denoise_iterations,
                kappa=self.denoise_kappa,
                gamma=0.2,
            )

        t = otsu_threshold(data)
        prob = sigmoid_probability(data, float(t))
        base_mask = prob >= 0.5

        sharpness = np.zeros_like(data, dtype=np.float64)
        for z in range(data.shape[0]):
            sharpness[z] = _local_std(data[z], radius=self.sharpness_radius)

        # Percentil 50: el 85 solo conserva el borde nítido (un anillo de 1 px)
        # que la apertura morfológica posterior borra por completo.
        s_thresh = float(np.percentile(sharpness[base_mask], 50)) if base_mask.any() else 0.0
        sharp_mask = sharpness > s_thresh

        mask = base_mask & sharp_mask

        if self.open_iterations:
            mask = binary_open(mask, structure="cross", iterations=self.open_iterations)
        if self.close_iterations:
            mask = binary_close(mask, structure="cross", iterations=self.close_iterations)

        if self.keep_largest:
            mask = largest_component(mask)

        return SegmentationResult(
            probability=prob.astype(np.float32),
            mask=mask,
            threshold=0.5,
            method=self.name,
        )
