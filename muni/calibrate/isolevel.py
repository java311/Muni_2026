"""Umbral (isonivel) automático para el modo manual/asistido."""

from __future__ import annotations

import numpy as np

from muni.core.volume import Volume3D
from muni.preprocess.denoise import anisotropic_diffusion
from muni.segment.threshold import otsu_threshold


def otsu_gray_threshold(volume: Volume3D, denoise: bool = True, iterations: int = 5) -> int:
    """Umbral de gris de Otsu sobre el volumen, en las unidades originales.

    Devuelve el valor ``t`` tal que los píxeles ``<= t`` son neurona (teñida
    oscura). Sustituye al isonivel manual del Muni original por una estimación
    automática, devolviendo el umbral en la misma escala de intensidad que
    ``volume.data`` para poder compararlo directamente.
    """
    raw = np.asarray(volume.data, dtype=np.float32)
    mn, mx = float(raw.min()), float(raw.max())

    if mx <= mn:
        return int(round(mn))

    # Normalizar a [0, 255] solo para el cálculo interno.
    data = (raw - mn) * (255.0 / (mx - mn))
    if denoise:
        data = anisotropic_diffusion(data, iterations=iterations, kappa=15.0, gamma=0.2)

    t_norm = otsu_threshold(data)
    # Mapear de vuelta a la escala original de intensidad.
    return int(round(mn + (t_norm / 255.0) * (mx - mn)))


def probability_threshold(volume: Volume3D) -> float:
    """Umbral de probabilidad por defecto para la extracción de superficie."""
    return 0.5
