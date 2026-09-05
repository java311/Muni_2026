"""Realce de contraste implementado desde cero (sin OpenCV).

Solo ecualización global de histograma y estiramiento de contraste lineales.
Un CLAHE real puede añadirse después si los datos lo requieren.
"""

from __future__ import annotations

import numpy as np


def contrast_stretch(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Estiramiento lineal de contraste por percentiles ``low``/``high``."""
    img = np.asarray(image, dtype=np.float32)
    plo, phi = np.percentile(img, [low, high])
    if phi == plo:
        return img
    return np.clip((img - plo) * (255.0 / (phi - plo)), 0.0, 255.0)


def histogram_equalize(image: np.ndarray, nbins: int = 256) -> np.ndarray:
    """Ecualización global de histograma para imágenes ``uint8`` o ``float``.

    Se asume intensidad en [0, 255].
    """
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)

    hist, _ = np.histogram(img, bins=nbins, range=(0, 256))
    cdf = hist.cumsum()
    cdf_min = int(cdf[cdf > 0].min()) if np.any(cdf > 0) else 0
    total = img.size - cdf_min
    if total <= 0:
        return img.astype(np.float32)

    # Tabla de transformación h(v) = round((cdf(v)-cdf_min)/(N-cdf_min) * 255)
    table = np.clip(np.rint((cdf - cdf_min) * 255.0 / total), 0, 255).astype(np.uint8)
    return table[img].astype(np.float32)
