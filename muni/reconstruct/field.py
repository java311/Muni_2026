"""Construcción del campo escalar a partir de la segmentación.

El campo escalar es lo que consume el extractor de superficie. Se prefiere el
campo de **probabilidad** (suave) a la máscara binaria, porque permite
interpolar los vértices con precisión sub-vóxel.
"""

from __future__ import annotations

import numpy as np

from muni.segment.base import SegmentationResult


def probability_field(segmentation: SegmentationResult) -> np.ndarray:
    """Devuelve el campo de probabilidad ``float32`` ``(Z, Y, X)``."""
    return np.asarray(segmentation.probability, dtype=np.float32)


def binary_field(segmentation: SegmentationResult) -> np.ndarray:
    """Devuelve un campo binario ``float32`` ``(Z, Y, X)`` a partir de la máscara."""
    return np.asarray(segmentation.mask, dtype=np.float32)


def scalar_field(segmentation: SegmentationResult, kind: str = "probability") -> np.ndarray:
    """Campo escalar listo para extracción.

    ``kind``: ``"probability"`` (por defecto, suave) o ``"binary"``.
    """
    if kind == "probability":
        return probability_field(segmentation)
    if kind == "binary":
        return binary_field(segmentation)
    raise ValueError(f"kind desconocido: {kind!r}")
