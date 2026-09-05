"""Interfaz común de segmentación."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from muni.core.volume import Volume3D


@dataclass
class SegmentationResult:
    """Resultado de segmentar un volumen.

    Parameters
    ----------
    probability:
        Campo de probabilidad ``float32`` de forma ``(Z, Y, X)`` en [0, 1]:
        valores altos indican neurona.
    mask:
        Máscara binaria booleana del primer plano (neurona).
    threshold:
        Umbral usado para binarizar la probabilidad (por defecto 0.5).
    method:
        Nombre del método (``"classical"``, ``"unet"``...).
    """

    probability: np.ndarray
    mask: np.ndarray
    threshold: float = 0.5
    method: str = "classical"

    def __post_init__(self) -> None:
        self.probability = np.asarray(self.probability, dtype=np.float32)
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.probability.shape != self.mask.shape:
            raise ValueError(
                f"probability {self.probability.shape} y mask {self.mask.shape} deben coincidir."
            )
        self.threshold = float(self.threshold)


class Segmenter(ABC):
    """Contrato para cualquier motor de segmentación."""

    name: str = "base"

    @abstractmethod
    def segment(self, volume: Volume3D) -> SegmentationResult:
        """Segmenta un volumen y devuelve probabilidad + máscara."""
