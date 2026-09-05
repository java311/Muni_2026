"""Interfaz común de extracción de superficie."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from muni.core.meshdata import MeshData


@dataclass
class SurfaceExtractionResult:
    """Resultado de extraer una superficie de un campo escalar."""

    mesh: MeshData
    isovalue: float = 0.5
    method: str = "marching_cubes"

    def __post_init__(self) -> None:
        self.isovalue = float(self.isovalue)


class SurfaceExtractor(ABC):
    """Contrato para los extractores de superficie."""

    name: str = "base"

    @abstractmethod
    def extract(
        self,
        field: np.ndarray,
        isovalue: float = 0.5,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> SurfaceExtractionResult:
        """Extrae la isosuperficie ``field == isovalue``.

        Parameters
        ----------
        field:
            Campo escalar ``float32`` de forma ``(Z, Y, X)``.
        isovalue:
            Valor de la isosuperficie.
        spacing:
            Tamaño físico ``(dz, dy, dx)`` (o ``(1,1,1)`` para unidades de grid).
        """
