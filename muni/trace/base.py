"""Interfaz común de tracing dendrítico."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TraceResult:
    """Resultado del tracing dendrítico.

    Parameters
    ----------
    coords:
        Coordenadas ``(N, 3)`` de los voxels del esqueleto en espacio de grid
        ``(z, y, x)``.
    radii:
        Radio estimado (µm) de cada voxel del esqueleto.
    parents:
        Índice del padre de cada nodo. ``-1`` para la raíz.
    branch_labels:
        Etiqueta de rama para cada nodo (0 = raíz, 1, 2, ...).
    total_length_um:
        Longitud total del esqueleto en micras.
    spacing:
        Espaciado físico ``(dz, dy, dx)`` en micras usado durante el tracing.
    method:
        Nombre del método de tracing.
    """

    coords: np.ndarray
    radii: np.ndarray
    parents: np.ndarray
    branch_labels: np.ndarray
    total_length_um: float
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    method: str = "base"

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=np.float64)
        self.radii = np.asarray(self.radii, dtype=np.float64)
        self.parents = np.asarray(self.parents, dtype=np.int64)
        self.branch_labels = np.asarray(self.branch_labels, dtype=np.int64)
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError(f"coords debe ser (N, 3), se recibió {self.coords.shape}.")
        n = len(self.coords)
        if not (len(self.radii) == n == len(self.parents) == len(self.branch_labels)):
            raise ValueError("coords, radii, parents y branch_labels deben tener la misma longitud.")

    @property
    def n_nodes(self) -> int:
        return len(self.coords)

    @property
    def n_branches(self) -> int:
        return int(self.branch_labels.max()) + 1 if len(self.branch_labels) else 0


class Tracer(ABC):
    """Contrato para motores de tracing."""

    name: str = "base"

    @abstractmethod
    def trace(
        self,
        mask: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> TraceResult:
        """Extrae el esqueleto de una máscara binaria 3D.

        Parameters
        ----------
        mask:
            Máscara binaria ``(Z, Y, X)`` donde ``True`` = neurona.
        spacing:
            Tamaño físico ``(dz, dy, dx)`` en micras.
        """
