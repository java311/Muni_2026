"""Interfaz común de tracing dendrítico."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# Tipos SWC (convención estándar de morfología neuronal).
SWC_SOMA = 1
SWC_AXON = 2
SWC_DENDRITE = 3
SWC_SPINE = 4


@dataclass
class TraceResult:
    """Resultado del tracing: bosque de esqueletos tipados.

    Parameters
    ----------
    coords:
        Coordenadas ``(N, 3)`` de los voxels del esqueleto en espacio de grid
        ``(z, y, x)``.
    radii:
        Radio estimado (µm) de cada voxel del esqueleto.
    parents:
        Índice del padre de cada nodo. ``-1`` para cada raíz (puede haber más
        de una si el volumen tiene componentes conexos separados).
    branch_labels:
        Etiqueta de rama para cada nodo (0, 1, 2, ...).
    total_length_um:
        Longitud total del esqueleto en micras.
    spacing:
        Espaciado físico ``(dz, dy, dx)`` en micras usado durante el tracing.
    method:
        Nombre del método de tracing.
    types:
        Tipo SWC de cada nodo (1 = soma, 2 = axón, 3 = dendrita, 4 = espina).
        Si es ``None`` todos los nodos se marcan como dendrita.
    soma_center_um, soma_radii_um, soma_axes:
        Elipsoide del soma detectado, en coordenadas físicas de mundo
        ``(x, y, z)``. ``soma_axes`` tiene los autovectores como columnas.
    """

    coords: np.ndarray
    radii: np.ndarray
    parents: np.ndarray
    branch_labels: np.ndarray
    total_length_um: float
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    method: str = "base"
    types: np.ndarray | None = None
    soma_center_um: np.ndarray | None = None
    soma_radii_um: np.ndarray | None = None
    soma_axes: np.ndarray | None = None

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
        if self.types is None:
            self.types = np.full(n, SWC_DENDRITE, dtype=np.int64)
        else:
            self.types = np.asarray(self.types, dtype=np.int64)
            if len(self.types) != n:
                raise ValueError("types debe tener la misma longitud que coords.")
        for attr in ("soma_center_um", "soma_radii_um", "soma_axes"):
            value = getattr(self, attr)
            if value is not None:
                setattr(self, attr, np.asarray(value, dtype=np.float64))

    @property
    def n_nodes(self) -> int:
        return len(self.coords)

    @property
    def n_branches(self) -> int:
        return int(self.branch_labels.max()) + 1 if len(self.branch_labels) else 0

    @property
    def has_soma(self) -> bool:
        return self.soma_center_um is not None and self.soma_radii_um is not None

    def world_coords(self) -> np.ndarray:
        """Coordenadas de los nodos en µm de mundo ``(x, y, z)``."""
        world = np.empty((self.n_nodes, 3), dtype=np.float64)
        world[:, 0] = self.coords[:, 2] * self.spacing[2]
        world[:, 1] = self.coords[:, 1] * self.spacing[1]
        world[:, 2] = self.coords[:, 0] * self.spacing[0]
        return world


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
