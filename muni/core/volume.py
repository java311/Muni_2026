"""Volumen escalar 3D con metadatos físicos.

Convención de ejes: el array ``data`` tiene forma ``(Z, Y, X)``, es decir, el
primer índice es el plano de la pila, el segundo la fila (alto) y el tercero la
columna (ancho). ``data[z]`` es una imagen 2D completa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Volume3D:
    """Campo escalar tridimensional (pila de cortes) con espaciado físico.

    Parameters
    ----------
    data:
        Array ``float32`` de forma ``(Z, Y, X)``.
    spacing_xy_um:
        Tamaño del píxel en micras en los ejes X e Y.
    spacing_z_um:
        Distancia entre planos consecutivos en micras (eje Z).
    origin:
        Coordenada física de la esquina ``(z=0, y=0, x=0)`` en micras.
    name:
        Nombre opcional (p. ej. el nombre del paquete).
    units:
        Unidad de ``spacing`` y ``origin``.
    """

    data: np.ndarray
    spacing_xy_um: float = 1.0
    spacing_z_um: float = 1.0
    origin: Sequence[float] = (0.0, 0.0, 0.0)
    name: str = ""
    units: str = "um"

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        if data.ndim != 3:
            raise ValueError(f"Volume3D requiere un array 3D, se recibió forma {data.shape}.")
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
        self.data = data
        self.origin = tuple(float(v) for v in self.origin)
        self.spacing_xy_um = float(self.spacing_xy_um)
        self.spacing_z_um = float(self.spacing_z_um)
        if self.spacing_xy_um <= 0 or self.spacing_z_um <= 0:
            raise ValueError("Los espaciados deben ser positivos.")

    # ------------------------------------------------------------------ shape
    @property
    def shape(self) -> tuple[int, int, int]:
        """Forma ``(Z, Y, X)``."""
        return int(self.data.shape[0]), int(self.data.shape[1]), int(self.data.shape[2])

    @property
    def dimz(self) -> int:
        return self.shape[0]

    @property
    def dimy(self) -> int:
        return self.shape[1]

    @property
    def dimx(self) -> int:
        return self.shape[2]

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    # ---------------------------------------------------------------- helpers
    def min(self) -> float:
        return float(np.min(self.data))

    def max(self) -> float:
        return float(np.max(self.data))

    def plane(self, z: int) -> np.ndarray:
        """Devuelve el corte 2D ``(Y, X)`` en el plano ``z``."""
        if not 0 <= z < self.dimz:
            raise IndexError(f"Plano {z} fuera de rango [0, {self.dimz}).")
        return self.data[z]

    # -------------------------------------------------------------- units
    def physical_size(self) -> tuple[float, float, float]:
        """Tamaño físico ``(Z, Y, X)`` del volumen en micras."""
        dz = self.dimz * self.spacing_z_um
        dy = self.dimy * self.spacing_xy_um
        dx = self.dimx * self.spacing_xy_um
        return dz, dy, dx

    def __repr__(self) -> str:  # pragma: no cover - diagnóstico
        return (
            f"Volume3D(shape={self.shape}, dtype={self.dtype}, "
            f"spacing_xy={self.spacing_xy_um}{self.units}, "
            f"spacing_z={self.spacing_z_um}{self.units}, name={self.name!r})"
        )
