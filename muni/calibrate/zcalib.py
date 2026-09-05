"""Calibración de la resolución en Z a partir del diámetro de una dendrita.

Conserva la idea física del Muni original: el usuario marca una dendrita e
indica su diámetro en micras; el software cuenta en cuántos planos está presente
y cuántos píxeles mide, de lo que se deriva el espaciado XY y Z.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from muni.core.volume import Volume3D


@dataclass
class DendriteCalibration:
    """Resultado de medir una dendrita a lo largo del eje Z."""

    up_plane: int
    down_plane: int
    z_extent_planes: int
    max_diameter_px: int

    def spacing_xy_um(self, diameter_um: float) -> float:
        """µm/píxel en XY asumiendo dendrita cilíndrica de ``diameter_um`` µm."""
        if self.max_diameter_px <= 0:
            raise ValueError("Diámetro en píxeles no positivo; la dendrita no se detectó.")
        return diameter_um / self.max_diameter_px

    def spacing_z_um(self, diameter_um: float) -> float:
        """µm/plano en Z asumiendo dendrita cilíndrica de ``diameter_um`` µm."""
        if self.z_extent_planes <= 0:
            raise ValueError("Extensión en Z no positiva; la dendrita no se detectó.")
        return diameter_um / self.z_extent_planes


def _sample_line(x1: float, y1: float, x2: float, y2: float) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(filas, columnas)`` de los píxeles a lo largo de la línea.

    Se muestrea un punto por píxel a lo largo del eje dominante, garantizando
    cobertura completa de líneas horizontales/verticales/diagonales.
    """
    n = max(int(round(max(abs(x2 - x1), abs(y2 - y1)))) + 1, 1)
    xs = np.rint(np.linspace(x1, x2, n)).astype(int)
    ys = np.rint(np.linspace(y1, y2, n)).astype(int)
    return ys, xs


def _longest_run(mask: np.ndarray) -> int:
    """Longitud máxima de una racha de ``True``."""
    if not mask.any():
        return 0
    # Bordes para detectar transiciones.
    padded = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])
    return int((ends - starts).max()) if starts.size else 0


def measure_on_grayscale(
    volume: Volume3D,
    isolevel: float,
    seed_plane: int,
    line: Sequence[float],
) -> DendriteCalibration:
    """Mide la extensión de la dendrita en un volumen en gris (neurona oscura).

    Parameters
    ----------
    volume:
        Volumen ``(Z, Y, X)`` en gris.
    isolevel:
        Umbral de gris: píxeles ``<= isolevel`` son dendrita.
    seed_plane:
        Plano donde el usuario marcó la dendrita.
    line:
        ``(x1, y1, x2, y2)`` de la línea que cruza la dendrita.
    """
    x1, y1, x2, y2 = (float(v) for v in line)
    ys, xs = _sample_line(x1, y1, x2, y2)

    def probe(plane: int) -> tuple[bool, int]:
        inside = volume.plane(plane)[ys, xs] <= isolevel
        run = _longest_run(inside)
        return run > 0, run

    return _measure(probe, seed_plane, volume.dimz)


def measure_on_mask(
    mask: np.ndarray,
    seed_plane: int,
    line: Sequence[float],
) -> DendriteCalibration:
    """Igual que :func:`measure_on_grayscale` pero sobre una máscara booleana."""
    x1, y1, x2, y2 = (float(v) for v in line)
    ys, xs = _sample_line(x1, y1, x2, y2)
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise ValueError("La máscara debe ser 3D (Z, Y, X).")

    def probe(plane: int) -> tuple[bool, int]:
        inside = m[plane][ys, xs]
        run = _longest_run(inside)
        return run > 0, run

    return _measure(probe, seed_plane, m.shape[0])


def _measure(probe, seed_plane: int, nplanes: int) -> DendriteCalibration:
    """Común a gris y máscara: busca el tramo conexo de planos con dendrita."""
    if not 0 <= seed_plane < nplanes:
        raise IndexError(f"Plano semilla {seed_plane} fuera de rango [0, {nplanes}).")

    max_diameter = 0
    seed_present, run = probe(seed_plane)
    max_diameter = max(max_diameter, run)

    up = seed_plane
    while up - 1 >= 0:
        present, run = probe(up - 1)
        max_diameter = max(max_diameter, run)
        if not present:
            break
        up -= 1

    down = seed_plane
    while down + 1 < nplanes:
        present, run = probe(down + 1)
        max_diameter = max(max_diameter, run)
        if not present:
            break
        down += 1

    extent = (down - up + 1) if seed_present else 0
    return DendriteCalibration(
        up_plane=up,
        down_plane=down,
        z_extent_planes=extent,
        max_diameter_px=max_diameter,
    )
