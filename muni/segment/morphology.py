"""Operaciones morfológicas binarias implementadas desde cero (sin scipy)."""

from __future__ import annotations

import numpy as np


def _neighbor_shifts(ndim: int, structure: str) -> list[tuple[int, ...]]:
    """Desplazamientos de vecindad (sin incluir el centro) para erosión/dilatación."""
    if structure == "cross":
        shifts = []
        for axis in range(ndim):
            for step in (-1, 1):
                s = [0] * ndim
                s[axis] = step
                shifts.append(tuple(s))
        return shifts
    # estructura "box": vecindad 3x3 (2D) o 3x3x3 (3D)
    return [tuple(v) for v in np.ndindex(*([3] * ndim)) if not all(c == 1 for c in v)]


def _offsets(ndim: int, structure: str) -> list[tuple[int, ...]]:
    """Igual que ``_neighbor_shifts`` pero como desplazamientos np.roll (0..n-1)."""
    return _neighbor_shifts(ndim, structure)


def binary_dilate(mask: np.ndarray, structure: str = "cross", iterations: int = 1) -> np.ndarray:
    """Dilatación binaria (True se expande)."""
    m = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        acc = m.copy()
        for shift in _offsets(m.ndim, structure):
            acc |= np.roll(m, shift, axis=tuple(range(m.ndim)))
        m = acc
    return m


def binary_erode(mask: np.ndarray, structure: str = "cross", iterations: int = 1) -> np.ndarray:
    """Erosión binaria (True se contrae)."""
    m = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        acc = m.copy()
        for shift in _offsets(m.ndim, structure):
            acc &= np.roll(m, shift, axis=tuple(range(m.ndim)))
        m = acc
    return m


def binary_open(mask: np.ndarray, structure: str = "cross", iterations: int = 1) -> np.ndarray:
    """Apertura (erosión + dilatación): elimina píxeles aislados."""
    return binary_dilate(binary_erode(mask, structure, iterations), structure, iterations)


def binary_close(mask: np.ndarray, structure: str = "cross", iterations: int = 1) -> np.ndarray:
    """Cierre (dilatación + erosión): rellena huecos pequeños."""
    return binary_erode(binary_dilate(mask, structure, iterations), structure, iterations)
