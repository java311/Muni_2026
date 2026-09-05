"""Filtros de suavizado que preservan bordes, implementados desde cero.

La difusión anisotrópica (Perona–Malik) reduce ruido sin difuminar los bordes
de las dendritas, que es justo lo que el antiguo Muni perdía al promediar RGB.
"""

from __future__ import annotations

import numpy as np

from muni.core.volume import Volume3D


def _conductance(grad: np.ndarray, kappa: float, mode: str) -> np.ndarray:
    """Conductancia de Perona–Malik en función del gradiente normalizado."""
    g = (grad / kappa) ** 2
    if mode == "exp":
        return np.exp(-g)
    # modo "quad" (inverse quadratic), por defecto
    return 1.0 / (1.0 + g)


def anisotropic_diffusion(
    image: np.ndarray,
    iterations: int = 10,
    kappa: float = 15.0,
    gamma: float = 0.2,
    mode: str = "quad",
) -> np.ndarray:
    """Difusión anisotrópica de Perona–Malik.

    Acepta arrays 2D ``(H, W)`` o 3D ``(Z, Y, X)``. En 2D usa una vecindad de
    4 píxeles; en 3D, una vecindad de 6 vóxeles.

    Parameters
    ----------
    image:
        Imagen/volumen en ``float32``.
    iterations:
        Número de pasos de difusión.
    kappa:
        Constante de conductancia (controla qué gradiente se considera "borde").
    gamma:
        Tasa de actualización ``lambda`` (estabilidad típica: 0 <= gamma <= 1/ndim).
    mode:
        ``"exp"`` (c1) o ``"quad"`` (c2, por defecto).
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim not in (2, 3):
        raise ValueError(f"Se esperaba imagen 2D o volumen 3D, se recibió forma {img.shape}.")

    for _ in range(iterations):
        if img.ndim == 2:
            n = np.roll(img, 1, axis=0)
            s = np.roll(img, -1, axis=0)
            e = np.roll(img, -1, axis=1)
            w = np.roll(img, 1, axis=1)
            gn, gs, ge, gw = n - img, s - img, e - img, w - img
            cn = _conductance(gn, kappa, mode)
            cs = _conductance(gs, kappa, mode)
            ce = _conductance(ge, kappa, mode)
            cw = _conductance(gw, kappa, mode)
            img = img + gamma * (cn * gn + cs * gs + ce * ge + cw * gw)
        else:
            z0 = np.roll(img, 1, axis=0)
            z1 = np.roll(img, -1, axis=0)
            y0 = np.roll(img, 1, axis=1)
            y1 = np.roll(img, -1, axis=1)
            x0 = np.roll(img, 1, axis=2)
            x1 = np.roll(img, -1, axis=2)
            g = [z0 - img, z1 - img, y0 - img, y1 - img, x0 - img, x1 - img]
            c = [_conductance(gi, kappa, mode) for gi in g]
            img = img + gamma * sum(ci * gi for ci, gi in zip(c, g))

    return img


def denoise_volume(
    volume: Volume3D,
    iterations: int = 10,
    kappa: float = 15.0,
    gamma: float = 0.2,
    inplace: bool = False,
) -> Volume3D:
    """Aplica difusión anisotrópica 3D a un :class:`Volume3D`.

    Devuelve un nuevo volumen (o modifica el original si ``inplace=True``).
    """
    out = anisotropic_diffusion(volume.data, iterations, kappa, gamma)
    if inplace:
        volume.data = out
        return volume
    return Volume3D(
        data=out,
        spacing_xy_um=volume.spacing_xy_um,
        spacing_z_um=volume.spacing_z_um,
        origin=volume.origin,
        name=volume.name,
        units=volume.units,
    )


def normalize(image: np.ndarray, out_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Normaliza min-max al rango ``out_range`` (sin librerías externas)."""
    img = np.asarray(image, dtype=np.float32)
    lo, hi = out_range
    mn, mx = img.min(), img.max()
    if mx == mn:
        return np.full_like(img, (lo + hi) / 2.0)
    return lo + (img - mn) * (hi - lo) / (mx - mn)
