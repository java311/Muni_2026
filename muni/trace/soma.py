"""Detección automática del soma a partir de la máscara segmentada.

El soma se modela como una **esfera** centrada en el punto más grueso de la
máscara, con radio igual al grosor local (pico de la transformada de distancia).
Por construcción la esfera queda dentro de la máscara, así que nunca supera al
modelo. Es el ancla del árbol de neuritas; no forma parte del esqueleto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class Soma:
    """Esfera que aproxima el soma."""

    center_um: np.ndarray  # (3,) mundo (x, y, z)
    radii_um: np.ndarray  # (3,) radios (iguales, salvo refinamiento futuro)
    axes: np.ndarray  # (3, 3) autovectores como columnas (mundo)
    radius_um: float  # grosor local en el centro (pico de la EDT)


def detect_soma(mask: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Soma | None:
    """Detecta el soma en el punto más grueso de la máscara.

    El radio libre de la transformada de distancia en el pico es la distancia al
    fondo más cercano; una esfera con ese radio cabe dentro de la máscara, de
    modo que no puede desbordar el modelo. Las neuritas son mucho más finas y no
    desplazan ese pico.

    Parameters
    ----------
    mask:
        Máscara binaria 3D ``(Z, Y, X)``.
    spacing:
        Espaciado físico ``(dz, dy, dx)`` en micras.

    Returns
    -------
    El :class:`Soma` detectado o ``None`` si la máscara está vacía.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask debe ser 3D, se recibió forma {mask.shape}.")
    if not mask.any():
        return None

    dz, dy, dx = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    edt = distance_transform_edt(mask, sampling=(dz, dy, dx))
    z, y, x = np.unravel_index(int(np.argmax(edt)), edt.shape)
    radius = float(edt[z, y, x])
    if radius <= 0.0:
        return None

    center = np.array([x * dx, y * dy, z * dz], dtype=np.float64)
    radii = np.full(3, radius, dtype=np.float64)
    return Soma(
        center_um=center,
        radii_um=radii,
        axes=np.eye(3),
        radius_um=radius,
    )


def clamp_soma(soma: Soma | None, neurite_radius_um: float, max_ratio: float = 3.0) -> Soma | None:
    """Acota el radio del soma a un múltiplo del grosor de las neuritas.

    El pico de la transformada de distancia cae a veces en una zona donde se
    cruzan varias neuritas gruesas (no en el soma real), inflando el radio. Como
    el soma nunca es más de unas pocas veces el grosor de sus prolongaciones, se
    recorta a ``max_ratio * neurite_radius_um``.
    """
    if soma is None or neurite_radius_um <= 0.0:
        return soma
    limit = max_ratio * neurite_radius_um
    if soma.radius_um <= limit:
        return soma
    scale = limit / soma.radius_um
    return Soma(
        center_um=soma.center_um,
        radii_um=soma.radii_um * scale,
        axes=soma.axes,
        radius_um=limit,
    )

