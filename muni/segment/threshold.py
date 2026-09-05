"""Umbralización automática implementada desde cero (sin OpenCV/scikit-image)."""

from __future__ import annotations

import numpy as np


def _to_uint8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.dtype == np.uint8:
        return img
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def otsu_threshold(image: np.ndarray) -> int:
    """Umbral de Otsu (maximiza la varianza entre clases).

    Devuelve el valor de gris ``t`` tal que los píxeles ``<= t`` se consideran
    primer plano (en Golgi-Cox, la neurona teñida es más oscura).
    """
    img = _to_uint8(image)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = float(img.size)
    if total == 0:
        return 0

    sum_all = float(np.dot(np.arange(256, dtype=np.float64), hist))
    sum_bg = 0.0
    weight_bg = 0.0
    best = -1.0
    best_t = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best:
            best = between
            best_t = t

    return best_t


def _box_filter_axis(img: np.ndarray, axis: int, radius: int) -> np.ndarray:
    """Filtro de caja (media) a lo largo de un eje, con bordes reflejados.

    Con ``np.cumsum`` exclusivo (cero al inicio), la suma de la ventana del
    píxel ``i`` es ``cum[i+k] - cum[i]`` con ``k = 2r+1``.
    """
    k = 2 * radius + 1
    n = img.shape[axis]
    pad = [(0, 0)] * img.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(img, pad, mode="edge")
    cum = np.cumsum(padded, axis=axis)

    # Anteponemos un cero a lo largo del eje para poder indexar cum[i] (i=0 incluido).
    zero_sl = [slice(None)] * img.ndim
    zero_sl[axis] = slice(0, 1)
    zero = np.zeros_like(cum[tuple(zero_sl)])
    cum = np.concatenate([zero, cum], axis=axis)

    hi = [slice(None)] * img.ndim
    hi[axis] = slice(k, k + n)
    lo = [slice(None)] * img.ndim
    lo[axis] = slice(0, n)
    return (cum[tuple(hi)] - cum[tuple(lo)]) / k


def _box_mean(image: np.ndarray, radius: int) -> np.ndarray:
    """Media local en ventana de radio ``radius`` (separable por eje)."""
    out = np.asarray(image, dtype=np.float64)
    for axis in range(out.ndim):
        out = _box_filter_axis(out, axis, radius)
    return out


def adaptive_mean_threshold(
    image: np.ndarray, radius: int = 7, C: float = 0.0
) -> np.ndarray:
    """Umbral adaptativo por media local (para iluminación no uniforme).

    Un píxel es primer plano si ``intensidad <= media_local - C``.
    Devuelve una máscara booleana.
    """
    img = np.asarray(image, dtype=np.float64)
    mean = _box_mean(img, radius)
    return img <= (mean - C)


def sigmoid_probability(
    image: np.ndarray, threshold: float, scale: float | None = None
) -> np.ndarray:
    """Convierte intensidad en probabilidad de "neurona" con una sigmoide suave.

    Como la neurona es oscura, ``p = sigmoid((umbral - intensidad)/escala)``.
    Si ``scale`` es ``None`` se estima como el 5 % del rango de la imagen.
    """
    img = np.asarray(image, dtype=np.float64)
    if scale is None:
        rng = float(img.max() - img.min())
        scale = max(rng * 0.05, 1.0)
    z = (threshold - img) / scale
    # sigmoide estable
    return (1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))).astype(np.float32)
