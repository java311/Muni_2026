"""Carga y ordenado de pilas de imágenes a :class:`Volume3D`.

Solo se usa Pillow como codec de decodificación (BMP/JPG/TIFF/PNG); no hay aquí
ningún procesado de reconstrucción.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from muni.core.volume import Volume3D

DEFAULT_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".png")


def natural_key(value: str | Path) -> list[object]:
    """Clave de orden "natural": ``imagen2`` < ``imagen10``."""
    s = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", s)]


def list_images(folder: str | Path, extensions: Sequence[str] = DEFAULT_EXTENSIONS) -> list[Path]:
    """Lista las imágenes de una carpeta ordenadas de forma natural."""
    folder = Path(folder)
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    found = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(found, key=natural_key)


def _image_to_gray_float32(path: Path) -> np.ndarray:
    """Decodifica una imagen y la devuelve como gris ``float32`` en [0, 255]."""
    with Image.open(path) as im:
        im = im.convert("L")  # gris 8 bits
        arr = np.asarray(im, dtype=np.float32)
    return arr


def sha1_of_file(path: Path) -> str:
    """Hash SHA-1 de un archivo (para procedencia)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_stack(
    paths: Sequence[str | Path],
    *,
    spacing_xy_um: float = 1.0,
    spacing_z_um: float = 1.0,
    name: str = "",
    compute_hashes: bool = False,
) -> tuple[Volume3D, list[Path]]:
    """Carga una pila de imágenes en un :class:`Volume3D` ``(Z, Y, X)``.

    Parameters
    ----------
    paths:
        Rutas de las imágenes, en el orden Z deseado (se recomienda pasarlas ya
        ordenadas con :func:`natural_key`).
    spacing_xy_um, spacing_z_um:
        Espaciado físico (se puede refinar después con la calibración).
    name:
        Nombre del volumen.
    compute_hashes:
        Si es ``True`` calcula el SHA-1 de cada archivo.

    Returns
    -------
    (volumen, rutas_resueltas)

    Raises
    ------
    ValueError
        Si hay menos de 1 imagen o si las dimensiones no coinciden.
    """
    resolved = [Path(p) for p in paths]
    if not resolved:
        raise ValueError("No se proporcionaron imágenes.")

    first = _image_to_gray_float32(resolved[0])
    h, w = first.shape
    stack = np.empty((len(resolved), h, w), dtype=np.float32)
    stack[0] = first

    for i, path in enumerate(resolved[1:], start=1):
        arr = _image_to_gray_float32(path)
        if arr.shape != (h, w):
            raise ValueError(
                f"La imagen {path.name} tiene forma {arr.shape}, se esperaba {(h, w)}. "
                "Todas las imágenes de la pila deben tener el mismo tamaño."
            )
        stack[i] = arr

    vol = Volume3D(
        data=stack,
        spacing_xy_um=spacing_xy_um,
        spacing_z_um=spacing_z_um,
        name=name,
    )
    return vol, resolved


def load_folder(
    folder: str | Path,
    *,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    spacing_xy_um: float = 1.0,
    spacing_z_um: float = 1.0,
    compute_hashes: bool = False,
) -> tuple[Volume3D, list[Path]]:
    """Carga una carpeta completa, con orden natural, en un :class:`Volume3D`."""
    folder = Path(folder)
    images = list_images(folder, extensions)
    return load_stack(
        images,
        spacing_xy_um=spacing_xy_um,
        spacing_z_um=spacing_z_um,
        name=folder.name,
        compute_hashes=compute_hashes,
    )


def stack_hashes(paths: Iterable[str | Path]) -> list[str]:
    """SHA-1 de cada imagen, en el mismo orden."""
    return [sha1_of_file(Path(p)) for p in paths]
