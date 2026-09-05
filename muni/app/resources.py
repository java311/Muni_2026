"""Localización de los assets reutilizados del Muni antiguo.

Funciona en desarrollo (raíz del repositorio) y empaquetado con PyInstaller
(``sys._MEIPASS/assets``).
"""

from __future__ import annotations

import sys
from pathlib import Path


def assets_dir() -> Path:
    if getattr(sys, "frozen", False):  # empaquetado con PyInstaller
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:  # desarrollo: muni/app -> raíz del repo
        base = Path(__file__).resolve().parents[2]
    return base / "assets"


def icon_path(name: str) -> str:
    """Devuelve la ruta de un asset, buscando en raíz e imágenes/iconos."""
    root = assets_dir()
    for candidate in (root / name, root / "images" / name, root / "icons" / name):
        if candidate.exists():
            return str(candidate)
    return str(root / name)


def load_icon(name: str):
    """Carga un icono desde assets sin fallar si falta."""
    from PySide6.QtGui import QIcon

    return QIcon(icon_path(name))