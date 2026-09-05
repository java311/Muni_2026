"""Metadatos del modelo: calibración, parámetros y procedencia.

Estos metadatos no caben de forma nativa en GLTF, así que se guardan en un
sidecar ``*.muni.json`` junto al ``.glb``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1


@dataclass
class ModelMeta:
    """Metadatos científicos asociados a un modelo reconstruido."""

    # Dimensiones y espaciado físico
    dimx: int = 0
    dimy: int = 0
    dimz: int = 0
    spacing_xy_um: float = 1.0
    spacing_z_um: float = 1.0

    # Calibración
    isolevel: float = 0.5
    objective: int = 0  # magnificación del objetivo del microscopio
    dendrite_diameter_um: float = 0.0  # diámetro de dendrita indicado por el usuario

    # Parámetros del pipeline
    segmenter: str = "classical"
    extractor: str = "dual_contouring"
    zoom_applied: bool = False
    zoom_factor: int = 1

    # Colores de escena (R,G,B en 0..255)
    colors: dict[str, list[int]] = field(
        default_factory=lambda: {
            "neuron": [0, 128, 192],
            "background": [0, 0, 0],
            "ruler": [175, 16, 44],
            "container": [255, 0, 0],
        }
    )

    # Procedencia
    source_stack: str = ""  # carpeta o patrón de la pila original
    source_hashes: list[str] = field(default_factory=list)  # sha1 de cada imagen

    # Reservado para análisis futuros (p. ej. conteo de dendritas) — NO se usa aún.
    analysis: dict[str, Any] = field(default_factory=dict)

    format_version: int = FORMAT_VERSION

    # ------------------------------------------------------------------ IO
    @classmethod
    def load(cls, path: str | Path) -> "ModelMeta":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def sidecar_path(cls, glb_path: str | Path) -> Path:
        """Dado ``modelo.glb`` devuelve ``modelo.muni.json``."""
        p = Path(glb_path)
        return p.with_suffix(".muni.json")

    def __repr__(self) -> str:  # pragma: no cover - diagnóstico
        return (
            f"ModelMeta({self.dimx}x{self.dimy}x{self.dimz}, "
            f"spacing_xy={self.spacing_xy_um}um, spacing_z={self.spacing_z_um}um, "
            f"segmenter={self.segmenter!r}, extractor={self.extractor!r})"
        )
