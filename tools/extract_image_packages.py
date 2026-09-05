"""Extrae los paquetes de imágenes (.rar) del sistema Muni antiguo a fixtures.

Busca ``7z`` (o ``7za``) en el PATH y en ubicaciones comunes de Windows, y
extrae cada ``image-package#N.rar`` a ``tests/fixtures/image-package-N``.

Uso:
    python tools/extract_image_packages.py [ruta_Muni-old] [destino]

Por defecto usa ``Muni-old/image-packages`` (hermano de la raíz del proyecto) y
``tests/fixtures``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

CANDIDATES = [
    "7z",
    "7za",
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
]


def find_7z() -> str | None:
    for cand in CANDIDATES:
        exe = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        if exe:
            return exe
    return None


def main(argv: list[str]) -> int:
    project_root = Path(__file__).resolve().parents[1]
    source_default = project_root.parent / "Muni-old" / "image-packages"
    dest_default = project_root / "tests" / "fixtures"

    source = Path(argv[1]) if len(argv) > 1 else source_default
    dest = Path(argv[2]) if len(argv) > 2 else dest_default

    seven = find_7z()
    if seven is None:
        print("No se encontró 7z/7za. Instala 7-Zip o pásalo en el PATH.", file=sys.stderr)
        return 1

    rars = sorted(source.glob("*.rar"))
    if not rars:
        print(f"No hay .rar en {source}", file=sys.stderr)
        return 1

    for rar in rars:
        num = rar.name.split("#")[-1].split(".")[0]
        out = dest / f"image-package-{num}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"Extrayendo {rar.name} -> {out}")
        subprocess.run([seven, "x", str(rar), f"-o{out}", "-y"], check=True)

    print("Listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
