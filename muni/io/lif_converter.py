"""Conversión de archivos Leica (.lif) a stacks de PNG."""

from __future__ import annotations

from pathlib import Path

from readlif.reader import LifFile


def convert_lif_to_png(lif_path: str | Path, output_dir: str | Path) -> int:
    """Convierte un archivo .lif a una pila de imágenes PNG.

    Returns
    -------
    Número de archivos PNG guardados.
    """
    lif_path = Path(lif_path)
    output_dir = Path(output_dir)

    if not lif_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {lif_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    lif = LifFile(str(lif_path))
    n_images = lif.num_images

    if n_images == 0:
        raise ValueError("El archivo .lif no contiene imágenes.")

    saved = 0

    for img_idx in range(n_images):
        img = lif.get_image(img_idx)
        n_z = max(img.nz, 1)
        n_t = max(img.nt, 1)
        n_c = max(img.channels, 1)

        for t in range(n_t):
            for z in range(n_z):
                for c in range(n_c):
                    try:
                        frame = img.get_frame(z=z, t=t, c=c)
                        fname = f"series{img_idx}_z{z:04d}_t{t:02d}_c{c:02d}.png"
                        frame.save(str(output_dir / fname))
                        saved += 1
                    except Exception:  # noqa: BLE001, S110
                        pass

    return saved
