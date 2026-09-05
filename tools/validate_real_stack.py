"""Valida el pipeline completo sobre un paquete real de imágenes (F6).

Uso:
    python tools/validate_real_stack.py [carpeta] [--extractor dual_contouring|marching_cubes]
                                       [--downsample N] [--out DIR]

Genera ``modelo.glb`` + ``modelo.muni.json`` y muestra estadísticas y tiempos.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from muni.calibrate.isolevel import otsu_gray_threshold  # noqa: E402
from muni.core.gltf_io import save_model  # noqa: E402
from muni.core.meta import ModelMeta  # noqa: E402
from muni.io.stack_loader import load_folder  # noqa: E402
from muni.postprocess.smooth import taubin_smooth  # noqa: E402
from muni.reconstruct.dual_contouring import DualContouring  # noqa: E402
from muni.reconstruct.marching_cubes import MarchingCubes  # noqa: E402
from muni.segment.classical import ClassicalSegmenter  # noqa: E402


def downsample(volume, factor: int):
    """Media-pooling (Z, Y, X) por bloques de ``factor`` en Y y X."""
    if factor <= 1:
        return volume
    data = volume.data
    h, w = data.shape[1], data.shape[2]
    h2, w2 = h // factor * factor, w // factor * factor
    d = data[:, :h2, :w2].reshape(data.shape[0], h2 // factor, factor, w2 // factor, factor)
    d = d.mean(axis=(2, 4)).astype(np.float32)
    from muni.core.volume import Volume3D

    return Volume3D(
        d,
        spacing_xy_um=volume.spacing_xy_um * factor,
        spacing_z_um=volume.spacing_z_um,
        name=volume.name,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", nargs="?", default=None, help="Carpeta con la pila")
    ap.add_argument("--extractor", default="dual_contouring", choices=["dual_contouring", "marching_cubes"])
    ap.add_argument("--downsample", type=int, default=1, help="Reducir Y,X por este factor")
    ap.add_argument("--out", default=None, help="Carpeta de salida para el GLB")
    args = ap.parse_args(argv)

    fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    folder = Path(args.folder) if args.folder else (fixtures / "image-package-2")

    t0 = time.perf_counter()
    print(f"Cargando {folder} ...", flush=True)
    volume, paths = load_folder(folder)
    if args.downsample > 1:
        volume = downsample(volume, args.downsample)
    print(f"  volumen: {volume.shape} ({volume.data.nbytes/1e6:.1f} MB)", flush=True)
    print(f"  carga: {time.perf_counter()-t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    threshold = otsu_gray_threshold(volume, denoise=True)
    print(f"Umbral de gris (Otsu): {threshold} ({time.perf_counter()-t1:.1f}s)", flush=True)

    t2 = time.perf_counter()
    print("Segmentando ...", flush=True)
    seg = ClassicalSegmenter(denoise=True, keep_largest=True).segment(volume)
    print(f"  'neurona' = {100*seg.mask.mean():.2f}% del volumen, {time.perf_counter()-t2:.1f}s", flush=True)

    t3 = time.perf_counter()
    print(f"Extrayendo superficie con {args.extractor} ...", flush=True)
    extractor = {"dual_contouring": DualContouring, "marching_cubes": MarchingCubes}[args.extractor]()
    spacing = (volume.spacing_z_um, volume.spacing_xy_um, volume.spacing_xy_um)
    result = extractor.extract(seg.probability, isovalue=0.5, spacing=spacing)
    mesh = result.mesh
    print(f"  malla: {mesh.vertex_count} vértices, {mesh.face_count} triángulos "
          f"({time.perf_counter()-t3:.1f}s)", flush=True)

    t4 = time.perf_counter()
    mesh = taubin_smooth(mesh, iterations=3)
    print(f"  suavizado: {time.perf_counter()-t4:.1f}s", flush=True)

    lo, hi = mesh.bounds()
    meta = ModelMeta(
        dimx=volume.dimx,
        dimy=volume.dimy,
        dimz=volume.dimz,
        spacing_xy_um=volume.spacing_xy_um,
        spacing_z_um=volume.spacing_z_um,
        isolevel=result.isovalue,
        objective=40,
        segmenter=seg.method,
        extractor=result.method,
        source_stack=folder.name,
    )
    out_dir = Path(args.out) if args.out else (fixtures / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    glb = out_dir / f"{folder.name}_{args.extractor}.glb"
    save_model(glb, mesh, meta)
    print(f"Guardado: {glb} (+ sidecar .muni.json)", flush=True)
    print(f"Bounding box (µm): {lo} a {hi}", flush=True)
    print(f"Tiempo total: {time.perf_counter()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())