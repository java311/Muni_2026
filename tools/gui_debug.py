"""Depurador GUI: reproducción del flujo del asistente con captura de pantalla real.

Ejecuta el pipelino igual que el wizard (hilo + set_mesh), muestra la ventana en
la plataforma nativa, hace grab() del viewport a PNG y registra cada etapa.

Uso:
    python tools/gui_debug.py --mode mc   # Marching Cubes
    python tools/gui_debug.py --mode dc   # Dual Contouring
    python tools/gui_debug.py --glb <archivo>   # cargar un GLB existente
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
import time
from pathlib import Path

faulthandler.enable()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QSurfaceFormat  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def _pump(app, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)


def _synthetic_stack():
    from muni.core.volume import Volume3D

    z, y, x = np.mgrid[0:12, 0:48, 0:48].astype(np.float32)
    d = np.sqrt((z - 6) ** 2 + (y - 24) ** 2 + (x - 24) ** 2)
    field = 220.0 - 190.0 * (1.0 / (1.0 + np.exp((d - 12.0) / 1.0)))
    return Volume3D(field.astype(np.float32), name="debug")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mc", "dc"], default="mc")
    ap.add_argument("--glb", default=None)
    args = ap.parse_args(argv)

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)

    from muni.view.main_window import MainWindow

    win = MainWindow()
    win.show()
    _pump(app, 0.5)
    print("ETAPA 1: ventana mostrada", flush=True)

    if args.glb:
        from muni.core.gltf_io import load_model

        mesh, meta = load_model(args.glb)
        print(f"ETAPA 2: GLB cargado {mesh.vertex_count}v/{mesh.face_count}f", flush=True)
        win.viewport.set_mesh(mesh)
    else:
        print("ETAPA 2: preparando reconstrucción (como el wizard)", flush=True)
        from muni.view.wizard import ReconstructionWizard

        wiz = ReconstructionWizard()
        wiz.volume = _synthetic_stack()
        wiz.paths = [Path(".") / f"p{i}.png" for i in range(wiz.volume.dimz)]
        wiz._fill_planes()
        wiz.seed_plane = wiz.volume.dimz // 2
        wiz.line = (8, 24, 39, 24)
        wiz.threshold_gray = 128
        if args.mode == "marching_cubes":
            wiz.radio_mc.setChecked(True)
        else:
            wiz.radio_dc.setChecked(True)

        wiz._run()
        print("ETAPA 3: hilo lanzado, esperando...", flush=True)
        deadline = time.monotonic() + 180.0
        while wiz.mesh is None and time.monotonic() < deadline:
            _pump(app, 0.3)
        _pump(app, 1.0)
        print(
            f"ETAPA 4: mesh={'sí' if wiz.mesh is not None else 'NO'} "
            f"result={wiz.result()} error={'n/a'}", flush=True
        )
        if wiz.mesh is None:
            print("ERROR: la reconstrucción no produjo malla", flush=True)
        else:
            print(
                f"  malla: {wiz.mesh.vertex_count}v {wiz.mesh.face_count}f "
                f"bounds={wiz.mesh.bounds()}", flush=True
            )
            # Mismo camino que MainWindow._new_from_stack
            win.viewport.set_mesh(wiz.mesh, units="um")

    _pump(app, 2.0)
    print("ETAPA 5: capturando pantalla...", flush=True)
    out_dir = Path(__file__).resolve().parent
    win.grab().save(str(out_dir / f"debug_{args.mode or 'glb'}_window.png"))
    # grabFramebuffer es la API correcta para QOpenGLWidget (renderiza el contenido GL).
    fb = win.viewport.grabFramebuffer()
    fb.save(str(out_dir / f"debug_{args.mode or 'glb'}_viewport_fb.png"))
    win.viewport.grab().save(str(out_dir / f"debug_{args.mode or 'glb'}_viewport.png"))
    print(f"ETAPA 6: capturas guardadas en {out_dir}", flush=True)

    QTimer.singleShot(300, app.quit)
    app.exec()
    os._exit(0)  # noqa: PLR1722 - evitar el segfault de teardown GL en esta máquina


if __name__ == "__main__":
    import os

    raise SystemExit(main())