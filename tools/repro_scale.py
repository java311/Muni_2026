"""Reproduce el bug de escala: flujo del asistente (con calibración de dendrita)
sobre la pila real, y captura el render para inspección.

Uso: python tools/repro_scale.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faulthandler  # noqa: E402
import numpy as np  # noqa: E402

faulthandler.enable()

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QSurfaceFormat  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def main() -> int:
    from muni.calibrate.zcalib import measure_on_grayscale
    from muni.io.stack_loader import load_folder
    from muni.reconstruct.marching_cubes import MarchingCubes
    from muni.segment.classical import ClassicalSegmenter

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)

    vol, _ = load_folder("tests/fixtures/image-package-2")
    print("volumen:", vol.shape, flush=True)

    # Calibración Z como el asistente: línea horizontal cruzando la celda en un plano medio.
    line = (0, 287, 763, 287)
    cal = measure_on_grayscale(vol, 108, vol.dimz // 2, line)
    diam = 10.0
    if cal.z_extent_planes > 0 and cal.max_diameter_px > 0:
        vol.spacing_xy_um = cal.spacing_xy_um(diam)
        vol.spacing_z_um = cal.spacing_z_um(diam)
    print(f"calibración: extent={cal.z_extent_planes} dpx={cal.max_diameter_px} "
          f"xy={vol.spacing_xy_um:.4f}µm/px z={vol.spacing_z_um:.4f}µm/plano", flush=True)

    seg = ClassicalSegmenter(denoise=True, keep_largest=True).segment(vol)
    mesh = MarchingCubes(inside_high=True).extract(
        seg.probability, 0.5,
        spacing=(vol.spacing_z_um, vol.spacing_xy_um, vol.spacing_xy_um),
    ).mesh
    lo, hi = mesh.bounds()
    ex = hi - lo
    print(f"malla física (µm) min={lo.round(2)} max={hi.round(2)}", flush=True)
    print(f"extensión física (µm): Z={ex[0]:.1f} Y={ex[1]:.1f} X={ex[2]:.1f}", flush=True)

    from muni.view.main_window import MainWindow

    win = MainWindow()
    win.viewport.set_mesh(mesh)
    win.show()
    # varias pasadas de pintado para asegurar el framebuffer
    for _ in range(5):
        app.processEvents()
        time.sleep(0.1)
    fb = win.viewport.grabFramebuffer()
    out = Path("tools/repro_scale_viewport.png")
    fb.save(str(out))
    print(f"captura: {out}", flush=True)
    QTimer.singleShot(200, app.quit)
    app.exec()
    os._exit(0)  # noqa: PLR1722


if __name__ == "__main__":
    raise SystemExit(main())