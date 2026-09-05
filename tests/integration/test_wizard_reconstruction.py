"""Prueba del flujo completo del asistente: pulsa 'Reconstruir' y espera la malla.

Regresión del bug donde ``run_in_thread`` perdía la referencia al Worker y la
tarea jamás se ejecutaba (el botón no hacía nada).
"""

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pyside = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from muni.core.volume import Volume3D  # noqa: E402
from muni.view.wizard import ReconstructionWizard  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def _synthetic_stack():
    """Pequeño volumen: esfera oscura sobre fondo claro."""
    z, y, x = np.mgrid[0:10, 0:36, 0:36].astype(np.float32)
    d = np.sqrt((z - 5) ** 2 + (y - 18) ** 2 + (x - 18) ** 2)
    field = 220.0 - 190.0 * (1.0 / (1.0 + np.exp((d - 9.0) / 1.0)))
    return Volume3D(field.astype(np.float32), name="prueba")


def _pump(app, timeout_s: float) -> None:
    """Procesa eventos hasta que pasen ``timeout_s`` (sin bloquear el hilo de la tarea)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)


@pytest.mark.parametrize("method", ["dual_contouring", "marching_cubes"])
def test_wizard_reconstruction_runs(app, method):
    wiz = ReconstructionWizard()
    wiz.volume = _synthetic_stack()
    wiz.paths = [Path(".") / f"p{i}.png" for i in range(wiz.volume.dimz)]
    wiz._fill_planes()
    wiz.seed_plane = wiz.volume.dimz // 2
    # Línea cruzando la esfera (de borde a borde por su centro).
    wiz.line = (8, 18, 27, 18)
    wiz.threshold_gray = 128
    if method == "marching_cubes":
        wiz.radio_mc.setChecked(True)
    else:
        wiz.radio_dc.setChecked(True)

    wiz._run()
    # Esperar a que la tarea acabe (malla lista -> accept()) o falle.
    deadline = time.monotonic() + 90.0
    while wiz.mesh is None and time.monotonic() < deadline:
        _pump(app, 0.3)
    _pump(app, 0.5)

    assert wiz.mesh is not None, "La reconstrucción no produjo malla (¿se ejecutó el hilo?)"
    assert not wiz.mesh.is_empty
    assert wiz.meta is not None
    assert wiz.meta.extractor == method
    assert wiz.result() == ReconstructionWizard.DialogCode.Accepted
    wiz.deleteLater()
    app.processEvents()