"""Prueba del flujo completo del asistente: segmentar → esqueleto → malla.

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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from muni.core.volume import Volume3D
from muni.trace.graph import branch_ancestors
from muni.view.wizard import ReconstructionWizard


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def _synthetic_stack():
    """Soma oscuro + una dendrita horizontal sobre fondo claro."""
    z, y, x = np.mgrid[0:10, 0:36, 0:36].astype(np.float32)
    d = np.sqrt((z - 5) ** 2 + (y - 18) ** 2 + (x - 14) ** 2)
    soma = 220.0 - 190.0 * (1.0 / (1.0 + np.exp((d - 6.0) / 1.0)))
    dendrite = (np.abs(y - 18) <= 2) & (np.abs(z - 5) <= 2) & (x >= 14) & (x < 34)
    field = np.where(dendrite, 30.0, soma)
    return Volume3D(field.astype(np.float32), name="prueba")


def _pump(app, timeout_s: float) -> None:
    """Procesa eventos hasta que pasen ``timeout_s`` (sin bloquear el hilo de la tarea)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)


def _wait_until(app, predicate, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("La tarea no completó a tiempo")
        _pump(app, 0.3)


@pytest.mark.parametrize("method", ["dual_contouring", "marching_cubes"])
def test_wizard_reconstruction_runs(app, method):
    wiz = ReconstructionWizard()
    wiz.volume = _synthetic_stack()
    wiz.paths = [Path(".") / f"p{i}.png" for i in range(wiz.volume.dimz)]
    wiz._fill_planes()
    wiz._fill_seg_planes()
    wiz._fill_skel_planes()
    wiz.seed_plane = wiz.volume.dimz // 2
    wiz.line = (8, 18, 27, 18)
    wiz.threshold_gray = 128
    wiz._apply_calibration()
    assert wiz.spacing is not None
    if method == "marching_cubes":
        wiz.radio_mc.setChecked(True)
    else:
        wiz.radio_dc.setChecked(True)

    wiz._preview_segmentation()
    _wait_until(app, lambda: wiz.seg_result is not None, 90.0)

    wiz._generate_skeleton()
    _wait_until(app, lambda: wiz.trace_result is not None, 90.0)

    wiz._preview_model()
    _wait_until(app, lambda: wiz._pending_mesh is not None, 90.0)
    _pump(app, 0.3)

    wiz._finish()
    _pump(app, 0.5)

    assert wiz.mesh is not None, "La reconstrucción no produjo malla"
    assert not wiz.mesh.is_empty
    assert wiz.meta is not None
    assert wiz.meta.extractor == method
    assert wiz.meta.analysis.get("total_length_um", 0.0) > 0.0
    assert wiz.result() == ReconstructionWizard.DialogCode.Accepted
    wiz.deleteLater()
    app.processEvents()


def test_wizard_can_build_tube_mesh_from_skeleton(app):
    wiz = ReconstructionWizard()
    wiz.volume = _synthetic_stack()
    wiz.paths = [Path(".") / f"p{i}.png" for i in range(wiz.volume.dimz)]
    wiz._fill_planes()
    wiz._fill_seg_planes()
    wiz._fill_skel_planes()
    wiz.seed_plane = wiz.volume.dimz // 2
    wiz.line = (8, 18, 27, 18)
    wiz.threshold_gray = 128
    wiz._apply_calibration()

    wiz._preview_segmentation()
    _wait_until(app, lambda: wiz.seg_result is not None, 90.0)
    wiz._generate_skeleton()
    _wait_until(app, lambda: wiz.trace_result is not None, 90.0)

    wiz.radio_tubes.setChecked(True)
    wiz._preview_model()
    _wait_until(app, lambda: wiz._pending_mesh is not None, 90.0)
    _pump(app, 0.3)

    wiz._finish()
    _pump(app, 0.3)
    assert wiz.mesh is not None and not wiz.mesh.is_empty
    assert wiz.meta is not None and wiz.meta.extractor == "skeleton_tubes"
    wiz.deleteLater()
    app.processEvents()


def _reach_skeleton(app, stack=None, line=(8, 18, 27, 18)):
    wiz = ReconstructionWizard()
    wiz.volume = stack if stack is not None else _synthetic_stack()
    wiz.paths = [Path(".") / f"p{i}.png" for i in range(wiz.volume.dimz)]
    wiz._fill_planes()
    wiz._fill_seg_planes()
    wiz._fill_skel_planes()
    wiz.seed_plane = wiz.volume.dimz // 2
    wiz.line = line
    wiz.threshold_gray = 128
    wiz._apply_calibration()
    wiz._preview_segmentation()
    _wait_until(app, lambda: wiz.seg_result is not None, 90.0)
    wiz._generate_skeleton()
    _wait_until(app, lambda: wiz.trace_result is not None, 90.0)
    return wiz


def _branched_stack():
    """Soma + tronco + dos brazos, para tener ramas hijas."""
    z, y, x = np.mgrid[0:20, 0:44, 0:44].astype(np.float32)
    d = np.sqrt((z - 10) ** 2 + (y - 20) ** 2 + (x - 12) ** 2)
    soma = 220.0 - 190.0 * (1.0 / (1.0 + np.exp((d - 5.0) / 1.0)))
    trunk = (np.abs(y - 20) <= 2) & (np.abs(z - 10) <= 2) & (x >= 10) & (x < 26)
    arm_x = (np.abs(y - 20) <= 2) & (np.abs(z - 10) <= 2) & (x >= 26) & (x < 40)
    arm_y = (np.abs(x - 26) <= 2) & (np.abs(z - 10) <= 2) & (y >= 20) & (y < 40)
    field = np.where(trunk | arm_x | arm_y, 30.0, soma)
    return Volume3D(field.astype(np.float32), name="ramas")


def test_wizard_tree_selection_and_soma(app):
    wiz = _reach_skeleton(app)

    assert wiz.tree_branches.topLevelItemCount() >= 1

    # Seleccionar un nodo con el visor 3D marca rama o soma.
    wiz._on_skeleton_node_clicked(0)
    assert wiz._selected_branch is not None or wiz._selected_soma

    # Seleccionar una rama en el árbol la marca y resalta el 3D sin fallar.
    branch_ids = [bid for bid in wiz._tree_items if bid != -1]
    assert branch_ids
    wiz.tree_branches.setCurrentItem(wiz._tree_items[branch_ids[0]])
    assert wiz._selected_branch == branch_ids[0]
    assert not wiz._selected_soma

    # Aplicar el tipo "Soma" a una rama designa el núcleo allí.
    wiz.combo_type.setCurrentText("Soma")
    wiz.tree_branches.setCurrentItem(wiz._tree_items[branch_ids[0]])
    wiz._on_apply_type()
    assert wiz.trace_result.has_soma
    assert -1 in wiz._tree_items  # entrada "Soma" en el árbol

    # Quitar soma lo elimina.
    wiz._on_soma_clear()
    assert not wiz.trace_result.has_soma

    wiz.deleteLater()
    app.processEvents()


def test_wizard_include_parents_delete_and_undo(app):
    wiz = _reach_skeleton(app, stack=_branched_stack(), line=(8, 20, 38, 20))

    branch_ids = [bid for bid in wiz._tree_items if bid != -1]
    assert len(branch_ids) >= 2, "se esperaba un esqueleto ramificado"

    # Buscar una rama con padres (no la raíz).
    deep = None
    for bid in branch_ids:
        chain = branch_ancestors(
            wiz.trace_result.branch_labels, wiz.trace_result.parents, bid
        )
        if len(chain) > 1:
            deep = bid
            break
    if deep is None:  # pragma: no cover - no debería ocurrir con el stack ramificado
        wiz.deleteLater()
        import pytest

        pytest.skip("el esqueleto sintético no generó ramas hijas")

    # Checkbox: la selección incluye la rama y todos sus padres.
    wiz.tree_branches.setCurrentItem(wiz._tree_items[deep])
    wiz.chk_include_parents.setChecked(True)
    expected = set(
        branch_ancestors(wiz.trace_result.branch_labels, wiz.trace_result.parents, deep)
    )
    assert wiz._highlight_branch_set() == expected
    assert len(expected) > 1
    # Las filas de los padres quedan seleccionadas.
    selected = {it.data(0, Qt.ItemDataRole.UserRole) for it in wiz.tree_branches.selectedItems()}
    assert expected.issubset(selected)

    # Supr borra solo la rama primaria; Ctrl+Z la restaura (1 paso).
    before_nodes = wiz.trace_result.n_nodes
    wiz._on_delete_branch()
    assert wiz.trace_result.n_nodes < before_nodes
    assert wiz._undo_snapshot is not None
    wiz._on_undo()
    assert wiz.trace_result.n_nodes == before_nodes
    assert wiz._undo_snapshot is None
    wiz._on_undo()  # segundo deshacer: no hace nada
    assert wiz.trace_result.n_nodes == before_nodes

    # Los atajos están conectados.
    assert wiz._shortcut_delete is not None
    assert wiz._shortcut_undo is not None

    wiz.deleteLater()
    app.processEvents()