"""Asistente de reconstrucción en 5 pasos revisables.

1. Abrir carpeta/archivos de la pila.
2. Calibración: umbral (isonivel), línea sobre una dendrita y su diámetro.
3. Segmentación con vista previa (clásica o por contraste local).
4. Esqueleto automático con vista previa y edición (podar, borrar/añadir ramas,
   tipar, refinar radios, deshacer).
5. Modelo 3D: superficie de la máscara o tubos del esqueleto, con vista previa.

Cada paso invalida los productos aguas abajo cuando cambian sus entradas, de
modo que la reconstrucción puede revisarse de arriba abajo antes de finalizar.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from muni.app.resources import load_icon
from muni.calibrate.isolevel import otsu_gray_threshold
from muni.calibrate.zcalib import measure_on_grayscale
from muni.core.meta import ModelMeta
from muni.core.volume import Volume3D
from muni.io.stack_loader import load_folder, load_stack, natural_key
from muni.postprocess.smooth import taubin_smooth
from muni.reconstruct.dual_contouring import DualContouring
from muni.reconstruct.marching_cubes import MarchingCubes
from muni.reconstruct.tube_mesh import tube_mesh_from_trace
from muni.segment.classical import ClassicalSegmenter
from muni.segment.focused import FocusedSegmenter
from muni.trace.base import SWC_AXON, SWC_DENDRITE, SWC_SOMA
from muni.trace.edit import (
    add_branch,
    clear_soma,
    delete_branch,
    prune_trace,
    refit_radii,
    set_branch_type,
    set_soma_from_seed,
)
from muni.trace.graph import branch_ancestors, branch_start, branch_tree, edge_length
from muni.trace.skimage_tracer import SkimageTracer
from muni.trace.soma import detect_soma
from muni.view.gl_viewport import GLViewport
from muni.view.slice_view import SliceView
from muni.view.worker import run_in_thread

RADIO_METHODS = {
    "dual_contouring": DualContouring,
    "marching_cubes": MarchingCubes,
}
TYPE_OPTIONS = {
    "Dendrita": SWC_DENDRITE,
    "Axón": SWC_AXON,
    "Soma": SWC_SOMA,
}
TYPE_LABELS = {value: name for name, value in TYPE_OPTIONS.items()}


class ReconstructionWizard(QDialog):
    """Asistente de 5 pasos. Expone ``mesh``, ``meta``, ``trace_result``."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Muni — Nueva reconstrucción")
        self.setWindowIcon(load_icon("Muni.ico"))
        self.resize(1100, 720)

        # Resultados finales (contrato con la ventana principal).
        self.mesh = None
        self.meta: ModelMeta | None = None
        self.trace_result = None
        self.trace_spacing: tuple[float, float, float] | None = None

        # Estado del pipeline.
        self.volume: Volume3D | None = None
        self.paths: list = []
        self.objective: int = 40
        self.threshold_gray: int = 128
        self.line: tuple[int, int, int, int] | None = None
        self.seed_plane: int = 0
        self.spacing: tuple[float, float, float] | None = None
        self.seg_result = None
        self._undo_snapshot = None
        self._add_start: tuple[int, int, int] | None = None
        self._selected_branch: int | None = None
        self._selected_soma: bool = False
        self._soma_seed_grid: tuple[int, int, int] | None = None
        self._pending_mesh = None
        self._pending_extractor: str | None = None
        self._threads: list = []
        self._busy = False

        # ------------------------------------------------------------- páginas
        self.stack = QStackedWidget()
        self.page1 = self._build_page_open()
        self.page2 = self._build_page_calibration()
        self.page3 = self._build_page_segmentation()
        self.page4 = self._build_page_skeleton()
        self.page5 = self._build_page_model()
        for p in (self.page1, self.page2, self.page3, self.page4, self.page5):
            self.stack.addWidget(p)

        # ------------------------------------------------------ progreso global
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.lbl_progress = QLabel("Listo.")
        self.lbl_progress.setStyleSheet("color: #999;")

        # ------------------------------------------------------------- navegación
        btn_row = QHBoxLayout()
        self.btn_back = QPushButton("← Atrás")
        self.btn_next = QPushButton("Siguiente →")
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_back.clicked.connect(self._go_back)
        self.btn_next.clicked.connect(self._go_next)
        btn_row.addWidget(self.btn_back)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_next)

        layout = QVBoxLayout(self)
        layout.addWidget(self.stack)
        layout.addWidget(self.progress)
        layout.addWidget(self.lbl_progress)
        layout.addLayout(btn_row)

        self._update_nav(0)

    # ============================================================ constructoras
    def _build_page_open(self) -> QWidget:
        page = QWidget()
        box = QGroupBox("Paso 1 — Abrir la pila de imágenes")
        self.list_files = QListWidget()
        self.btn_folder = QPushButton("Abrir carpeta…")
        self.btn_files = QPushButton("Abrir archivos…")
        self.btn_folder.clicked.connect(self._open_folder)
        self.btn_files.clicked.connect(self._open_files)
        self.lbl_stack_info = QLabel("Sin imágenes cargadas.")
        self.lbl_stack_info.setStyleSheet("color: #999;")

        hb = QHBoxLayout()
        hb.addWidget(self.btn_folder)
        hb.addWidget(self.btn_files)
        hb.addStretch(1)

        lay = QVBoxLayout(box)
        lay.addLayout(hb)
        lay.addWidget(self.list_files)
        lay.addWidget(self.lbl_stack_info)

        outer = QVBoxLayout(page)
        outer.addWidget(box)
        outer.addStretch(1)
        return page

    def _build_page_calibration(self) -> QWidget:
        page = QWidget()
        box = QGroupBox("Paso 2 — Calibración (dibuja una dendrita, ajusta el umbral)")

        self.slice_view = SliceView()
        self.slice_view.draw_mode = True
        self.slice_view.line_drawn.connect(self._on_line_drawn)

        self.list_planes = QListWidget()
        self.list_planes.currentRowChanged.connect(self._on_plane_changed)

        self.spin_objective = QSpinBox()
        self.spin_objective.setRange(1, 200)
        self.spin_objective.setValue(40)
        self.spin_objective.setSuffix("×")

        self.btn_auto_threshold = QPushButton("Estimar umbral automático (Otsu)")
        self.btn_auto_threshold.clicked.connect(self._auto_threshold)

        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setRange(0, 255)
        self.slider_threshold.setValue(128)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)

        self.lbl_threshold = QLabel("Umbral (gris): 128")

        self.chk_show_sharpness = QCheckBox("Previsualizar nitidez (contraste local)")
        self.chk_show_sharpness.setChecked(False)
        self.chk_show_sharpness.toggled.connect(self._on_sharpness_toggled)

        self.spin_diameter = QDoubleSpinBox()
        self.spin_diameter.setRange(0.1, 1000.0)
        self.spin_diameter.setValue(4.0)
        self.spin_diameter.setSuffix(" µm")
        self.spin_diameter.setDecimals(2)
        self.spin_diameter.valueChanged.connect(lambda _v: self._invalidate_calibration())

        self.lbl_calib_state = QLabel("Dibuja una línea cruzando una dendrita (botón izquierdo).")
        self.lbl_calib_state.setStyleSheet("color: #999;")
        self.lbl_calib_state.setWordWrap(True)

        right = QVBoxLayout()
        right.addWidget(QLabel("Objetivo del microscopio:"))
        right.addWidget(self.spin_objective)
        right.addWidget(QLabel("Planos:"))
        right.addWidget(self.list_planes)
        right.addWidget(self.chk_show_sharpness)
        right.addWidget(self.btn_auto_threshold)
        right.addWidget(self.lbl_threshold)
        right.addWidget(self.slider_threshold)
        right.addWidget(QLabel("Diámetro de la dendrita:"))
        right.addWidget(self.spin_diameter)
        right.addWidget(self.lbl_calib_state)
        right.addStretch(1)

        hb = QHBoxLayout(box)
        hb.addWidget(self.slice_view, 3)
        hb.addLayout(right, 1)

        outer = QVBoxLayout(page)
        outer.addWidget(box)
        outer.addStretch(1)
        return page

    def _build_page_segmentation(self) -> QWidget:
        page = QWidget()
        box = QGroupBox("Paso 3 — Segmentación (vista previa revisable)")

        self.seg_preview = SliceView()
        self.seg_preview.set_overlay_color(40, 255, 40)
        self.list_seg_planes = QListWidget()
        self.list_seg_planes.setMaximumHeight(140)
        self.list_seg_planes.currentRowChanged.connect(self._on_seg_plane_changed)

        self.radio_seg_classical = QRadioButton("Clásico (Otsu)")
        self.radio_seg_focused = QRadioButton("Enfoque (contraste local)")
        self.radio_seg_classical.setChecked(True)
        self._grp_segmentation = QButtonGroup(self)
        self._grp_segmentation.addButton(self.radio_seg_classical)
        self._grp_segmentation.addButton(self.radio_seg_focused)
        self.radio_seg_classical.toggled.connect(self._on_seg_param_changed)

        self.chk_denoise = QCheckBox("Difusión anisotrópica")
        self.chk_denoise.setChecked(True)
        self.spin_denoise_iters = QSpinBox()
        self.spin_denoise_iters.setRange(0, 50)
        self.spin_denoise_iters.setValue(5)
        self.spin_open = QSpinBox()
        self.spin_open.setRange(0, 5)
        self.spin_open.setValue(1)
        self.spin_close = QSpinBox()
        self.spin_close.setRange(0, 5)
        self.spin_close.setValue(1)
        self.spin_sharpness = QSpinBox()
        self.spin_sharpness.setRange(1, 15)
        self.spin_sharpness.setValue(3)
        self.chk_keep_largest = QCheckBox("Conservar solo el mayor componente")
        self.chk_keep_largest.setChecked(False)

        for widget in (self.chk_denoise, self.chk_keep_largest):
            widget.toggled.connect(self._on_seg_param_changed)
        for widget in (
            self.spin_denoise_iters,
            self.spin_open,
            self.spin_close,
            self.spin_sharpness,
        ):
            widget.valueChanged.connect(self._on_seg_param_changed)

        self.btn_seg_preview = QPushButton("Previsualizar segmentación")
        self.btn_seg_preview.clicked.connect(self._preview_segmentation)

        self.lbl_seg_info = QLabel("Sin segmentación.")
        self.lbl_seg_info.setStyleSheet("color: #999;")
        self.lbl_seg_info.setWordWrap(True)

        right = QVBoxLayout()
        right.addWidget(QLabel("Método:"))
        right.addWidget(self.radio_seg_classical)
        right.addWidget(self.radio_seg_focused)
        grid = QGridLayout()
        grid.addWidget(QLabel("Iteraciones de diffusion:"), 0, 0)
        grid.addWidget(self.spin_denoise_iters, 0, 1)
        grid.addWidget(QLabel("Apertura / cierre:"), 1, 0)
        row = QHBoxLayout()
        row.addWidget(self.spin_open)
        row.addWidget(self.spin_close)
        grid.addLayout(row, 1, 1)
        grid.addWidget(QLabel("Radio de nitidez:"), 2, 0)
        grid.addWidget(self.spin_sharpness, 2, 1)
        right.addLayout(grid)
        right.addWidget(self.chk_denoise)
        right.addWidget(self.chk_keep_largest)
        right.addWidget(self.btn_seg_preview)
        right.addWidget(QLabel("Planos:"))
        right.addWidget(self.list_seg_planes)
        right.addWidget(self.lbl_seg_info)
        right.addStretch(1)

        hb = QHBoxLayout(box)
        hb.addWidget(self.seg_preview, 3)
        hb.addLayout(right, 1)

        outer = QVBoxLayout(page)
        outer.addWidget(box)
        outer.addStretch(1)
        return page

    def _build_page_skeleton(self) -> QWidget:
        page = QWidget()
        box = QGroupBox("Paso 4 — Esqueleto (vista previa y edición)")

        self.gl_skeleton = GLViewport()
        self.gl_skeleton.show_container = False
        self.gl_skeleton.show_skeleton = True
        self.gl_skeleton.skeleton_node_clicked.connect(self._on_skeleton_node_clicked)

        self.btn_skeleton = QPushButton("Generar esqueleto")
        self.btn_skeleton.clicked.connect(self._generate_skeleton)

        self.lbl_skel_stats = QLabel("Sin esqueleto.")
        self.lbl_skel_stats.setStyleSheet("color: #999;")
        self.lbl_skel_stats.setWordWrap(True)

        self.spin_spur = QDoubleSpinBox()
        self.spin_spur.setRange(0.0, 100.0)
        self.spin_spur.setValue(2.0)
        self.spin_spur.setSuffix(" µm")
        self.btn_prune = QPushButton("Podar espolones")
        self.btn_prune.clicked.connect(self._on_prune)

        self.tree_branches = QTreeWidget()
        self.tree_branches.setColumnCount(4)
        self.tree_branches.setHeaderLabels(["Rama", "Longitud", "Radio", "Tipo"])
        self.tree_branches.setMaximumHeight(220)
        self.tree_branches.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.tree_branches.currentItemChanged.connect(self._on_branch_current_changed)

        self.chk_include_parents = QCheckBox("Seleccionar rama y sus padres (hacia el soma)")
        self.chk_include_parents.toggled.connect(self._on_include_parents_toggled)

        self.lbl_select_hint = QLabel(
            "Ctrl + clic en el visor 3D para seleccionar dendritas o el soma. "
            "Supr elimina la rama seleccionada; Ctrl+Z deshace."
        )
        self.lbl_select_hint.setStyleSheet("color: #7c4;")
        self.lbl_select_hint.setWordWrap(True)

        # Supr borra la rama cuando el árbol tiene el foco; Ctrl+Z deshace.
        self._shortcut_delete = QShortcut(QKeySequence.StandardKey.Delete, self.tree_branches)
        self._shortcut_delete.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._shortcut_delete.activated.connect(self._on_delete_branch)
        self._shortcut_undo = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._shortcut_undo.activated.connect(self._on_undo)

        self.btn_set_soma = QPushButton("Definir soma (núcleo)")
        self.btn_set_soma.setCheckable(True)
        self.btn_set_soma.toggled.connect(self._on_set_soma_toggled)
        self.spin_soma_scale = QDoubleSpinBox()
        self.spin_soma_scale.setRange(0.5, 2.0)
        self.spin_soma_scale.setSingleStep(0.1)
        self.spin_soma_scale.setValue(1.0)
        self.spin_soma_scale.setPrefix("×")
        self.spin_soma_scale.valueChanged.connect(self._on_soma_scale_changed)
        self.btn_soma_auto = QPushButton("Auto")
        self.btn_soma_auto.clicked.connect(self._on_soma_auto)
        self.btn_soma_clear = QPushButton("Quitar soma")
        self.btn_soma_clear.clicked.connect(self._on_soma_clear)

        self.combo_type = QComboBox()
        self.combo_type.addItems(list(TYPE_OPTIONS))
        self.btn_apply_type = QPushButton("Aplicar tipo")
        self.btn_apply_type.clicked.connect(self._on_apply_type)

        self.btn_delete_branch = QPushButton("Eliminar rama (+hijos)")
        self.btn_delete_branch.clicked.connect(self._on_delete_branch)

        self.btn_refit = QPushButton("Refinar radios")
        self.btn_refit.clicked.connect(self._on_refit_radii)

        self.btn_add = QPushButton("Añadir rama")
        self.btn_add.setCheckable(True)
        self.btn_add.toggled.connect(self._on_add_toggled)

        self.btn_undo = QPushButton("Deshacer")
        self.btn_undo.clicked.connect(self._on_undo)

        self.lbl_edit_hint = QLabel("")
        self.lbl_edit_hint.setStyleSheet("color: #999;")
        self.lbl_edit_hint.setWordWrap(True)

        self.slice_skel = SliceView()
        self.slice_skel.mark_mode = False
        self.slice_skel.point_clicked.connect(self._on_skel_click)
        self.list_skel_planes = QListWidget()
        self.list_skel_planes.setMaximumHeight(120)
        self.list_skel_planes.currentRowChanged.connect(self._on_skel_plane_changed)

        right = QVBoxLayout()
        right.addWidget(self.btn_skeleton)
        right.addWidget(self.lbl_skel_stats)
        row = QHBoxLayout()
        row.addWidget(self.spin_spur)
        row.addWidget(self.btn_prune)
        right.addLayout(row)
        right.addWidget(QLabel("Ramas:"))
        right.addWidget(self.tree_branches)
        right.addWidget(self.chk_include_parents)
        soma_box = QGroupBox("Soma (núcleo)")
        soma_row = QHBoxLayout(soma_box)
        soma_row.addWidget(self.btn_set_soma)
        soma_row.addWidget(self.spin_soma_scale)
        soma_row.addWidget(self.btn_soma_auto)
        soma_row.addWidget(self.btn_soma_clear)
        right.addWidget(soma_box)
        right.addWidget(self.lbl_select_hint)
        row2 = QHBoxLayout()
        row2.addWidget(self.combo_type)
        row2.addWidget(self.btn_apply_type)
        right.addLayout(row2)
        right.addWidget(self.btn_delete_branch)
        row3 = QHBoxLayout()
        row3.addWidget(self.btn_refit)
        row3.addWidget(self.btn_add)
        row3.addWidget(self.btn_undo)
        right.addLayout(row3)
        right.addWidget(self.lbl_edit_hint)
        right.addWidget(QLabel("Planos (para añadir ramas):"))
        right.addWidget(self.list_skel_planes)
        right.addWidget(self.slice_skel)
        right.addStretch(1)

        hb = QHBoxLayout(box)
        hb.addWidget(self.gl_skeleton, 3)
        hb.addLayout(right, 2)

        outer = QVBoxLayout(page)
        outer.addWidget(box)
        outer.addStretch(1)
        return page

    def _build_page_model(self) -> QWidget:
        page = QWidget()
        box = QGroupBox("Paso 5 — Modelo 3D")

        self.gl_model = GLViewport()
        self.radio_mask_surface = QRadioButton("Superficie de la máscara (isosuperficie)")
        self.radio_tubes = QRadioButton("Tubos del esqueleto (fitting)")
        self.radio_mask_surface.setChecked(True)
        self._grp_source = QButtonGroup(self)
        self._grp_source.addButton(self.radio_mask_surface)
        self._grp_source.addButton(self.radio_tubes)
        self.radio_mask_surface.toggled.connect(self._on_model_param_changed)

        self.radio_dc = QRadioButton("Dual Contouring")
        self.radio_mc = QRadioButton("Marching Cubes")
        self.radio_dc.setChecked(True)
        self._grp_extraction = QButtonGroup(self)
        self._grp_extraction.addButton(self.radio_dc)
        self._grp_extraction.addButton(self.radio_mc)
        self.radio_dc.toggled.connect(self._on_model_param_changed)

        self.spin_iso = QDoubleSpinBox()
        self.spin_iso.setRange(0.01, 0.99)
        self.spin_iso.setValue(0.5)
        self.spin_iso.setSingleStep(0.05)
        self.spin_iso.valueChanged.connect(self._on_model_param_changed)

        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(0, 20)
        self.spin_smooth.setValue(3)
        self.spin_smooth.valueChanged.connect(self._on_model_param_changed)

        self.spin_tube_sides = QSpinBox()
        self.spin_tube_sides.setRange(4, 32)
        self.spin_tube_sides.setValue(8)
        self.spin_tube_sides.valueChanged.connect(self._on_model_param_changed)

        self.btn_preview_model = QPushButton("Previsualizar modelo")
        self.btn_preview_model.clicked.connect(self._preview_model)

        self.lbl_model_info = QLabel("Sin modelo.")
        self.lbl_model_info.setStyleSheet("color: #999;")
        self.lbl_model_info.setWordWrap(True)

        self.btn_finish = QPushButton("Finalizar")
        self.btn_finish.clicked.connect(self._finish)
        self.btn_finish.setEnabled(False)

        right = QVBoxLayout()
        right.addWidget(QLabel("Origen del modelo:"))
        right.addWidget(self.radio_mask_surface)
        right.addWidget(self.radio_tubes)
        grid = QGridLayout()
        grid.addWidget(QLabel("Extractor:"), 0, 0)
        grid.addWidget(self.radio_dc, 0, 1)
        grid.addWidget(self.radio_mc, 1, 1)
        grid.addWidget(QLabel("Isonivel 3D:"), 2, 0)
        grid.addWidget(self.spin_iso, 2, 1)
        grid.addWidget(QLabel("Suavizado (Taubin):"), 3, 0)
        grid.addWidget(self.spin_smooth, 3, 1)
        grid.addWidget(QLabel("Lados del tubo:"), 4, 0)
        grid.addWidget(self.spin_tube_sides, 4, 1)
        right.addLayout(grid)
        right.addWidget(self.btn_preview_model)
        right.addWidget(self.lbl_model_info)
        right.addStretch(1)
        right.addWidget(self.btn_finish)

        hb = QHBoxLayout(box)
        hb.addWidget(self.gl_model, 3)
        hb.addLayout(right, 2)

        outer = QVBoxLayout(page)
        outer.addWidget(box)
        outer.addStretch(1)
        return page

    # ============================================================== navegación
    def _update_nav(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.btn_back.setEnabled(index > 0 and not self._busy)
        self.btn_next.setVisible(index < 4)
        self.btn_next.setEnabled(not self._busy and self._can_advance(index))
        if index == 1:
            self.slice_view.fit_to_window()
        if index == 3:
            self._refresh_skeleton_slice()
        if index == 4:
            self._sync_model_controls()

    def _can_advance(self, index: int) -> bool:
        if index == 0:
            return self.volume is not None
        if index == 1:
            return self.volume is not None and self.line is not None
        if index == 2:
            return self.seg_result is not None
        if index == 3:
            return self.trace_result is not None
        return False

    def _go_back(self) -> None:
        if self._busy:
            return
        index = self.stack.currentIndex()
        if index > 0:
            self._update_nav(index - 1)

    def _go_next(self) -> None:
        if self._busy:
            return
        index = self.stack.currentIndex()
        if not self._can_advance(index):
            if index == 0:
                QMessageBox.warning(self, "Muni", "Primero abre una pila de imágenes.")
            elif index == 1:
                QMessageBox.warning(self, "Muni", "Dibuja una línea cruzando una dendrita.")
            elif index == 2:
                QMessageBox.warning(self, "Muni", "Previsualiza la segmentación antes de seguir.")
            elif index == 3:
                QMessageBox.warning(self, "Muni", "Genera el esqueleto antes de seguir.")
            return
        if index == 1:
            self.objective = self.spin_objective.value()
            self._apply_calibration()
        self._update_nav(index + 1)
        if index + 1 == 3 and self.trace_result is None:
            self._generate_skeleton()
        if index + 1 == 4 and self._pending_mesh is None:
            self._preview_model()

    # ============================================================ invalidación
    def _invalidate_calibration(self) -> None:
        self.spacing = None
        self._invalidate_skeleton()

    def _invalidate_segmentation(self) -> None:
        self.seg_result = None
        self.seg_preview.clear()
        self._invalidate_skeleton()
        self.lbl_seg_info.setText("Sin segmentación (parámetros cambiados).")

    def _invalidate_skeleton(self) -> None:
        self.trace_result = None
        self._undo_snapshot = None
        self._add_start = None
        self._clear_selection()
        self._soma_seed_grid = None
        self._invalidate_mesh()
        if hasattr(self, "gl_skeleton"):
            self.gl_skeleton.clear_skeleton()
            self.tree_branches.clear()
            self.lbl_skel_stats.setText("Sin esqueleto.")

    def _invalidate_mesh(self) -> None:
        self._pending_mesh = None
        self._pending_extractor = None
        if hasattr(self, "gl_model"):
            self.gl_model.clear_mesh()
        if hasattr(self, "btn_finish"):
            self.btn_finish.setEnabled(False)
        if hasattr(self, "lbl_model_info"):
            self.lbl_model_info.setText("Sin modelo (parámetros cambiados).")

    # ================================================================== paso 1
    def _open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Carpeta con la pila de imágenes")
        if not folder:
            return
        try:
            volume, paths = load_folder(folder)
        except ValueError as exc:
            QMessageBox.critical(self, "Muni", str(exc))
            return
        self._set_stack(volume, paths)

    def _open_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Imágenes de la pila", "",
            "Imágenes (*.bmp *.jpg *.jpeg *.tif *.tiff *.png)",
        )
        if len(files) < 4:
            QMessageBox.information(self, "Muni", "Selecciona al menos 4 imágenes.")
            return
        files = sorted(files, key=natural_key)
        try:
            volume, paths = load_stack(files)
        except ValueError as exc:
            QMessageBox.critical(self, "Muni", str(exc))
            return
        self._set_stack(volume, paths)

    def _set_stack(self, volume: Volume3D, paths: list) -> None:
        self.volume = volume
        self.paths = paths
        self.list_files.clear()
        for p in paths:
            self.list_files.addItem(p.name)
        self.lbl_stack_info.setText(
            f"{volume.dimz} planos · {volume.dimx}×{volume.dimy} px · "
            f"{volume.data.nbytes / 1e6:.1f} MB"
        )
        self._fill_planes()
        self._fill_seg_planes()
        self._fill_skel_planes()
        self.list_planes.setCurrentRow(volume.dimz // 2)
        self._invalidate_calibration()
        self._invalidate_segmentation()
        self._update_nav(0)

    # ================================================================== paso 2
    def _fill_planes(self) -> None:
        self.list_planes.clear()
        if self.volume is None:
            return
        for i in range(self.volume.dimz):
            self.list_planes.addItem(self.paths[i].name if i < len(self.paths) else f"Plano {i}")

    def _on_plane_changed(self, row: int) -> None:
        if row < 0 or self.volume is None:
            return
        self.seed_plane = row
        plane = self.volume.plane(row)
        self.slice_view.set_plane(plane, self._threshold_overlay())
        self.slice_view.set_line(self.line)
        self.slice_view.fit_to_window()

    def _on_line_drawn(self, line: tuple) -> None:
        self.line = line
        self.lbl_calib_state.setText(
            f"Línea: ({line[0]}, {line[1]}) → ({line[2]}, {line[3]}) en el plano {self.seed_plane}. "
            "Ajusta el umbral si lo necesitas."
        )
        self.lbl_calib_state.setStyleSheet("color: #7c4;")
        self._invalidate_calibration()
        self._update_nav(self.stack.currentIndex())

    def _auto_threshold(self) -> None:
        if self.volume is None:
            return
        self.threshold_gray = otsu_gray_threshold(self.volume, denoise=True)
        self.slider_threshold.blockSignals(True)
        self.slider_threshold.setValue(self.threshold_gray)
        self.slider_threshold.blockSignals(False)
        self.lbl_threshold.setText(f"Umbral (gris): {self.threshold_gray}")
        self.slice_view.set_overlay(self._threshold_overlay())
        self._invalidate_calibration()

    def _on_threshold_changed(self, value: int) -> None:
        self.threshold_gray = value
        self.lbl_threshold.setText(f"Umbral (gris): {value}")
        self.slice_view.set_overlay(self._threshold_overlay())
        self._invalidate_calibration()

    def _threshold_overlay(self) -> np.ndarray | None:
        if self.volume is None:
            return None
        if self.chk_show_sharpness.isChecked():
            return self._sharpness_overlay()
        return self.volume.plane(self.seed_plane) <= self.threshold_gray

    def _on_sharpness_toggled(self, checked: bool) -> None:
        if self.volume is None:
            return
        if checked:
            self.slice_view.set_overlay_color(40, 255, 40)
        else:
            self.slice_view.set_overlay_color(255, 40, 40)
        self.slice_view.set_overlay(self._threshold_overlay())

    def _sharpness_overlay(self) -> np.ndarray | None:
        if self.volume is None:
            return None
        from muni.segment.threshold import _box_mean

        plane = self.volume.plane(self.seed_plane).astype(np.float64)
        mean = _box_mean(plane, 3)
        mean_sq = _box_mean(plane * plane, 3)
        std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))
        s_max = float(std.max())
        if s_max < 1e-10:
            return None
        return std > s_max * 0.3

    def _apply_calibration(self) -> None:
        if self.volume is None or self.line is None:
            return
        diameter = float(self.spin_diameter.value())
        cal = measure_on_grayscale(self.volume, self.threshold_gray, self.seed_plane, self.line)
        if cal.z_extent_planes > 0 and cal.max_diameter_px > 0:
            self.volume.spacing_xy_um = cal.spacing_xy_um(diameter)
            self.volume.spacing_z_um = cal.spacing_z_um(diameter)
            self.spacing = (
                self.volume.spacing_z_um,
                self.volume.spacing_xy_um,
                self.volume.spacing_xy_um,
            )
        else:
            QMessageBox.warning(
                self,
                "Muni",
                "No se detectó la dendrita en Z; se usará espaciado 1.0 (sin escala física).",
            )
            self.spacing = (1.0, 1.0, 1.0)

    # ================================================================== paso 3
    def _on_seg_param_changed(self, *_args) -> None:
        self._invalidate_segmentation()
        self._update_nav(self.stack.currentIndex())

    def _fill_seg_planes(self) -> None:
        self.list_seg_planes.clear()
        if self.volume is None:
            return
        for i in range(self.volume.dimz):
            self.list_seg_planes.addItem(
                self.paths[i].name if i < len(self.paths) else f"Plano {i}"
            )
        self.list_seg_planes.setCurrentRow(self.volume.dimz // 2)

    def _on_seg_plane_changed(self, row: int) -> None:
        if row < 0 or self.volume is None:
            return
        plane = self.volume.plane(row)
        overlay = self.seg_result.mask[row] if self.seg_result is not None else None
        self.seg_preview.set_plane(plane, overlay)
        self.seg_preview.fit_to_window()

    def _seg_kwargs(self) -> dict:
        return {
            "denoise": self.chk_denoise.isChecked(),
            "denoise_iterations": self.spin_denoise_iters.value(),
            "open_iterations": self.spin_open.value(),
            "close_iterations": self.spin_close.value(),
            "keep_largest": self.chk_keep_largest.isChecked(),
        }

    def _preview_segmentation(self) -> None:
        if self.volume is None:
            return
        volume = self.volume
        focused = self.radio_seg_focused.isChecked()
        kwargs = self._seg_kwargs()

        def task(progress):
            progress(10, "Segmentando…")
            if focused:
                kwargs["sharpness_radius"] = self.spin_sharpness.value()
                seg = FocusedSegmenter(**kwargs).segment(volume)
            else:
                seg = ClassicalSegmenter(**kwargs).segment(volume)
            progress(100, "Segmentación lista.")
            return seg

        self._invalidate_segmentation()
        self._spawn(task, self._on_segmentation_done)

    def _on_segmentation_done(self, seg) -> None:
        if seg is None:
            return
        self.seg_result = seg
        self._fill_seg_planes()
        self._on_seg_plane_changed(self.list_seg_planes.currentRow())
        self.lbl_seg_info.setText(
            f"Método: {seg.method} · vóxeles: {int(seg.mask.sum()):,}"
        )
        self._invalidate_skeleton()
        self._update_nav(self.stack.currentIndex())

    # ================================================================== paso 4
    def _fill_skel_planes(self) -> None:
        self.list_skel_planes.clear()
        if self.volume is None:
            return
        for i in range(self.volume.dimz):
            self.list_skel_planes.addItem(
                self.paths[i].name if i < len(self.paths) else f"Plano {i}"
            )
        self.list_skel_planes.setCurrentRow(self.volume.dimz // 2)

    def _generate_skeleton(self) -> None:
        if self.seg_result is None or self.spacing is None:
            return
        mask = self.seg_result.mask
        spacing = self.spacing
        spur = float(self.spin_spur.value())

        def task(progress):
            progress(20, "Esqueletizando…")
            result = SkimageTracer().trace(mask, spacing=spacing, min_spur_um=spur)
            progress(100, "Esqueleto listo.")
            return result

        self._spawn(task, self._on_skeleton_done)

    def _on_skeleton_done(self, result) -> None:
        if result is None:
            return
        self.trace_result = result
        self._undo_snapshot = None
        self._clear_selection()
        self._soma_seed_grid = None
        self._invalidate_mesh()
        self._refresh_skeleton()
        self._update_nav(self.stack.currentIndex())

    def _highlight_branch_set(self) -> set[int]:
        """Ramas a resaltar: la seleccionada, o ella y sus padres si el check lo pide."""
        result = self.trace_result
        if result is None or self._selected_soma or self._selected_branch is None:
            return set()
        if not self.chk_include_parents.isChecked():
            return {int(self._selected_branch)}
        return set(
            branch_ancestors(result.branch_labels, result.parents, int(self._selected_branch))
        )

    def _apply_highlight(self) -> None:
        """Repinta el esqueleto respetando la selección actual."""
        result = self.trace_result
        if result is None:
            return
        self.gl_skeleton.set_skeleton(
            result,
            highlight_branches=self._highlight_branch_set(),
            highlight_soma=self._selected_soma,
        )
        self.gl_skeleton.show_skeleton = True

    def _clear_selection(self) -> None:
        self._selected_branch = None
        self._selected_soma = False

    def _on_include_parents_toggled(self, _checked: bool) -> None:
        self._select_in_tree()
        self._apply_highlight()

    def _refresh_skeleton(self) -> None:
        result = self.trace_result
        if result is None:
            return
        soma_txt = ""
        if result.has_soma:
            soma_txt = f" · soma r={float(np.mean(result.soma_radii_um)):.2f} µm"
        self.lbl_skel_stats.setText(
            f"Longitud: {result.total_length_um:.1f} µm · "
            f"ramas: {result.n_branches} · nodos: {result.n_nodes}{soma_txt}"
        )
        self._fill_branches()
        self._refresh_skeleton_slice()
        self._apply_highlight()

    def _branch_stats(self, bid: int) -> tuple[float, float, int]:
        """``(longitud_um, radio_medio_um, tipo_swc_dominante)`` de una rama."""
        result = self.trace_result
        idx = np.flatnonzero(result.branch_labels == bid)
        length = 0.0
        for node in idx:
            p = int(result.parents[node])
            if p >= 0 and result.branch_labels[p] == bid:
                length += edge_length(result.coords[node], result.coords[p], result.spacing)
        radius = float(np.median(result.radii[idx]))
        swc_type = int(np.bincount(result.types[idx]).argmax())
        return length, radius, swc_type

    def _fill_branches(self) -> None:
        self.tree_branches.clear()
        self._tree_items: dict[int, QTreeWidgetItem] = {}
        result = self.trace_result
        if result is None:
            return

        if result.has_soma:
            radius = float(np.mean(result.soma_radii_um))
            soma_item = QTreeWidgetItem(["Soma", f"r={radius:.2f} µm", "", "Soma"])
            soma_item.setData(0, Qt.ItemDataRole.UserRole, -1)
            self.tree_branches.addTopLevelItem(soma_item)
            self._tree_items[-1] = soma_item

        for bid, parent_bid, _depth in branch_tree(result.branch_labels, result.parents):
            length, radius, swc_type = self._branch_stats(bid)
            item = QTreeWidgetItem(
                [
                    f"Rama {bid}",
                    f"{length:.1f} µm",
                    f"r={radius:.2f} µm",
                    TYPE_LABELS.get(swc_type, "?"),
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, int(bid))
            self._tree_items[bid] = item
            if parent_bid is None or parent_bid not in self._tree_items:
                self.tree_branches.addTopLevelItem(item)
            else:
                self._tree_items[parent_bid].addChild(item)
        self.tree_branches.expandAll()
        self._select_in_tree()

    def _select_in_tree(self) -> None:
        """Sincroniza la selección del árbol con la rama/soma seleccionados.

        Con ``chk_include_parents`` marcado también marca las filas de los padres.
        """
        target = -1 if self._selected_soma else self._selected_branch
        if target is None:
            return
        item = getattr(self, "_tree_items", {}).get(target)
        if item is None:
            return
        tree = self.tree_branches
        tree.blockSignals(True)
        try:
            tree.clearSelection()
            tree.setCurrentItem(item)
            item.setSelected(True)
            for bid in self._highlight_branch_set():
                ancestor_item = self._tree_items.get(bid)
                if ancestor_item is not None:
                    ancestor_item.setSelected(True)
        finally:
            tree.blockSignals(False)

    def _on_branch_current_changed(self, current: QTreeWidgetItem | None, _previous) -> None:
        if current is None:
            return
        bid = current.data(0, Qt.ItemDataRole.UserRole)
        if bid is None:
            return
        if int(bid) == -1:
            self._selected_soma = True
            self._selected_branch = None
        else:
            self._selected_soma = False
            self._selected_branch = int(bid)
        self._select_in_tree()
        self._apply_highlight()

    def _on_skeleton_node_clicked(self, node: int) -> None:
        result = self.trace_result
        if result is None or node < 0 or node >= result.n_nodes:
            return
        if self.btn_set_soma.isChecked():
            self.btn_set_soma.setChecked(False)
            self._apply_soma(tuple(int(v) for v in result.coords[node]))
            return
        if int(result.types[node]) == SWC_SOMA:
            self._selected_soma = True
            self._selected_branch = None
        else:
            self._selected_soma = False
            self._selected_branch = int(result.branch_labels[node])
        self._select_in_tree()
        self._apply_highlight()

    def _refresh_skeleton_slice(self) -> None:
        result = self.trace_result
        if result is None or self.volume is None:
            self.slice_skel.clear()
            return
        plane = self.list_skel_planes.currentRow()
        if plane < 0:
            plane = self.volume.dimz // 2
        sel = np.isclose(result.coords[:, 0], plane)
        nodes = result.coords[sel]
        markers = (
            np.column_stack([nodes[:, 2], nodes[:, 1], result.types[sel]])
            if len(nodes)
            else None
        )
        self.slice_skel.set_plane(self.volume.plane(plane), None)
        self.slice_skel.set_markers(markers)
        self.slice_skel.fit_to_window()

    def _on_skel_plane_changed(self, row: int) -> None:
        self._refresh_skeleton_slice()

    def _current_branch_id(self) -> int | None:
        item = self.tree_branches.currentItem()
        if item is None:
            QMessageBox.information(self, "Muni", "Selecciona una rama en la lista.")
            return None
        bid = item.data(0, Qt.ItemDataRole.UserRole)
        if bid is None or int(bid) == -1:
            QMessageBox.information(
                self, "Muni", "Selecciona una rama (el soma no es una rama)."
            )
            return None
        return int(bid)

    def _push_undo(self) -> None:
        """Guarda un único paso de deshacer (el estado justo antes de la edición)."""
        if self.trace_result is not None:
            self._undo_snapshot = self.trace_result

    def _on_undo(self) -> None:
        if self._undo_snapshot is None:
            return
        self.trace_result = self._undo_snapshot
        self._undo_snapshot = None
        self._clear_selection()
        self._invalidate_mesh()
        self._refresh_skeleton()
        self._update_nav(self.stack.currentIndex())

    def _after_edit(self) -> None:
        self._invalidate_mesh()
        self._refresh_skeleton()
        self._update_nav(self.stack.currentIndex())

    def _on_prune(self) -> None:
        if self.trace_result is None:
            return
        self._push_undo()
        self.trace_result = prune_trace(self.trace_result, float(self.spin_spur.value()))
        self._clear_selection()
        self._after_edit()

    def _on_apply_type(self) -> None:
        bid = self._current_branch_id()
        if bid is None or self.trace_result is None:
            return
        swc_type = TYPE_OPTIONS[self.combo_type.currentText()]
        if swc_type == SWC_SOMA:
            self._set_soma_on_branch(bid)
            return
        self._push_undo()
        self.trace_result = set_branch_type(self.trace_result, bid, swc_type)
        self._clear_selection()
        self._after_edit()

    def _on_delete_branch(self) -> None:
        bid = self._current_branch_id()
        if bid is None or self.trace_result is None:
            return
        idx = np.flatnonzero(self.trace_result.branch_labels == bid)
        start = branch_start(idx, self.trace_result.parents)
        if self.trace_result.parents[start] < 0:
            answer = QMessageBox.question(
                self,
                "Muni",
                "Esa rama contiene la raíz (soma). Se borrará casi todo el esqueleto. ¿Continuar?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._push_undo()
        try:
            self.trace_result = delete_branch(self.trace_result, bid)
        except ValueError as exc:
            self._undo_snapshot = None
            QMessageBox.warning(self, "Muni", str(exc))
            return
        self._clear_selection()
        self._after_edit()

    def _on_refit_radii(self) -> None:
        if self.trace_result is None or self.volume is None:
            return
        self._push_undo()
        try:
            self.trace_result = refit_radii(
                self.trace_result,
                self.volume.data,
                self.trace_result.spacing,
                float(self.threshold_gray),
            )
        except ValueError as exc:
            self._undo_snapshot = None
            QMessageBox.warning(self, "Muni", str(exc))
            return
        self._clear_selection()
        self._after_edit()

    def _on_add_toggled(self, checked: bool) -> None:
        self._add_start = None
        self.slice_skel.mark_mode = checked
        if checked:
            self.lbl_edit_hint.setText(
                "Clic en el punto de inicio de la rama (sobre el esqueleto)."
            )
        else:
            self.lbl_edit_hint.setText("")

    def _on_skel_click(self, x: int, y: int) -> None:
        if not self.btn_add.isChecked() or self.trace_result is None:
            return
        plane = self.list_skel_planes.currentRow()
        if plane < 0:
            return
        if self._add_start is None:
            self._add_start = (plane, int(y), int(x))
            self.lbl_edit_hint.setText("Clic en el punto final de la rama.")
            return
        start = self._add_start
        end = (plane, int(y), int(x))
        self._add_start = None
        self.btn_add.setChecked(False)
        self._add_branch_async(start, end)

    def _add_branch_async(self, start: tuple, end: tuple) -> None:
        if self.seg_result is None or self.trace_result is None:
            return
        before = self.trace_result
        mask = self.seg_result.mask
        spacing = self.trace_result.spacing

        def task(progress):
            progress(50, "Añadiendo rama…")
            return add_branch(before, mask, spacing, start, end)

        self._push_undo()
        self._spawn(task, self._on_add_done)

    def _on_add_done(self, result) -> None:
        if result is None:
            return
        self.trace_result = result
        self._clear_selection()
        self._after_edit()

    # ============================================================== soma (paso 4)
    def _on_set_soma_toggled(self, checked: bool) -> None:
        if checked:
            self.lbl_select_hint.setText(
                "Ctrl + clic sobre el núcleo en el visor 3D para fijar el soma."
            )
        else:
            self.lbl_select_hint.setText(
                "Ctrl + clic en el visor 3D para seleccionar dendritas o el soma."
            )

    def _seed_from_current_soma(self) -> tuple[int, int, int] | None:
        result = self.trace_result
        if result is None or not result.has_soma:
            return None
        center = result.soma_center_um  # mundo (x, y, z)
        spacing = result.spacing
        return (
            round(center[2] / spacing[0]),
            round(center[1] / spacing[1]),
            round(center[0] / spacing[2]),
        )

    def _apply_soma(
        self, seed_grid: tuple[int, int, int], *, push_undo: bool = True
    ) -> None:
        if self.trace_result is None or self.seg_result is None:
            return
        if push_undo:
            self._push_undo()
        try:
            self.trace_result = set_soma_from_seed(
                self.trace_result,
                self.seg_result.mask,
                self.trace_result.spacing,
                seed_grid,
                radius_scale=float(self.spin_soma_scale.value()),
            )
        except ValueError as exc:
            if push_undo:
                self._undo_snapshot = None
            QMessageBox.warning(self, "Muni", str(exc))
            return
        self._soma_seed_grid = tuple(int(v) for v in seed_grid)
        self._selected_soma = True
        self._selected_branch = None
        self._after_edit()

    def _set_soma_on_branch(self, bid: int) -> None:
        result = self.trace_result
        idx = np.flatnonzero(result.branch_labels == bid)
        if idx.size == 0:
            return
        node = int(idx[int(np.argmax(result.radii[idx]))])
        self._apply_soma(tuple(int(v) for v in result.coords[node]))

    def _on_soma_scale_changed(self, _value: float) -> None:
        if self.trace_result is None:
            return
        seed = self._soma_seed_grid
        if seed is None:
            seed = self._seed_from_current_soma()
        if seed is None:
            return
        self._apply_soma(seed, push_undo=False)

    def _on_soma_auto(self) -> None:
        if self.trace_result is None or self.seg_result is None:
            return
        detected = detect_soma(self.seg_result.mask, self.trace_result.spacing)
        if detected is None:
            QMessageBox.warning(self, "Muni", "No se detectó ningún soma en la máscara.")
            return
        spacing = self.trace_result.spacing
        center = detected.center_um  # mundo (x, y, z)
        seed = (
            round(center[2] / spacing[0]),
            round(center[1] / spacing[1]),
            round(center[0] / spacing[2]),
        )
        self._apply_soma(seed)

    def _on_soma_clear(self) -> None:
        if self.trace_result is None or not self.trace_result.has_soma:
            return
        self._push_undo()
        self.trace_result = clear_soma(self.trace_result)
        self._soma_seed_grid = None
        self._clear_selection()
        self._after_edit()

    # ================================================================== paso 5
    def _on_model_param_changed(self, *_args) -> None:
        self._invalidate_mesh()
        self._sync_model_controls()
        self._update_nav(self.stack.currentIndex())

    def _sync_model_controls(self) -> None:
        tubes = self.radio_tubes.isChecked()
        self.radio_dc.setEnabled(not tubes)
        self.radio_mc.setEnabled(not tubes)
        self.spin_iso.setEnabled(not tubes)
        self.spin_smooth.setEnabled(not tubes)
        self.spin_tube_sides.setEnabled(tubes)

    def _preview_model(self) -> None:
        if self.radio_tubes.isChecked():
            if self.trace_result is None:
                return
            sides = self.spin_tube_sides.value()
            result = self.trace_result

            def task(progress):
                progress(40, "Generando tubos…")
                return tube_mesh_from_trace(result, sides=sides), "skeleton_tubes"

        else:
            if self.seg_result is None or self.spacing is None:
                return
            probability = self.seg_result.probability
            spacing = self.spacing
            isolevel = float(self.spin_iso.value())
            method = "dual_contouring" if self.radio_dc.isChecked() else "marching_cubes"
            smooth_iters = self.spin_smooth.value()

            def task(progress):
                progress(30, "Extrayendo superficie…")
                extracted = RADIO_METHODS[method]().extract(
                    probability, isovalue=isolevel, spacing=spacing
                )
                progress(70, "Suavizando malla…")
                mesh = taubin_smooth(extracted.mesh, iterations=smooth_iters)
                return mesh, extracted.method

        self._invalidate_mesh()
        self._spawn(task, self._on_model_done)

    def _on_model_done(self, result) -> None:
        if result is None:
            return
        mesh, extractor = result
        if mesh is None or mesh.is_empty:
            QMessageBox.warning(
                self, "Muni", "La malla quedó vacía: revisa el origen y los parámetros."
            )
            return
        self._pending_mesh = mesh
        self._pending_extractor = extractor
        self.gl_model.set_mesh(mesh)
        self.lbl_model_info.setText(
            f"Extractor: {extractor} · triángulos: {mesh.face_count:,}"
        )
        self.btn_finish.setEnabled(True)
        self._update_nav(self.stack.currentIndex())

    def _build_meta(self, extractor: str) -> ModelMeta:
        volume = self.volume
        meta = ModelMeta(
            dimx=volume.dimx,
            dimy=volume.dimy,
            dimz=volume.dimz,
            spacing_xy_um=volume.spacing_xy_um,
            spacing_z_um=volume.spacing_z_um,
            isolevel=float(self.spin_iso.value()) if extractor != "skeleton_tubes" else 0.0,
            objective=self.objective,
            dendrite_diameter_um=float(self.spin_diameter.value()),
            segmenter=self.seg_result.method if self.seg_result is not None else "",
            extractor=extractor,
            source_stack=", ".join(p.name for p in self.paths[:3]),
        )
        if self.trace_result is not None:
            analysis = {
                "total_length_um": float(self.trace_result.total_length_um),
                "n_branches": int(self.trace_result.n_branches),
                "n_nodes": int(self.trace_result.n_nodes),
            }
            if self.trace_result.has_soma:
                analysis["soma_radius_um"] = float(np.mean(self.trace_result.soma_radii_um))
            meta.analysis = analysis
        return meta

    def _finish(self) -> None:
        if self._pending_mesh is None:
            return
        self.mesh = self._pending_mesh
        self.meta = self._build_meta(self._pending_extractor or "skeleton_tubes")
        self.trace_spacing = self.spacing
        self.accept()

    # ================================================================ threading
    def _spawn(self, task, on_finished) -> None:
        self._busy = True
        self._update_nav(self.stack.currentIndex())
        thread = run_in_thread(
            task,
            on_finished=lambda result: self._finish_task(on_finished, result),
            on_error=self._on_error,
            on_progress=self._on_progress,
        )
        self._threads.append(thread)

    def _finish_task(self, callback, result) -> None:
        self._busy = False
        callback(result)
        self._update_nav(self.stack.currentIndex())

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress.setValue(percent)
        self.lbl_progress.setText(message)

    def _on_error(self, message: str) -> None:
        self._busy = False
        print(f"[muni-error]{os.linesep}{message}", file=sys.stderr, flush=True)
        QMessageBox.critical(self, "Muni — Error", message)
        self._update_nav(self.stack.currentIndex())