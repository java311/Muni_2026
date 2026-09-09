"""Asistente de reconstrucción en 4 pasos (equivalente al Image_Form antiguo).

1. Abrir carpeta/archivos de la pila.
2. Parámetros del microscopio.
3. Calibración: dibujar una dendrita y ajustar el umbral (isonivel) con preview.
4. Procesado: reconstrucción en segundo plano, con progreso y cancelación.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
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
from muni.segment.classical import ClassicalSegmenter
from muni.segment.focused import FocusedSegmenter
from muni.trace.skimage_tracer import SkimageTracer
from muni.view.slice_view import SliceView
from muni.view.worker import run_in_thread

RADIO_METHODS = {
    "dual_contouring": DualContouring,
    "marching_cubes": MarchingCubes,
}


class ReconstructionWizard(QDialog):
    """Asistente de 4 pasos. Expone ``mesh`` y ``meta`` al terminar."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Muni — Nueva reconstrucción")
        self.setWindowIcon(load_icon("Muni.ico"))
        self.resize(980, 640)

        self.mesh = None
        self.meta: ModelMeta | None = None
        self.trace_result = None
        self.trace_spacing: tuple[float, float, float] | None = None

        self.volume: Volume3D | None = None
        self.paths: list = []
        self.objective: int = 40
        self.threshold_gray: int = 128
        self.line: tuple[int, int, int, int] | None = None
        self.seed_plane: int = 0
        self._threads: list = []
        self._seg_result = None
        self._pending_mesh = None
        self._pending_meta = None
        self._pending_trace_result = None
        self._pending_trace_spacing = None

        # ------------------------------------------------------------- páginas
        self.stack = QStackedWidget()
        self.page1 = self._build_page_open()
        self.page2 = self._build_page_params()
        self.page3 = self._build_page_calibration()
        self.page4 = self._build_page_process()
        for p in (self.page1, self.page2, self.page3, self.page4):
            self.stack.addWidget(p)

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
        layout.addLayout(btn_row)

        self._update_nav(0)

    # -------------------------------------------------------------- constructoras
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

    def _build_page_params(self) -> QWidget:
        page = QWidget()
        box = QGroupBox("Paso 2 — Parámetros del microscopio")
        self.spin_objective = QSpinBox()
        self.spin_objective.setRange(1, 200)
        self.spin_objective.setValue(40)
        self.spin_objective.setSuffix("×")

        grid = QGridLayout(box)
        grid.addWidget(QLabel("Objetivo (magnificación):"), 0, 0)
        grid.addWidget(self.spin_objective, 0, 1)

        note = QLabel(
            "La resolución XY y Z se obtiene de la calibración con la dendrita "
            "(paso siguiente); no hace falta acercar las imágenes."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #999;")

        outer = QVBoxLayout(page)
        outer.addWidget(box)
        outer.addWidget(note)
        outer.addStretch(1)
        return page

    def _build_page_calibration(self) -> QWidget:
        page = QWidget()
        box = QGroupBox("Paso 3 — Calibración (dibuja una dendrita y ajusta el umbral)")

        self.slice_view = SliceView()
        self.slice_view.draw_mode = True
        self.slice_view.line_drawn.connect(self._on_line_drawn)

        self.list_planes = QListWidget()
        self.list_planes.currentRowChanged.connect(self._on_plane_changed)

        self.btn_auto_threshold = QPushButton("Estimar umbral automático (Otsu)")
        self.btn_auto_threshold.clicked.connect(self._auto_threshold)

        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setRange(0, 255)
        self.slider_threshold.setValue(128)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)

        self.lbl_threshold = QLabel("Umbral (gris): 128")
        self.lbl_calib_state = QLabel("Dibuja una línea cruzando una dendrita (botón izquierdo).")
        self.lbl_calib_state.setStyleSheet("color: #999;")
        self.lbl_calib_state.setWordWrap(True)

        self.chk_show_sharpness = QCheckBox("Previsualizar nitidez (contraste local)")
        self.chk_show_sharpness.setChecked(False)
        self.chk_show_sharpness.toggled.connect(self._on_sharpness_toggled)

        right = QVBoxLayout()
        right.addWidget(QLabel("Planos:"))
        right.addWidget(self.list_planes)
        right.addWidget(self.chk_show_sharpness)
        right.addWidget(self.btn_auto_threshold)
        right.addWidget(self.lbl_threshold)
        right.addWidget(self.slider_threshold)
        right.addWidget(self.lbl_calib_state)
        right.addStretch(1)

        hb = QHBoxLayout(box)
        hb.addWidget(self.slice_view, 3)
        hb.addLayout(right, 1)

        outer = QVBoxLayout(page)
        outer.addWidget(box)
        outer.addStretch(1)
        return page

    def _build_page_process(self) -> QWidget:
        page = QWidget()
        box = QGroupBox("Paso 4 — Procesado")

        self.radio_dc = QRadioButton("Dual Contouring (recomendado)")
        self.radio_mc = QRadioButton("Marching Cubes (referencia)")
        self.radio_dc.setChecked(True)

        self.spin_diameter = QDoubleSpinBox()
        self.spin_diameter.setRange(0.1, 1000.0)
        self.spin_diameter.setValue(4.0)
        self.spin_diameter.setSuffix(" µm")
        self.spin_diameter.setDecimals(2)

        self.spin_iso = QDoubleSpinBox()
        self.spin_iso.setRange(0.01, 0.99)
        self.spin_iso.setValue(0.5)
        self.spin_iso.setSingleStep(0.05)

        self.chk_trace = QCheckBox("Generar tracing dendrítico (SWC)")
        self.chk_trace.setChecked(False)

        self.radio_seg_classical = QRadioButton("Clásico (Otsu)")
        self.radio_seg_focused = QRadioButton("Enfoque")
        self.radio_seg_classical.setChecked(True)

        self._grp_extraction = QButtonGroup(self)
        self._grp_extraction.addButton(self.radio_dc)
        self._grp_extraction.addButton(self.radio_mc)

        self._grp_segmentation = QButtonGroup(self)
        self._grp_segmentation.addButton(self.radio_seg_classical)
        self._grp_segmentation.addButton(self.radio_seg_focused)

        self.btn_run = QPushButton("Reconstruir")
        self.btn_run.clicked.connect(self._run)
        self.btn_run.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.lbl_progress = QLabel("Listo.")
        self.lbl_progress.setStyleSheet("color: #999;")

        self.seg_preview = SliceView()
        self.seg_preview.set_overlay_color(40, 255, 40)

        self.list_seg_planes = QListWidget()
        self.list_seg_planes.setMaximumHeight(150)
        self.list_seg_planes.currentRowChanged.connect(self._on_seg_plane_changed)

        self.btn_resgment = QPushButton("Volver a segmentar")
        self.btn_resgment.clicked.connect(self._reset_to_segmentation)
        self.btn_resgment.hide()

        self.btn_finish = QPushButton("Finalizar")
        self.btn_finish.clicked.connect(self._finish)
        self.btn_finish.hide()

        grid = QGridLayout(box)
        grid.addWidget(QLabel("Método de extracción:"), 0, 0)
        grid.addWidget(self.radio_dc, 0, 1)
        grid.addWidget(self.radio_mc, 1, 1)
        grid.addWidget(QLabel("Diámetro de la dendrita marcada:"), 2, 0)
        grid.addWidget(self.spin_diameter, 2, 1)
        grid.addWidget(QLabel("Umbral de probabilidad (isonivel 3D):"), 3, 0)
        grid.addWidget(self.spin_iso, 3, 1)
        grid.addWidget(QLabel("Método de segmentación:"), 4, 0, 2, 1)
        grid.addWidget(self.radio_seg_classical, 4, 1)
        grid.addWidget(self.radio_seg_focused, 5, 1)
        grid.addWidget(self.chk_trace, 6, 0, 1, 2)
        grid.addWidget(self.btn_run, 7, 0, 1, 2)
        grid.addWidget(self.progress, 8, 0, 1, 2)
        grid.addWidget(self.lbl_progress, 9, 0, 1, 2)
        grid.addWidget(self.btn_resgment, 10, 0, 1, 2)
        grid.addWidget(self.btn_finish, 11, 0, 1, 2)

        seg_row = 12
        grid.addWidget(QLabel("Vista de segmentación:"), seg_row, 0)
        grid.addWidget(self.list_seg_planes, seg_row, 1)
        grid.addWidget(self.seg_preview, seg_row + 1, 0, 1, 2)

        outer = QVBoxLayout(page)
        outer.addWidget(box)
        outer.addStretch(1)
        return page

    # ---------------------------------------------------------------- navegación
    def _update_nav(self, index: int) -> None:
        self.btn_back.setEnabled(index > 0)
        self.btn_next.setVisible(index < 3)
        self.stack.setCurrentIndex(index)
        if index == 2:
            self.slice_view.fit_to_window()
        if index == 3:
            self._fill_seg_planes()

    def _go_back(self) -> None:
        if self.stack.currentIndex() > 0:
            self._update_nav(self.stack.currentIndex() - 1)

    def _go_next(self) -> None:
        index = self.stack.currentIndex()
        if index == 0 and self.volume is None:
            QMessageBox.warning(self, "Muni", "Primero abre una pila de imágenes.")
            return
        if index == 2 and self.line is None:
            QMessageBox.warning(self, "Muni", "Dibuja una línea cruzando una dendrita.")
            return
        if index < 3:
            self._update_nav(index + 1)
        if index == 2:
            self.objective = self.spin_objective.value()
            self.btn_run.setEnabled(True)

    # ---------------------------------------------------------------- paso 1
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
        self.list_planes.setCurrentRow(volume.dimz // 2)
        self._update_nav(0)

    # ---------------------------------------------------------------- paso 3
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

    def _auto_threshold(self) -> None:
        if self.volume is None:
            return
        self.threshold_gray = otsu_gray_threshold(self.volume, denoise=True)
        self.slider_threshold.setValue(self.threshold_gray)
        self.lbl_threshold.setText(f"Umbral (gris): {self.threshold_gray}")
        self.slice_view.set_overlay(self._threshold_overlay())

    def _on_threshold_changed(self, value: int) -> None:
        self.threshold_gray = value
        self.lbl_threshold.setText(f"Umbral (gris): {value}")
        self.slice_view.set_overlay(self._threshold_overlay())

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

    # ---------------------------------------------------------------- paso 4
    def _run(self) -> None:
        if self.volume is None or self.line is None:
            return
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_resgment.hide()
        self.btn_finish.hide()
        self.progress.setValue(0)
        self.lbl_progress.setText("Calibrando...")
        self._seg_result = None
        self._fill_seg_planes()

        try:
            self._start_pipeline()
        except Exception as exc:  # noqa: BLE001 - nunca dejar botones bloqueados
            self.btn_run.setEnabled(True)
            self.btn_cancel.setEnabled(True)
            import traceback

            print(f"[muni-error]{os.linesep}{traceback.format_exc()}", file=sys.stderr, flush=True)
            QMessageBox.critical(self, "Muni — Error", f"No se pudo iniciar la segmentación:\n{exc}")

    def _start_pipeline(self) -> None:
        if self.volume is None or self.line is None:
            return
        diameter = float(self.spin_diameter.value())

        volume = self.volume
        cal = measure_on_grayscale(volume, self.threshold_gray, self.seed_plane, self.line)
        if cal.z_extent_planes > 0 and cal.max_diameter_px > 0:
            volume.spacing_xy_um = cal.spacing_xy_um(diameter)
            volume.spacing_z_um = cal.spacing_z_um(diameter)
            self.lbl_progress.setText(
                f"µm/píxel: {volume.spacing_xy_um:.4f} · µm/plano: {volume.spacing_z_um:.4f}"
            )
        else:
            QMessageBox.warning(
                self, "Muni", "No se detectó la dendrita en Z; se usará espaciado 1.0 (sin escala física)."
            )

        def task(progress):
            progress(5, "Segmentando...")
            if self.radio_seg_focused.isChecked():
                seg = FocusedSegmenter(denoise=True, keep_largest=True).segment(volume)
            else:
                seg = ClassicalSegmenter(denoise=True, keep_largest=True).segment(volume)
            progress(50, "Segmentación lista.")
            return seg

        self._spawn(task, self._on_segmentation_done)

    def _on_segmentation_done(self, seg) -> None:
        self.btn_resgment.setEnabled(True)
        if seg is None:
            return
        self._seg_result = seg
        self._fill_seg_planes()
        self._on_seg_plane_changed(self.list_seg_planes.currentRow())
        self.btn_resgment.show()
        self.start_extraction()

    def start_extraction(self) -> None:
        if self.volume is None or self.line is None or self._seg_result is None:
            return
        isolevel = float(self.spin_iso.value())
        method = "dual_contouring" if self.radio_dc.isChecked() else "marching_cubes"
        do_trace = self.chk_trace.isChecked()
        volume = self.volume
        seg = self._seg_result
        spacing = (
            volume.spacing_z_um,
            volume.spacing_xy_um,
            volume.spacing_xy_um,
        )

        def task(progress):
            progress(55, "Extrayendo superficie...")
            extractor = RADIO_METHODS[method]()
            result = extractor.extract(seg.probability, isovalue=isolevel, spacing=spacing)
            progress(70, "Suavizando malla...")
            mesh = taubin_smooth(result.mesh, iterations=3)

            trace_result = None
            if do_trace:
                progress(80, "Tracing dendrítico...")
                tracer = SkimageTracer()
                trace_result = tracer.trace(seg.mask, spacing=spacing)
                progress(95, f"Longitud dendrítica: {trace_result.total_length_um:.1f} µm")

            meta = ModelMeta(
                dimx=volume.dimx,
                dimy=volume.dimy,
                dimz=volume.dimz,
                spacing_xy_um=volume.spacing_xy_um,
                spacing_z_um=volume.spacing_z_um,
                isolevel=isolevel,
                objective=self.objective,
                dendrite_diameter_um=float(self.spin_diameter.value()),
                segmenter=seg.method,
                extractor=result.method,
                source_stack=", ".join(p.name for p in self.paths[:3]),
            )
            progress(100, "Listo")
            return mesh, meta, trace_result, spacing

        self._spawn(task, self._on_extraction_done)

    def _on_extraction_done(self, result) -> None:
        self.btn_run.setEnabled(True)
        if result is None:
            return
        mesh, meta, trace_result, trace_spacing = result
        if mesh.is_empty:
            QMessageBox.warning(
                self, "Muni", "La malla quedó vacía: revisa el umbral o el método de segmentación."
            )
            return
        self._pending_mesh = mesh
        self._pending_meta = meta
        self._pending_trace_result = trace_result
        self._pending_trace_spacing = trace_spacing
        self.btn_finish.show()
        self.lbl_progress.setText("Extracción completada. Revisa la segmentación y pulsa Finalizar.")

    def _spawn(self, task, on_finished) -> None:
        thread = run_in_thread(
            task,
            on_finished=on_finished,
            on_error=self._on_error,
            on_progress=self._on_progress,
        )
        self._threads.append(thread)

    def _reset_to_segmentation(self) -> None:
        self.btn_resgment.hide()
        self.btn_finish.hide()
        self.btn_run.setEnabled(True)
        self._seg_result = None
        self._pending_mesh = None
        self._pending_meta = None
        self._pending_trace_result = None
        self._pending_trace_spacing = None
        self.list_seg_planes.clear()
        self.seg_preview.clear()
        self.lbl_progress.setText("Listo.")
        self.progress.setValue(0)

    def _fill_seg_planes(self) -> None:
        self.list_seg_planes.clear()
        if self.volume is None:
            return
        for i in range(self.volume.dimz):
            name = self.paths[i].name if i < len(self.paths) else f"Plano {i}"
            self.list_seg_planes.addItem(name)
        self.list_seg_planes.setCurrentRow(self.volume.dimz // 2)

    def _on_seg_plane_changed(self, row: int) -> None:
        if row < 0 or self.volume is None:
            return
        plane = self.volume.plane(row)
        overlay = self._seg_result.mask[row] if self._seg_result is not None else None
        self.seg_preview.set_plane(plane, overlay)
        self.seg_preview.fit_to_window()

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress.setValue(percent)
        self.lbl_progress.setText(message)

    def _on_error(self, message: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.btn_resgment.setEnabled(True)
        import sys

        print(f"[muni-error]{os.linesep}{message}", file=sys.stderr, flush=True)
        QMessageBox.critical(self, "Muni — Error", message)

    def _finish(self) -> None:
        self.mesh = self._pending_mesh
        self.meta = self._pending_meta
        self.trace_result = self._pending_trace_result
        self.trace_spacing = self._pending_trace_spacing
        self.accept()