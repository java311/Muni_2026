"""Prueba de integración del pipeline completo (segmentar → extraer → guardar)."""

import numpy as np

from muni.core.gltf_io import load_model, save_model
from muni.core.meta import ModelMeta
from muni.core.volume import Volume3D
from muni.postprocess.smooth import taubin_smooth
from muni.reconstruct.dual_contouring import DualContouring
from muni.reconstruct.marching_cubes import MarchingCubes
from muni.segment.classical import ClassicalSegmenter


def _synthetic_neuron(shape=(16, 32, 32)):
    z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float32)
    # "Soma" esférico + una dendrita horizontal (cilindro) para forzar topología.
    d_soma = np.sqrt((z - 8) ** 2 + (y - 16) ** 2 + (x - 12) ** 2)
    soma = 1.0 / (1.0 + np.exp((d_soma - 6.0) / 1.0))
    dend = (y > 14) & (y < 18) & (z > 6) & (z < 10) & (x >= 12) & (x < 30)
    field = np.maximum(soma, dend.astype(np.float32) * 0.95)
    return field


def test_full_pipeline_roundtrip(tmp_path):
    field = _synthetic_neuron()
    vol = Volume3D(field, spacing_xy_um=0.5, spacing_z_um=2.0, name="sintetico")

    # Segmentación clásica sobre el campo de probabilidad.
    seg = ClassicalSegmenter(denoise=False, keep_largest=True).segment(vol)

    # Extracción (ambos extractores deben producir malla válida).
    for extractor in (MarchingCubes(), DualContouring()):
        result = extractor.extract(
            seg.probability, isovalue=0.5, spacing=(vol.spacing_z_um, vol.spacing_xy_um, vol.spacing_xy_um)
        )
        mesh = result.mesh
        assert not mesh.is_empty
        assert mesh.validate() == []

        mesh = taubin_smooth(mesh, iterations=2)
        assert mesh.validate() == []

        meta = ModelMeta(
            dimx=vol.dimx,
            dimy=vol.dimy,
            dimz=vol.dimz,
            spacing_xy_um=vol.spacing_xy_um,
            spacing_z_um=vol.spacing_z_um,
            segmenter=seg.method,
            extractor=result.method,
        )
        glb = tmp_path / f"{extractor.name}.glb"
        save_model(glb, mesh, meta)

        back_mesh, back_meta = load_model(glb)
        assert back_mesh.vertex_count == mesh.vertex_count
        assert back_mesh.face_count == mesh.face_count
        assert back_meta is not None
        assert back_meta.spacing_z_um == 2.0
