"""Pruebas del tracing tipado: soma, bosque multicomponente, tipos y SWC."""

import numpy as np

from muni.trace.base import SWC_AXON, SWC_DENDRITE, SWC_SOMA
from muni.trace.skimage_tracer import SkimageTracer
from muni.trace.soma import detect_soma
from muni.trace.swc import load_swc, write_swc


def _sphere(shape, center, radius):
    z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    d = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    return d <= radius


def _neuron_mask():
    shape = (20, 40, 40)
    mask = _sphere(shape, (10, 20, 20), 5)
    z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    dendrite = (np.abs(y - 20) <= 3) & (np.abs(z - 10) <= 3) & (x >= 18) & (x < 38)
    axon = (np.abs(x - 20) <= 1) & (np.abs(z - 10) <= 1) & (y >= 6) & (y < 18)
    fragment = _sphere(shape, (10, 4, 4), 2)
    return mask | dendrite | axon | fragment


def _reachable_from_root(parents):
    for node in range(len(parents)):
        seen = set()
        cur = node
        while cur != -1 and cur not in seen:
            seen.add(int(cur))
            cur = int(parents[cur])
        if cur != -1:
            return False
    return True


def test_detect_soma_center_and_radius():
    mask = _sphere((20, 40, 40), (10, 20, 20), 5)
    soma = detect_soma(mask, spacing=(1.0, 1.0, 1.0))
    assert soma is not None
    assert np.allclose(soma.center_um, [20.0, 20.0, 10.0], atol=1.5)  # mundo (x, y, z)
    assert 3.0 < float(np.mean(soma.radii_um)) < 7.0


def test_detect_soma_on_elongated_shape_stays_thin():
    z, y, _ = np.mgrid[0:10, 0:60, 0:60]
    bar = (np.abs(y - 30) <= 3) & (np.abs(z - 5) <= 3)  # largo en X, grosor 3
    soma = detect_soma(bar, spacing=(1.0, 1.0, 1.0))
    assert soma is not None
    # El radio no debe inflarse con la longitud de la barra (~60).
    assert float(np.max(soma.radii_um)) <= 6.0


def test_soma_is_clamped_to_neurite_thickness():
    shape = (40, 80, 80)
    z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    blob = (z - 20) ** 2 + (y - 40) ** 2 + (x - 40) ** 2 <= 15**2
    proc1 = (np.abs(y - 40) <= 1) & (np.abs(z - 20) <= 1) & (x >= 2) & (x < 78)
    proc2 = (np.abs(x - 40) <= 1) & (np.abs(z - 20) <= 1) & (y >= 2) & (y < 78)
    result = SkimageTracer().trace(blob | proc1 | proc2, spacing=(1.0, 1.0, 1.0))
    assert result.has_soma
    # El pico de la EDT está en el blob (radio ~15); debe recortarse al grosor de
    # las neuritas (~2), no quedar del tamaño del cruce.
    assert result.soma_radii_um[0] <= 3.0 * float(np.median(result.radii)) + 1e-9
    assert result.soma_radii_um[0] < 10.0


def test_tracer_labels_every_node_no_orphans():
    result = SkimageTracer().trace(_neuron_mask(), spacing=(1.0, 1.0, 1.0))
    assert result.n_nodes > 0
    assert _reachable_from_root(result.parents)
    roots = np.flatnonzero(result.parents == -1)
    assert len(roots) >= 2  # soma+dendrita y fragmento separado
    assert (result.types == SWC_SOMA).sum() >= 1
    assert (result.types == SWC_AXON).sum() >= 1
    assert (result.types == SWC_DENDRITE).sum() >= 1
    assert result.total_length_um > 0
    assert len(result.branch_labels) == result.n_nodes


def test_swc_roundtrip_preserves_types(tmp_path):
    result = SkimageTracer().trace(_neuron_mask(), spacing=(1.0, 1.0, 1.0))
    path = write_swc(tmp_path / "neurona", result)
    back, spacing = load_swc(path)
    assert back.n_nodes == result.n_nodes
    assert np.array_equal(back.types, result.types)
    assert spacing == result.spacing
    assert back.soma_center_um is not None
