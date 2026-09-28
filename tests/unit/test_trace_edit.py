"""Pruebas de la edición del esqueleto: borrar, tipar, refinar radios y añadir."""

import numpy as np

from muni.trace.base import SWC_AXON, SWC_SOMA
from muni.trace.edit import (
    add_branch,
    clear_soma,
    delete_branch,
    refit_radii,
    reroot_trace,
    set_branch_type,
    set_soma_from_seed,
)
from muni.trace.graph import branch_ancestors, branch_tree
from muni.trace.skimage_tracer import SkimageTracer


def _y_mask():
    """Soma + tronco horizontal + dos brazos (uno en X, otro en Y)."""
    shape = (20, 44, 44)
    z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    soma = (z - 10) ** 2 + (y - 20) ** 2 + (x - 12) ** 2 <= 25
    trunk = (np.abs(y - 20) <= 2) & (np.abs(z - 10) <= 2) & (x >= 10) & (x < 26)
    arm_x = (np.abs(y - 20) <= 2) & (np.abs(z - 10) <= 2) & (x >= 26) & (x < 40)
    arm_y = (np.abs(x - 26) <= 2) & (np.abs(z - 10) <= 2) & (y >= 20) & (y < 40)
    return soma | trunk | arm_x | arm_y


def _trace():
    return SkimageTracer().trace(_y_mask(), spacing=(1.0, 1.0, 1.0))


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


def test_delete_branch_removes_subtree_and_keeps_graph_consistent():
    trace = _trace()
    branch_y = int(trace.branch_labels[np.argmax(trace.coords[:, 1])])
    before_nodes, before_len = trace.n_nodes, trace.total_length_um

    edited = delete_branch(trace, branch_y)

    assert edited.n_nodes < before_nodes
    assert edited.total_length_um < before_len
    assert _reachable_from_root(edited.parents)
    # La operación es inmutable.
    assert trace.n_nodes == before_nodes
    assert not np.any(edited.branch_labels == branch_y) or edited.n_nodes == 0


def test_delete_branch_rejects_unknown_id():
    trace = _trace()
    try:
        delete_branch(trace, 10_000)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("delete_branch debía rechazar una rama inexistente")


def test_set_branch_type_overrides_only_that_branch():
    trace = _trace()
    branch_x = int(trace.branch_labels[np.argmax(trace.coords[:, 2])])
    others = trace.types.copy()
    others[trace.branch_labels == branch_x] = -1

    edited = set_branch_type(trace, branch_x, SWC_AXON)

    assert np.all(edited.types[edited.branch_labels == branch_x] == SWC_AXON)
    assert np.array_equal(edited.types[others != -1], trace.types[others != -1])


def test_refit_radii_matches_cylinder_half_width():
    mask = _y_mask()
    trace = _trace()
    data = np.where(mask, 55.0, 255.0)

    edited = refit_radii(trace, data, spacing=(1.0, 1.0, 1.0), isolevel=128.0)

    # El brazo en X tiene semiancho 2 px; los rayos perpendiculares deben dar ~3.
    sel = (trace.coords[:, 2] > 30) & (trace.types != SWC_SOMA)
    assert sel.sum() >= 3
    assert 2.0 <= float(np.median(edited.radii[sel])) <= 3.5
    # El soma conserva su radio y el original no se muta.
    assert np.allclose(edited.radii[trace.types == SWC_SOMA], trace.radii[trace.types == SWC_SOMA])
    assert not np.array_equal(edited.radii, trace.radii)


def test_add_branch_reconnects_deleted_arm():
    mask = _y_mask()
    trace = _trace()
    branch_y = int(trace.branch_labels[np.argmax(trace.coords[:, 1])])
    pruned = delete_branch(trace, branch_y)
    before_nodes, before_len = pruned.n_nodes, pruned.total_length_um

    added = add_branch(
        pruned,
        mask,
        spacing=(1.0, 1.0, 1.0),
        start_grid=(10, 20, 25),
        end_grid=(10, 38, 26),
    )

    assert added.n_nodes > before_nodes
    assert added.total_length_um > before_len
    assert _reachable_from_root(added.parents)
    new_coords = added.coords[before_nodes:].astype(int).T
    assert mask[tuple(new_coords)].all()
    # Inmutable.
    assert pruned.n_nodes == before_nodes


def test_branch_ancestors_go_up_to_root():
    trace = _trace()
    tree = branch_tree(trace.branch_labels, trace.parents)
    roots = {bid for bid, parent, _ in tree if parent is None}
    for bid, _parent, depth in tree:
        chain = branch_ancestors(trace.branch_labels, trace.parents, bid)
        assert chain[0] == bid
        assert len(chain) == depth + 1
        assert chain[-1] in roots


def test_branch_tree_is_hierarchical_and_acyclic():
    trace = _trace()
    tree = branch_tree(trace.branch_labels, trace.parents)
    parent = {bid: pb for bid, pb, _ in tree}
    depths = [depth for _, _, depth in tree]

    assert depths.count(0) == 1  # la máscara es conexa: una sola raíz
    assert max(depths) >= 1
    for bid in parent:
        seen = set()
        cur: int | None = bid
        while cur is not None and cur not in seen:
            seen.add(cur)
            cur = parent[cur]
        assert cur is None


def test_reroot_makes_clicked_node_root_without_changing_length():
    trace = _trace()
    leaf = int(np.argmax(trace.coords[:, 1]))  # extremo del brazo en Y
    edited = reroot_trace(trace, leaf)

    assert int(edited.parents[leaf]) == -1
    assert edited.n_nodes == trace.n_nodes
    assert abs(edited.total_length_um - trace.total_length_um) < 1e-6
    assert _reachable_from_root(edited.parents)


def test_set_soma_from_seed_builds_ellipsoid_and_reroots():
    mask = _y_mask()
    trace = _trace()
    edited = set_soma_from_seed(trace, mask, (1.0, 1.0, 1.0), (10, 20, 12))

    assert edited.has_soma
    assert np.allclose(edited.soma_center_um, [12.0, 20.0, 10.0], atol=2.0)
    assert float(np.mean(edited.soma_radii_um)) > 2.0

    soma_nodes = np.flatnonzero(edited.types == SWC_SOMA)
    assert len(soma_nodes) == 1
    assert int(edited.parents[soma_nodes[0]]) == -1


def test_set_soma_radius_scale_scales_soma():
    mask = _y_mask()
    trace = _trace()
    small = set_soma_from_seed(trace, mask, (1.0, 1.0, 1.0), (10, 20, 12), radius_scale=1.0)
    big = set_soma_from_seed(trace, mask, (1.0, 1.0, 1.0), (10, 20, 12), radius_scale=2.0)
    assert float(big.soma_radii_um[0]) > float(small.soma_radii_um[0])


def test_clear_soma_removes_ellipsoid_and_type():
    mask = _y_mask()
    trace = _trace()
    with_soma = set_soma_from_seed(trace, mask, (1.0, 1.0, 1.0), (10, 20, 12))
    cleared = clear_soma(with_soma)

    assert not cleared.has_soma
    assert int((cleared.types == SWC_SOMA).sum()) == 0