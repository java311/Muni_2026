"""Pruebas de la geometría del esqueleto y del resaltado de selección."""

import numpy as np

from muni.trace.base import SWC_DENDRITE, TraceResult
from muni.view.skeleton_mesh import (
    CONTEXT_COLOR,
    HIGHLIGHT_COLOR,
    TYPE_COLORS,
    build_skeleton_geometry,
)


def _two_branches():
    # Camino 0->1 (rama 0) y 2->3->4 (rama 1). Grid (z, y, x).
    coords = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 2], [0, 2, 2]], dtype=float)
    parents = np.array([-1, 0, 1, 2, 3], dtype=np.int64)
    labels = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    return TraceResult(
        coords=coords,
        radii=np.ones(5),
        parents=parents,
        branch_labels=labels,
        total_length_um=4.0,
        spacing=(1.0, 1.0, 1.0),
        types=np.full(5, SWC_DENDRITE, dtype=np.int64),
    )


def test_highlight_branch_uses_highlight_color():
    _p, _n, groups = build_skeleton_geometry(_two_branches(), highlight_branches={1})
    colors = [color for _o, _c, color in groups]
    assert HIGHLIGHT_COLOR in colors
    assert TYPE_COLORS[SWC_DENDRITE] in colors


def test_highlight_multiple_branches_merges_into_one_group():
    _p, _n, groups = build_skeleton_geometry(_two_branches(), highlight_branches={0, 1})
    colors = [color for _o, _c, color in groups]
    assert colors == [HIGHLIGHT_COLOR]


def test_no_highlight_has_no_highlight_color():
    _p, _n, groups = build_skeleton_geometry(_two_branches())
    assert HIGHLIGHT_COLOR not in [color for _o, _c, color in groups]


def test_context_branch_uses_context_color():
    _p, _n, groups = build_skeleton_geometry(
        _two_branches(), highlight_branches={1}, context_branches={0}
    )
    colors = [color for _o, _c, color in groups]
    assert HIGHLIGHT_COLOR in colors
    assert CONTEXT_COLOR in colors


def test_highlight_wins_over_context_for_same_branch():
    _p, _n, groups = build_skeleton_geometry(
        _two_branches(), highlight_branches={0}, context_branches={0}
    )
    colors = [color for _o, _c, color in groups]
    assert HIGHLIGHT_COLOR in colors
    assert CONTEXT_COLOR not in colors