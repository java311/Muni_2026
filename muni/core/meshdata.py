"""Estructura de malla triangular indexada.

``MeshData`` es el formato interno canónico: vértices, normales por vértice e
índices de triángulos. Es independiente de OpenGL y de GLTF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MeshData:
    """Malla triangular con normales por vértice.

    Parameters
    ----------
    vertices:
        Array ``float32`` de forma ``(N, 3)``.
    normals:
        Array ``float32`` de forma ``(N, 3)``, una normal por vértice.
    faces:
        Array ``int32`` de forma ``(M, 3)`` con índices a ``vertices``.
    name:
        Nombre opcional.
    """

    vertices: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        self.normals = np.asarray(self.normals, dtype=np.float32)
        self.faces = np.asarray(self.faces, dtype=np.int32)

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices debe ser (N,3), se recibió {self.vertices.shape}.")
        if self.normals.shape != self.vertices.shape:
            raise ValueError(
                f"normals {self.normals.shape} debe coincidir con vertices {self.vertices.shape}."
            )
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces debe ser (M,3), se recibió {self.faces.shape}.")

    # ---------------------------------------------------------------- sizes
    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.vertex_count == 0 or self.face_count == 0

    # ------------------------------------------------------------- helpers
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve ``(mínimo, máximo)`` de la caja delimitadora por columna."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def face_normals(self) -> np.ndarray:
        """Normales de cara (producto cruzado, sin normalizar) por triángulo."""
        v = self.vertices[self.faces]  # (M, 3, 3)
        e1 = v[:, 1] - v[:, 0]
        e2 = v[:, 2] - v[:, 0]
        return np.cross(e1, e2)

    def validate(self) -> list[str]:
        """Devuelve una lista de problemas detectados (vacía si la malla es válida)."""
        errors: list[str] = []
        n = self.vertex_count
        if n == 0:
            errors.append("Sin vértices.")
        if self.face_count == 0:
            errors.append("Sin caras.")
        if not np.all(np.isfinite(self.vertices)):
            errors.append("Hay vértices no finitos (NaN/inf).")
        if not np.all(np.isfinite(self.normals)):
            errors.append("Hay normales no finitas (NaN/inf).")
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= n):
            errors.append("Hay índices de cara fuera de rango.")
        fn = self.face_normals()
        norms = np.linalg.norm(fn, axis=1)
        if self.face_count and np.any(norms == 0):
            errors.append("Hay triángulos degenerados (área cero).")
        return errors

    @classmethod
    def from_triangle_soup(
        cls,
        triangles: np.ndarray,
        name: str = "",
        weld: bool = True,
        vertex_normals: np.ndarray | None = None,
    ) -> "MeshData":
        """Construye una malla indexada a partir de una "sopa" de triángulos.

        Parameters
        ----------
        triangles:
            Array ``(T, 3, 3)`` con las coordenadas de los 3 vértices de cada triángulo.
        name:
            Nombre opcional.
        weld:
            Si es ``True``, se deduplican los vértices coincidentes (por igualdad
            exacta) para obtener una malla indexada compacta.
        vertex_normals:
            Array opcional ``(T*3, 3)`` con la normal por vértice de la sopa, en
            el mismo orden que ``triangles.reshape(-1, 3)``. Si se omite, las
            normales se calculan a partir de la geometría (media por área).
        """
        tris = np.asarray(triangles, dtype=np.float32)
        if tris.ndim != 3 or tris.shape[1:] != (3, 3):
            raise ValueError(f"triangles debe ser (T,3,3), se recibió {tris.shape}.")

        flat = tris.reshape(-1, 3)
        if weld and flat.shape[0]:
            # Deduplicación por igualdad exacta de bits.
            uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
            faces = inverse.reshape(-1, 3).astype(np.int32)
        else:
            uniq = flat
            faces = np.arange(flat.shape[0], dtype=np.int32).reshape(-1, 3)

        # Eliminar triángulos degenerados (índices repetidos o colineales).
        a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
        keep = (a != b) & (b != c) & (a != c)
        faces = faces[keep]
        if faces.size:
            tv = uniq[faces]
            cross = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
            keep_area = np.linalg.norm(cross, axis=1) > 0
            faces = faces[keep_area]

        if vertex_normals is not None:
            vn = np.asarray(vertex_normals, dtype=np.float32).reshape(-1, 3)
            if vn.shape[0] != flat.shape[0]:
                raise ValueError(
                    f"vertex_normals tiene {vn.shape[0]} filas, se esperaban {flat.shape[0]}."
                )
            normals = np.zeros((uniq.shape[0], 3), dtype=np.float32)
            np.add.at(normals, inverse, vn)
            lengths = np.linalg.norm(normals, axis=1)
            ok = lengths > 0
            normals[ok] = normals[ok] / lengths[ok, None]
            return cls(vertices=uniq, normals=normals, faces=faces, name=name)

        # Normales de vértice como media (ponderada por área) de las normales de cara.
        v = uniq
        tri_verts = v[faces]
        e1 = tri_verts[:, 1] - tri_verts[:, 0]
        e2 = tri_verts[:, 2] - tri_verts[:, 0]
        cross = np.cross(e1, e2)  # (M,3), magnitud = 2*área
        accum = np.zeros_like(v)
        np.add.at(accum, faces.reshape(-1), np.repeat(cross, 3, axis=0))
        lengths = np.linalg.norm(accum, axis=1)
        normals = np.zeros_like(v)
        ok = lengths > 0
        normals[ok] = accum[ok] / lengths[ok, None]

        return cls(vertices=v, normals=normals, faces=faces, name=name)

    def __repr__(self) -> str:  # pragma: no cover - diagnóstico
        return (
            f"MeshData(vertices={self.vertex_count}, faces={self.face_count}, "
            f"name={self.name!r})"
        )
