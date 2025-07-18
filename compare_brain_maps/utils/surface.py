"""utilities for surface maps"""

import numpy as np
from neuromaps.datasets import fetch_atlas
from nilearn.surface import PolyData, PolyMesh
from nilearn.surface.surface import _check_data_and_mesh_compat
from scipy.sparse import csr_array


def _adjacency(faces, mask=None):
    """return adjacency matrix for one hemisphere, excluding the medial wall"""
    n_vertices = faces.max() + 1

    edges = np.vstack([faces[:, [0, 1]], faces[:, [0, 2]], faces[:, [1, 2]]])
    sorted_edges = np.sort(edges, axis=1)
    unique_edges = np.unique(sorted_edges, axis=0)

    if mask is not None:
        (idx,) = np.where(mask)
        mask = np.any(np.isin(unique_edges, idx), axis=1)
        unique_edges = unique_edges[mask]

    i, j = unique_edges[:, 0], unique_edges[:, 1]

    ones = np.ones(len(unique_edges) * 2, dtype=np.int8)
    row_indices = np.concatenate([i, j])
    col_indices = np.concatenate([j, i])

    return csr_array((ones, (row_indices, col_indices)), shape=(n_vertices, n_vertices))


class Surface:
    """Surface object containing data, mesh, and medial wall mask for both hemispheres.

    Parameters
    ----------
    data : dict of numpy.1darray or str or pathlib.Path
        dictionary of data arrays/giftis whose keys must be a subset of {"left", "right"}
    atlas : str, optional
        atlas in `{"civet", "fsaverage", "fsLR"}`
    density : str, optional
        civet in `{"41k", "164k"}`\\
        fsaverage in `{"3k", "10k", "41k", "164k"}`\\
        fsLR in `{"4k", "8k", "32k", "164k"}`\\
        *note: {3k=2562, 4k=4002, 8k=7842, 10k=10242, 32k=32492, 41k=40962, 164k=163842}
    surface : str, optional
        civet in `{"white", "midthickness", "inflated", "veryinflated", "sphere"}`\\
        fsaverage in `{"white", "pial", "inflated", "sphere"}`\\
        fsLR in `{"midthickness", "inflated", "veryinflated", "sphere"}`

    Attributes
    ----------
    data : nilearn.surface.PolyData
        surface data for both hemispheres
    medial : nilearn.surface.PolyData
        medial wall mask for both hemispheres
    mesh : nilearn.surface.PolyMesh
        surface meshes (vertices and faces) for both hemispheres
    shape : tuple
        total number of vertices for both hemispheres

    Examples
    --------
    >>> import numpy as np
    >>> n_vertices = 2562
    >>> data = {
    ...     "left": np.random.randn(n_vertices),
    ...     "right": np.random.randn(n_vertices)
    ... }
    >>> surf = Surface(data, atlas="fsaverage", density="3k", surface="pial")
    >>> surf.shape
    (5124,)
    """

    def __init__(self, data, atlas="fsaverage", density="3k", surface="pial"):
        giftis = fetch_atlas(atlas, density)

        self.data = PolyData(**data)
        self.medial = PolyData(left=giftis["medial"].L, right=giftis["medial"].R)
        self.mesh = PolyMesh(left=giftis[surface].L, right=giftis[surface].R)

        _check_data_and_mesh_compat(self.mesh, self.data)
        _check_data_and_mesh_compat(self.mesh, self.medial)

    @property
    def shape(self):
        """number of vertices for l+r hemispheres"""
        return self.data.shape

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.shape}>"

    def get_adjacency(self):
        """Adjacency matrices for the left and right hemispheres of a surface atlas.
        Elements *i,j* are `1` if vertex pairs are adjacent, `0` otherwise.

        Returns
        -------
        dict of scipy.sparse.csr_array of shape (n_vertices, n_vertices)
            dictionary of sparse adjacency matrices per hemisphere, excluding the medial wall:
            - "left": csr_array for the left hemisphere
            - "right": csr_array for the right hemisphere
        """
        adjacency_left = _adjacency(self.mesh.parts["left"].faces, self.medial.parts["left"])
        adjacency_right = _adjacency(self.mesh.parts["right"].faces, self.medial.parts["right"])
        return {"left": adjacency_left, "right": adjacency_right}

    def get_distance(self):
        pass
