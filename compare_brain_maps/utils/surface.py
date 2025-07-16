"""utilities for surface maps"""

import numpy as np
from neuromaps.datasets import fetch_atlas
from neuromaps.points import make_surf_graph
from nilearn.surface import SurfaceImage
from scipy.sparse import csr_array


def load_surface_atlas(atlas: str, density: str, mesh: str) -> SurfaceImage:
    """Load a surface atlas at a given density.

    Parameters
    ----------
    atlas : str
        atlas in `{"civet", "fsaverage", "fsLR"}`
    density : str
        civet in `{"41k", "164k"}`\\
        fsaverage in `{"3k", "10k", "41k", "164k"}`\\
        fsLR in `{"4k", "8k", "32k", "164k"}`
    mesh : str
        civet in `{"white", "midthickness", "inflated", "veryinflated", "sphere"}`\\
        fsaverage in `{"white", "pial", "inflated", "sphere"}`\\
        fsLR in `{"midthickness", "inflated", "veryinflated", "sphere"}`


    Returns
    -------
    nilearn.surface.SurfaceImage
        contains the mesh (coordinates and faces) and data (mask excluding medial wall) of both hemispheres
    """
    giftis = fetch_atlas(atlas, density)
    return SurfaceImage(
        mesh={"left": giftis[mesh].L, "right": giftis[mesh].R},
        data={"left": giftis["medial"].L, "right": giftis["medial"].R},
    )


def _adjacency_matrix(faces, mask=None):
    """return adjacency matrix for one hemisphere, excluding the medial wall"""
    n_vertices = faces.max() + 1

    edges = np.vstack([faces[:, [0, 1]], faces[:, [0, 2]], faces[:, [1, 2]]])
    sorted_edges = np.sort(edges, axis=1)
    unique_edges = np.unique(sorted_edges, axis=0)
    i, j = unique_edges[:, 0], unique_edges[:, 1]

    ones = np.ones(len(unique_edges) * 2, dtype=np.int8)
    row_indices = np.concatenate([i, j])
    col_indices = np.concatenate([j, i])

    return csr_array((ones, (row_indices, col_indices)), shape=(n_vertices, n_vertices))


def adjacency_matrices(atlas: str, density: str, mesh: str) -> dict:
    """Adjacency matrices for the left and right hemispheres of a surface atlas.
    Elements *i,j* are `1` if vertex pairs are adjacent, `0` otherwise.

    Parameters
    ----------
    atlas : str
        Atlas in `{"civet", "fsaverage", "fsLR"}`
    density : str
        civet in `{"41k", "164k"}`\\
        fsaverage in `{"3k", "10k", "41k", "164k"}`\\
        fsLR in `{"4k", "8k", "32k", "164k"}`
    mesh : str
        civet in `{"white", "midthickness", "inflated", "veryinflated", "sphere"}`\\
        fsaverage in `{"white", "pial", "inflated", "sphere"}`\\
        fsLR in `{"midthickness", "inflated", "veryinflated", "sphere"}`

    Returns
    -------
    dict of scipy.sparse.csr_array
        dictionary containing sparse adjacency matrices per hemisphere, excluding the medial wall:
        - "left": csr_array for the left hemisphere
        - "right": csr_array for the right hemisphere
    """
    surface_atlas = load_surface_atlas(atlas, density, mesh)
    A_left = _adjacency_matrix(surface_atlas.mesh.parts["left"].faces)
    A_right = _adjacency_matrix(surface_atlas.mesh.parts["right"].faces)
    return {"left": A_left, "right": A_right}
