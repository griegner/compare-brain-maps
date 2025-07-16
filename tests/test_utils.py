import numpy as np
import pytest

from compare_brain_maps.utils import surface

atlas_density_params = [
    ("fsaverage", "3k"),
    ("fsaverage", "10k"),
    ("fsaverage", "41k"),
    ("fsaverage", "164k"),
    ("fsLR", "4k"),
    ("fsLR", "8k"),
    ("fsLR", "32k"),
    ("fsLR", "164k"),
    ("civet", "41k"),
    ("civet", "164k"),
]


@pytest.mark.parametrize("atlas, density", atlas_density_params)
def test_load_surface_atlas(atlas, density):
    """test all neuromaps atlases can be loaded as a SurfaceImage"""
    surface_atlas = surface.load_surface_atlas(atlas=atlas, density=density, mesh="inflated")
    assert len(surface_atlas.mesh.parts) == 2, "missing hemisphere"
    assert np.round(surface_atlas.mesh.n_vertices / 2000) == float(density[:-1]), "incorrect density"
    assert surface_atlas.mesh.n_vertices == surface_atlas.data.shape[0], "shape mismatch between mesh and data"
    assert surface_atlas.data.parts["left"].size > surface_atlas.data.parts["left"].sum(), "incorrect medial wall mask"


@pytest.mark.parametrize("atlas, density", atlas_density_params)
def test_surface_adjacency_matrices(atlas, density):
    """test all neuromaps atlases have valid adjacency matrices"""
    A = surface.adjacency_matrices(atlas=atlas, density=density, mesh="inflated")

    for hemi in ["left", "right"]:
        assert A[hemi].nnz > 0, f"{hemi} entries are all zero"
        assert (A[hemi] != A[hemi].T).nnz == 0, f"{hemi} is asymmetric"
        assert np.all(A[hemi].diagonal() == 0), f"{hemi} diagonals are non-zero"
        sparsity = A[hemi].nnz / (A[hemi].shape[0] ** 2)
        assert sparsity < 0.01, f"{hemi} is not sparse"
