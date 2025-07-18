import numpy as np
import pytest

from compare_brain_maps.utils import Surface

atlas_density_params = [
    ("fsaverage", "3k", 2562),
    ("fsaverage", "10k", 10242),
    ("fsaverage", "41k", 40962),
    ("fsaverage", "164k", 163842),
    ("fsLR", "4k", 4002),
    ("fsLR", "8k", 7842),
    ("fsLR", "32k", 32492),
    ("fsLR", "164k", 163842),
    ("civet", "41k", 40962),
    ("civet", "164k", 163842),
]


@pytest.mark.parametrize("atlas, density, n_vertices", atlas_density_params)
def test_surface(atlas, density, n_vertices):
    """test all neuromaps atlases can be loaded as a Surface"""
    data = {"left": np.ones(n_vertices), "right": np.ones(n_vertices)}
    surf = Surface(data=data, atlas=atlas, density=density, surface="inflated")
    assert len(surf.mesh.parts) == 2, "missing hemisphere"
    assert surf.mesh.n_vertices // 2 == n_vertices, "incorrect density"
    assert surf.medial.parts["left"].size > surf.medial.parts["left"].sum(), "incorrect medial wall mask"


@pytest.mark.parametrize("atlas, density, n_vertices", atlas_density_params)
def test_adjacency(atlas, density, n_vertices):
    data = {"left": np.ones(n_vertices), "right": np.ones(n_vertices)}
    surf = Surface(data=data, atlas=atlas, density=density, surface="inflated")
    adjacency = surf.get_adjacency()

    for hemi in ["left", "right"]:
        assert adjacency[hemi].nnz > 0, f"{hemi} entries are all zero"
        assert (adjacency[hemi] != adjacency[hemi].T).nnz == 0, f"{hemi} is asymmetric"
        assert np.all(adjacency[hemi].diagonal() == 0), f"{hemi} diagonals are non-zero"
        sparsity = adjacency[hemi].nnz / (adjacency[hemi].shape[0] ** 2)
        assert sparsity < 0.01, f"{hemi} is not sparse"
