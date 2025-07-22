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
    X = Surface(data=data, atlas=atlas, density=density, surface="inflated", mask_medial=False)
    X_masked = Surface(data=data, atlas=atlas, density=density, surface="inflated", mask_medial=True)
    assert len(X.mesh.parts) == 2, "missing hemisphere"
    assert X.mesh.n_vertices // 2 == n_vertices, "incorrect density"

    for hemi in ["left", "right"]:
        assert np.all(~np.isnan(X.data.parts[hemi])), f"{hemi} contains nans"
        assert np.isnan(X_masked.data.parts[hemi]).sum() > 0, f"{hemi} medial wall not excluded"


@pytest.mark.parametrize("atlas, density, n_vertices", atlas_density_params)
def test_adjacency(atlas, density, n_vertices):
    data = {"left": np.ones(n_vertices), "right": np.ones(n_vertices)}
    X = Surface(data=data, atlas=atlas, density=density, surface="inflated", mask_medial=False)
    A = X.get_adjacency()
    X_masked = Surface(data=data, atlas=atlas, density=density, surface="inflated", mask_medial=True)
    A_masked = X_masked.get_adjacency()

    for hemi in ["left", "right"]:
        assert A[hemi].nnz > 0, f"{hemi} entries are all zero"
        assert (A[hemi] != A[hemi].T).nnz == 0, f"{hemi} is asymmetric"
        assert np.all(A[hemi].diagonal() == 0), f"{hemi} diagonals are non-zero"
        sparsity = A[hemi].nnz / (A[hemi].shape[0] ** 2)
        assert sparsity < 0.01, f"{hemi} is not sparse"
        assert A[hemi].nnz > A_masked[hemi].nnz, f"{hemi} medial wall not excluded"
