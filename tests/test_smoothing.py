import numpy as np
import pytest

from compare_brain_maps.smoothing import NearestNeighborSmoother
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
def test_nearest_neighbor_smoother(atlas, density, n_vertices):
    rng = np.random.default_rng(seed=0)
    data = {"left": rng.standard_normal(n_vertices), "right": rng.standard_normal(n_vertices)}
    data = {key: (values - np.mean(values)) / np.std(values) for key, values in data.items()}  # standardize

    smoother = NearestNeighborSmoother(n_iterations=3)
    X = Surface(data=data, atlas=atlas, density=density, surface="inflated", mask_medial=False)
    X_smoothed = smoother.transform(X)

    assert isinstance(X_smoothed, Surface), "Surface object not returned"

    for hemi in ["left", "right"]:
        assert not np.array_equal(X.data.parts[hemi], X_smoothed.data.parts[hemi]), f"{hemi} smoothing not applied"
        np.testing.assert_almost_equal(X_smoothed.data.parts[hemi].mean(), 0, err_msg=f"{hemi} mean not zero")
        np.testing.assert_almost_equal(X_smoothed.data.parts[hemi].std(), 1, err_msg=f"{hemi} std not one")
