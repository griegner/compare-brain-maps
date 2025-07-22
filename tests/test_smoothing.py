import numpy as np
import pytest

from compare_brain_maps.smoothing import HeatKernelSmoother, NearestNeighborSmoother
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
        data, data_smoothed = X.data.parts[hemi], X_smoothed.data.parts[hemi]
        assert not np.array_equal(data, data_smoothed), f"{hemi} smoothing not applied"
        np.testing.assert_array_almost_equal(data.mean(), data_smoothed.mean(), decimal=3)
        assert data.std() > data_smoothed.std(), f"{hemi} smoothing does not reduce variability"


@pytest.mark.parametrize("atlas, density, n_vertices", [("fsaverage", "3k", 2562), ("fsLR", "4k", 4002)])
def test_heat_kernel_smoother(atlas, density, n_vertices):
    rng = np.random.default_rng(seed=0)
    data = {"left": rng.standard_normal(n_vertices), "right": rng.standard_normal(n_vertices)}
    data = {key: (values - np.mean(values)) / np.std(values) for key, values in data.items()}  # standardize

    smoother = HeatKernelSmoother()
    X = Surface(data=data, atlas=atlas, density=density, surface="inflated", mask_medial=False)
    X_smoothed = smoother.transform(X)

    assert isinstance(X_smoothed, Surface), "Surface object not returned"

    for hemi in ["left", "right"]:
        data, data_smoothed = X.data.parts[hemi], X_smoothed.data.parts[hemi]
        assert not np.array_equal(data, data_smoothed), f"{hemi} smoothing not applied"
        np.testing.assert_almost_equal(data.mean(), data_smoothed.mean(), decimal=2)
        assert data.std() > data_smoothed.std(), f"{hemi} smoothing does not reduce variability"
