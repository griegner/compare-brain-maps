import numpy as np
import pytest

from compare_brain_maps.resampling import PermutationResampler, SubsampleResampler
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
def test_subsample_resampler(atlas, density, n_vertices):
    """test resampler on pair of uncorrelated maps"""
    rng = np.random.default_rng(seed=0)

    X_data = {"left": rng.normal(size=n_vertices), "right": rng.normal(size=n_vertices)}
    Y_data = {"left": rng.normal(size=n_vertices), "right": rng.normal(size=n_vertices)}
    X = Surface(X_data, atlas=atlas, density=density, surface="inflated", mask_medial=False)
    Y = Surface(Y_data, atlas=atlas, density=density, surface="inflated", mask_medial=False)

    resampler = SubsampleResampler(n_subsamples=100, patch_size=4, seed=0)
    resampler.fit(X, Y)

    np.testing.assert_allclose(resampler.param_, 0, atol=0.015)
    np.testing.assert_allclose(np.mean(resampler.params_), 0, atol=0.005)


@pytest.mark.parametrize("atlas, density, n_vertices", atlas_density_params)
def test_permutation_resampler(atlas, density, n_vertices):
    """test resampler on pair of uncorrelated maps"""
    rng = np.random.default_rng(seed=0)

    X_data = {"left": rng.normal(size=n_vertices), "right": rng.normal(size=n_vertices)}
    Y_data = {"left": rng.normal(size=n_vertices), "right": rng.normal(size=n_vertices)}
    X = Surface(X_data, atlas=atlas, density=density, surface="inflated", mask_medial=False)
    Y = Surface(Y_data, atlas=atlas, density=density, surface="inflated", mask_medial=False)

    resampler = PermutationResampler(n_permutations=10, seed=0)
    resampler.fit(X, Y)

    np.testing.assert_allclose(resampler.param_, 0, atol=0.015)
    np.testing.assert_allclose(np.mean(resampler.params_), 0, atol=0.005)
