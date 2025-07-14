import numpy as np
import pytest

from compare_brain_maps.utils import surface


@pytest.mark.parametrize(
    "atlas, density",
    [
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
    ],
)
def test_load_polymesh(atlas, density):
    """test all neuromaps atlases can be loaded as a polymesh"""
    polymesh = surface.load_polymesh(atlas=atlas, density=density, mesh="inflated")
    assert len(polymesh.parts) == 2  # L/R hemispheres exist
    assert np.round(polymesh.n_vertices / 2000) == float(density[:-1])
