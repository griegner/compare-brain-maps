"""utilities for surface maps"""

from neuromaps.datasets import fetch_atlas
from nilearn.surface import PolyMesh


def load_polymesh(atlas, density, mesh):
    """Load a surface mesh from a specified atlas and density.

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
    nilearn.surface.PolyMesh
        a collection of "left" and "right" InMemoryMesh objects
    """
    giftis = fetch_atlas(atlas, density)[mesh]
    return PolyMesh(left=giftis.L, right=giftis.R)
