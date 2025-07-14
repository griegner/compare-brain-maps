"""utilities for surface maps"""

from neuromaps.datasets import fetch_atlas
from nilearn.surface import SurfaceImage


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
