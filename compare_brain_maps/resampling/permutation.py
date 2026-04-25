import numpy as np
from neuromaps.datasets import fetch_atlas
from neuromaps.nulls.spins import get_parcel_centroids
from scipy import spatial
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_random_state

from ._base import quantile_test


class PermutationResampler(BaseEstimator):
    """Resampling by permuting vertices through rotations of their spherical projections ("Spin Test").

    Parameters
    ----------
    n_permutations : int, optional
        number of permutations, by default 1000
    alpha : float, optional
        significance level for the quantile test, by default 0.05
    reuse_spins : bool, optional
        reuse spins across sequential calls to `fit`, by default True
    n_jobs : int, optional
        number of workers used by KDTree queries (`-1` uses all cores), by default -1
    seed : None, int or instance of RandomState, optional
        seed for the random state generator, by default None

    Attributes
    ----------
    param_ : float
        map-to-map correlation across both hemispheres combined
    params_ : ndarray of shape (n_permutations,)
        rotated map-to-map correlations across both hemispheres combined
    null_ : bool
        True if `param_` falls inside the central quantile interval

    References
    ----------
    .. [2] Alexander-Bloch, Shou, Liu, Satterthwaite, Glahn, Shinohara, Vandekar, Raznahan. On testing for
       spatial correspondence between maps of human brain structure and function.  NeuroImage (2018)
    """

    def __init__(self, n_permutations=1000, alpha=0.05, reuse_spins=True, seed=None, n_jobs=-1):
        super().__init__()
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.reuse_spins = reuse_spins
        self.seed = seed
        self.n_jobs = n_jobs

    def _corr(self, X_data, Y_data):
        xm = X_data - X_data.mean()
        ym = Y_data - Y_data.mean()
        den = np.linalg.norm(xm) * np.linalg.norm(ym)
        return (xm @ ym) / den

    def _gen_spinsamples(self, coords, hemiid):
        """rewrite of `neuromaps.nulls.spins.gen_spinsamples` with parellel processing"""

        if coords.shape[-1] != 3 or coords.squeeze().ndim != 2 or hemiid.ndim != 1 or len(coords) != len(hemiid):
            raise ValueError("Expected coords shape (N, 3) and hemiid shape (N,)")

        rng = check_random_state(self.seed)
        reflect = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        inds = np.arange(len(coords), dtype=np.int32)
        masks = [hemiid == 0, hemiid == 1]
        hemi_coords = [coords[m] for m in masks]
        hemi_inds = [inds[m] for m in masks]
        spinsamples = np.empty((len(coords), self.n_permutations), dtype=np.int32)

        for n in range(self.n_permutations):
            rot_l, temp = np.linalg.qr(rng.normal(size=(3, 3)))
            rot_l = rot_l @ np.diag(np.sign(np.diag(temp)))
            if np.linalg.det(rot_l) < 0:
                rot_l[:, 0] = -rot_l[:, 0]

            resampled = np.empty(len(coords), dtype=np.int32)
            for h, rot in enumerate((rot_l, reflect @ rot_l @ reflect)):
                coor = hemi_coords[h]
                if len(coor) == 0:
                    continue
                _, col = spatial.cKDTree(coor @ rot, balanced_tree=False).query(coor, k=1, workers=self.n_jobs)
                resampled[masks[h]] = hemi_inds[h][col]

            spinsamples[:, n] = resampled

        return spinsamples

    def _spin_vertices(self, X):
        n_vertices = X.shape[0] // 2
        spheres = fetch_atlas(X.atlas, X.density)["sphere"]
        coords, hemi = get_parcel_centroids(spheres, parcellation=None, method="surface")
        spins_arr = self._gen_spinsamples(coords, hemi)
        return {"left": spins_arr[:n_vertices], "right": spins_arr[n_vertices:] - n_vertices}

    def fit(self, X, Y):
        """Fit PermutationResampler.

        Parameters
        ----------
        X : Surface
            surface object containing the data and mesh for both hemispheres
        Y : Surface
            surface object containing the data and mesh for both hemispheres
        """
        # check X, Y are same surface
        if X.shape != Y.shape:
            raise ValueError(f"X and Y must have the same shape. X: {X.shape}, Y: {Y.shape}")

        # combine hemispheres
        X_n = np.concatenate([X.data.parts["left"], X.data.parts["right"]])
        Y_n = np.concatenate([Y.data.parts["left"], Y.data.parts["right"]])

        # full-map correlation across hemispheres
        rho_n_ = self._corr(X_n, Y_n)

        # permute patch correlation across hemispheres
        if self.reuse_spins and hasattr(self, "_spins") and self._spins["left"].shape[1] == self.n_permutations:
            spins = self._spins
        else:
            spins = self._spin_vertices(X)
            if self.reuse_spins:
                self._spins = spins

        # permute patch correlation across hemispheres
        params_ = np.zeros(self.n_permutations)
        n_vertices = X.shape[0] // 2
        for permutation in range(self.n_permutations):
            idx = np.concatenate([spins["left"][:, permutation], spins["right"][:, permutation] + n_vertices])
            params_[permutation] = self._corr(X_n[idx], Y_n)

        self.is_fitted_ = True
        self.param_ = rho_n_
        self.params_ = params_
        self.null_ = quantile_test(self.param_, self.params_, alpha=self.alpha)
        return self
