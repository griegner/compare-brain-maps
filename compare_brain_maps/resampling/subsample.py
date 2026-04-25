import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from ._base import quantile_test


class SubsampleResampler(BaseEstimator):
    """Resampling by subsampling patches of adjacent vertices.

    Parameters
    ----------
    n_subsamples : int, optional
        number of subsamples, by default 1000
    patch_size : int, optional
        number of iterations to grow the patch, by default 4
    alpha : float, optional
        significance level for the quantile test, by default 0.05
    reuse_patches : bool, optional
        reuse patches across sequantial calls to `fit`, by default True
    seed : None, int or instance of RandomState, optional
        seed for the random state generator, by default None

    Attributes
    ----------
    param_ : float
        map-to-map correlation across both hemispheres combined
    params_ : ndarray of shape (n_subsamples,)
        patch-to-patch correlations across both hemispheres combined
    null_ : bool
        True if `param_` falls inside the central quantile interval
    """

    def __init__(self, n_subsamples=1000, patch_size=4, alpha=0.05, reuse_patches=True, seed=None):
        super().__init__()
        self.n_subsamples = n_subsamples
        self.patch_size = patch_size
        self.alpha = alpha
        self.reuse_patches = reuse_patches
        self.seed = seed

    def _corr(self, X_data, Y_data):
        xm = X_data - X_data.mean()
        ym = Y_data - Y_data.mean()
        den = np.linalg.norm(xm) * np.linalg.norm(ym)
        return (xm @ ym) / den

    def _subsample_vertices(self, A, rng):
        n_vertices = A.shape[0]
        patches = np.zeros((n_vertices, self.n_subsamples), dtype=bool)
        patches[rng.randint(0, n_vertices, size=self.n_subsamples), np.arange(self.n_subsamples)] = True
        for _ in range(self.patch_size):
            patches = (A @ patches) > 0
        return patches

    def fit(self, X, Y):
        """Fit SubsampleResampler.

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
        n = X_n.shape[0]

        # full-map correlation across hemispheres
        rho_n_ = self._corr(X_n, Y_n)

        # subsample patch correlations across hemispheres
        if self.reuse_patches and hasattr(self, "_patches") and self._patches["left"].shape[1] == self.n_subsamples:
            patches = self._patches
        else:
            A = X.get_adjacency()  # X = Y
            rng = check_random_state(self.seed)
            patches = {
                "left": self._subsample_vertices(A["left"], rng),
                "right": self._subsample_vertices(A["right"], rng),
            }
            if self.reuse_patches:
                self._patches = patches

        params_ = np.zeros(self.n_subsamples)
        for subsample in range(self.n_subsamples):
            patch_left = patches["left"][:, subsample]
            patch_right = patches["right"][:, subsample]
            patch = np.concatenate([patch_left, patch_right]).astype(bool)
            m = patch.sum()
            rho_m_ = self._corr(X_n[patch], Y_n[patch])
            params_[subsample] = np.sqrt(m / n) * (rho_m_ - rho_n_)

        self.param_ = rho_n_
        self.params_ = params_
        self.null_ = quantile_test(self.param_, self.params_, alpha=self.alpha)

        return self
