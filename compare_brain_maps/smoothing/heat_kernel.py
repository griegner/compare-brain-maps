import copy

import numpy as np
from lapy import Solver, TriaMesh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state


class HeatKernelSmoother(TransformerMixin, BaseEstimator):
    """Heat kernel smoothing on a surface mesh using Laplace-Beltrami eigenvalues and eigenmodes.

    Parameters
    ----------
    sigma : float, optional
        smoothing parameter controlling the width of the heat kernel, by default 1.0
    n_modes : int, optional
        number of eigenmodes, by default 500
    reuse_eigenpairs : bool, optional
        reuse eigenpairs of mesh across sequential calls to `transform`, by default True
    seed : None, int or instance of RandomState, optional
        seed for the random state generator, by default None

    References
    ----------
    .. [1] Seo, Chung, Vorperian, Jiang, Navab, Pluim, Viergever.  Heat Kernel Smoothing Using
        Laplace-Beltrami Eigenfunctions. Medical Image Computing and Computer-Assisted Intervention (2010)
    """

    def __init__(self, sigma=1.0, n_modes=500, reuse_eigenpairs=True, seed=None):
        self.sigma = sigma
        self.n_modes = n_modes
        self.reuse_eigenpairs = reuse_eigenpairs
        self.seed = seed

    def fit(self, X):
        """For scikit-learn compatibility."""
        return self

    def transform(self, X):
        """Apply heat kernel smoothing to `Surface.data`.

        Parameters
        ----------
        X : Surface
            surface object containing the data and mesh for both hemispheres

        Returns
        -------
        Surface
            surface object with data smoothed on the surface mesh
        """
        X_smoothed = copy.deepcopy(X)

        # compute laplace-beltrami eigenpairs, or load precomputed
        if self.reuse_eigenpairs and hasattr(self, "_eigenpairs"):
            eigenpairs = self._eigenpairs
        else:
            eigenpairs = {}
            for hemi in ["left", "right"]:
                mesh = X_smoothed.mesh.parts[hemi]
                tria_mesh = TriaMesh(v=mesh.coordinates, t=mesh.faces)
                fem = Solver(geometry=tria_mesh, use_cholmod=True)  # requires scikit-sparse
                rng = check_random_state(self.seed)
                evals, emodes = fem.eigs(k=self.n_modes + 1, rng=rng)
                evals, emodes = evals[1:], emodes[:, 1:]  # first is constant
                eigenpairs[hemi] = (evals, emodes)
            if self.reuse_eigenpairs:
                self._eigenpairs = eigenpairs

        # apply smoothing using eigenpairs
        for hemi in ["left", "right"]:
            evals, emodes = eigenpairs[hemi]
            beta_ = np.linalg.solve((emodes.T @ emodes), emodes.T @ X_smoothed.data.parts[hemi])
            X_smoothed.data.parts[hemi] = np.sum(np.exp(-evals * self.sigma) * beta_ * emodes, axis=1)

        return X_smoothed
