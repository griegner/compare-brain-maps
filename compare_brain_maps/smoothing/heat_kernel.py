import copy

import numpy as np
from lapy import Solver, TriaMesh
from sklearn.base import BaseEstimator, TransformerMixin


class HeatKernelSmoother(TransformerMixin, BaseEstimator):
    """Heat kernel smoothing on a surface mesh using Laplace-Beltrami eigenvalues and eigenmodes.

    Parameters
    ----------
    sigma : float, optional
        smoothing parameter controlling the width of the heat kernel, by default 1.0
    n_modes : int, optional
        number of eigenmodes, by default 500
    """

    def __init__(self, sigma=1.0, n_modes=500):
        self.sigma = sigma
        self.n_modes = n_modes

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

        for hemi in ["left", "right"]:
            mesh = X_smoothed.mesh.parts[hemi]
            tria_mesh = TriaMesh(v=mesh.coordinates, t=mesh.faces)
            fem = Solver(geometry=tria_mesh, use_cholmod=True)  # require scikit-sparse
            evals, emodes = fem.eigs(k=self.n_modes + 1)
            evals, emodes = evals[1:], emodes[:, 1:]  # first is non-constant
            beta_ = np.linalg.solve((emodes.T @ emodes), emodes.T @ X_smoothed.data.parts[hemi])
            X_smoothed.data.parts[hemi] = np.sum(np.exp(-evals * self.sigma) * beta_ * emodes, axis=1)
        return X_smoothed
