import copy

from sklearn.base import BaseEstimator, TransformerMixin


class NearestNeighborSmoother(TransformerMixin, BaseEstimator):
    """Iterative smoothing on a surface mesh by averaging over nearest neighbor vertices.

    Parameters
    ----------
    n_iterations : int, optional
        number of smoothing iterations, by default 1
    """

    def __init__(self, n_iterations=1):
        self.n_iterations = n_iterations

    def fit(self, X):
        """For scikit-learn compatibility."""
        return self

    def transform(self, X):
        """Apply nearest neighbor smoothing to `Surface.data`.

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
        A = X_smoothed.get_adjacency()

        for hemi in ["left", "right"]:
            # normalize adjacency matrix
            A[hemi] = A[hemi] / A[hemi].sum(axis=0)
            for _ in range(self.n_iterations):
                X_smoothed.data.parts[hemi] = X_smoothed.data.parts[hemi] @ A[hemi]

        return X_smoothed
