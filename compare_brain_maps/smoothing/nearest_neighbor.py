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
            Surface mesh input containing 'coordinates'

        Returns
        -------
        SurfaceImage
            Smoothed surface data, centered and scaled to match the input distribution.
        """
        X_smoothed = copy.deepcopy(X)
        A = X_smoothed.get_adjacency()

        for hemi in ["left", "right"]:
            for _ in range(self.n_iterations):
                X_smoothed.data.parts[hemi] = X_smoothed.data.parts[hemi] @ A[hemi]

            # center and scale the smoothed data to match the input distribution
            X_smoothed.data.parts[hemi] = (
                X_smoothed.data.parts[hemi] - X_smoothed.data.parts[hemi].mean()
            ) / X_smoothed.data.parts[hemi].std() * X.data.parts[hemi].std() + X.data.parts[hemi].mean()

        return X_smoothed
