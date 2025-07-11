import numpy as np
from scipy.stats import norm
from sklearn.utils import check_random_state


def xcorr(x1, x2):
    """Estimate the cross-correlation between two zero-centered maps.

    Parameters
    ----------
    x1 : 1darray of shape (n_vertices,)
        First surface map (should be zero centered).
    x2 : 1darray of shape (n_vertices,)
        Second surface map (should be zero centered).

    Returns
    -------
    float
        Corr(x1, x2)
    """
    return x1 @ x2 / (x1.std() * x2.std()) / x1.size


def quantile_test(xcorr, xcorrs, alpha=0.05):
    """Test if a correlation value is outside the (alpha/2, 1-alpha/2) quantile interval.

    Parameters
    ----------
    xcorr : float
        Estimated correlation.
    xcorrs : 1darray of shape (n_subsamples,)
        Null distribution of estimated correlations.
    alpha : float, optional
        Significance level, by default 0.05.

    Returns
    -------
    bool
        True if xcorr is outside the quantile interval, False otherwise.
    """
    lwr, upr = np.quantile(xcorrs, [alpha / 2, 1 - alpha / 2])
    return xcorr < lwr or xcorr > upr


def subsample_vertices(A, n_iterations=1, seed=0):
    """Subsample a patch of neighboring vertices from an adjacency matrix.

    Parameters
    ----------
    A : ndarray of shape (n_vertices, n_vertices)
        Adjacency matrix.
    n_iterations : int, optional
        Number of iterations to grow the patch, by default 1
    seed : int, optional
        Random seed, by default 0

    Returns
    -------
    1darray of shape (n_vertices)
        Boolean mask of the surface patch.
    """
    rng = check_random_state(seed)
    n_vertices = A.shape[0]
    patch = np.zeros(n_vertices)
    patch[rng.integers(0, n_vertices)] = 1
    for _ in range(n_iterations):
        patch = (A @ patch) > 0
    return patch


def subsample_distribution(x1, x2, A, n_subsamples=1000, n_iterations=4, seed=0):
    """Compute the null distribution of correlations by subsampling patches.

    Parameters
    ----------
    x1 : 1darray of shape (n_vertices,)
        First surface map (should be zero centered).
    x2 : 1darray of shape (n_vertices,)
        Second surface map (should be zero centered).
    A : ndarray of shape (n_vertices, n_vertices)
        Adjacency matrix.
    n_subsamples : int, optional
        Number of subsamples to draw, by default 1000.
    n_iterations : int, optional
        Number of iterations to grow the patch, by default 4.
    seed : int, optional
        Random seed, by default 0

    Returns
    -------
    1darray of shape (n_subsamples,)
        Estimated cross-correlations from subsampled patches.
    """
    n_vertices = A.shape[0]
    xcorrs = np.zeros(n_subsamples)
    for subsample in range(n_subsamples):
        patch = subsample_vertices(A, n_iterations, seed=(seed + subsample))
        scale_std = np.sqrt(patch.sum() / n_vertices)
        xcorrs[subsample] = xcorr(x1[patch], x2[patch]) * scale_std
    return xcorrs


def bernstd(p, n_samples, alpha=0.05):
    """
    Bernoulli confidence intervals by the CLT.

    Parameters
    ----------
    p : np.ndarray or float
        Probability or array of probabilities.
    n_samples : int
        Number of samples.
    alpha : float, optional
        Significance level, by default 0.05.

    Returns
    -------
    ndarray of shape (2 x len(p))
        Confidence intervals.

    References
    ----------
    .. [1] Davenport. RFTtoolbox: A toolbox designed for generation and analysis
        of random fields both continuously sampled and on a lattice. (2025)
    """
    p = np.atleast_1d(p)
    z = norm.ppf(1 - alpha / 2)
    std_error = np.sqrt(p * (1 - p)) * z / np.sqrt(n_samples)
    return np.vstack((p - std_error, p + std_error))
