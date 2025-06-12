import numpy as np
from lapy import Solver, TriaMesh
from nilearn.surface import load_surf_mesh
from scipy.sparse import csr_matrix
from sklearn.utils import check_random_state


def adjacency_matrix(surface):
    """Sparse adjacency matrix of a surface mesh.

    Parameters
    ----------
    surface : str or Surface
        Surface mesh input that contains 'coordinates' (vertices) and 'faces' (triangles).

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n_vertices, n_vertices)
        Sparse adjacency matrix where entry (i, j) is 1 if vertices i and j are connected by an edge, 0 otherwise.
    """
    surface = load_surf_mesh(surface)
    n_vertices = surface.coordinates.shape[0]
    edges = np.vstack([surface.faces[:, [0, 1]], surface.faces[:, [0, 2]], surface.faces[:, [1, 2]]]).astype(np.int64)
    bigcol = edges[:, 0] > edges[:, 1]
    edges = np.concatenate(
        [
            edges[bigcol, 0] + edges[bigcol, 1] * n_vertices,
            edges[~bigcol, 1] + edges[~bigcol, 0] * n_vertices,
        ]
    )
    edges = np.unique(edges)
    u, v = edges // n_vertices, edges % n_vertices
    ones = np.ones_like(edges)
    ee = np.concatenate([ones, ones])
    uv = np.concatenate([u, v])
    vu = np.concatenate([v, u])
    return csr_matrix((ee, (uv, vu)), shape=(n_vertices, n_vertices))


def iterative_smoothing(surface, n_iterations=1, seed=0):
    """Iterative smoothing on a surface mesh averaging over nearest neighbors.

    Parameters
    ----------
    surface : str or Surface
        Surface mesh input that contains 'coordinates' (vertices) and 'faces' (triangles).
    n_iterations : int, optional
        Number of smoothing iterations, by default 1.
    seed : None, int, or instance of RandomState, optional
        Random seed, by default 0

    Returns
    -------
    1darray of shape (n_vertices,)
        Smoothed random field on the surface mesh.
    """
    rng = check_random_state(seed)
    surface = load_surf_mesh(surface)
    n_vertices = surface.coordinates.shape[0]
    x = rng.standard_normal(size=n_vertices)
    A = adjacency_matrix(surface)
    # normalize adjacency matrix rows (mean of neighbors)
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # avoid division by zero
    A = A.multiply(1 / row_sums[:, None])
    input_mean = np.nanmean(x, axis=0)
    input_std = np.nanstd(x, axis=0)
    for _ in range(n_iterations):
        x = A.dot(x)
    # match mean and variance
    output_mean = np.nanmean(x, axis=0)
    output_std = np.nanstd(x, axis=0)
    x_smooth = (x - output_mean) * (input_std / (output_std + 1e-10)) + input_mean
    return x_smooth


def heat_kernel_smoothing(surface, sigma=1.0, n_modes=1000, seed=0):
    """Heat kernel smoothing on a surface mesh using Laplace-Beltrami eigenvalues and eigenmodes.

    Parameters
    ----------
    surface : str or Surface
        Surface mesh input that contains 'coordinates' (vertices) and 'faces' (triangles).
    sigma : float, optional
        Smoothing parameter controlling the width of the heat kernel, by default 1.0
    n_modes : int, optional
        Number of Laplace-Beltrami eigenmodes, by default 1000
    seed : None, int, or instance of RandomState, optional
        Random seed, by default 0

    Returns
    -------
    1darray of shape (n_vertices,)
        Smoothed random field on the surface mesh.

    References
    ----------
    .. [1] Seo, Chung, Vorperian, Jiang, Navab, Pluim, Viergever.  Heat Kernel Smoothing Using
        Laplace-Beltrami Eigenfunctions. Medical Image Computing and Computer-Assisted Intervention (2010)
    """
    rng = check_random_state(seed)
    surface = load_surf_mesh(surface)
    z = rng.standard_normal(size=n_modes)
    mesh = TriaMesh(v=surface.coordinates, t=surface.faces)
    fem = Solver(mesh, lump=True, use_cholmod=True)
    evals, emodes = fem.eigs(k=n_modes + 1)
    evals, emodes = evals[1:], emodes[:, 1:]  # first in non-constant
    return np.sum(np.exp(-evals * sigma) * z * emodes, axis=1)
