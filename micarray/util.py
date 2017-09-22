import numpy as np
from scipy import linalg


def norm_of_columns(A, p=2):
    """Vector p-norm of each column of a matrix.

    Parameters
    ----------
    A : array_like
        Input matrix.
    p : int, optional
        p-th norm.

    Returns
    -------
    array_like
        p-norm of each column of A.
    """
    _, N = A.shape
    return np.asarray([linalg.norm(A[:, j], ord=p) for j in range(N)])


def coherence_of_columns(A):
    """Mutual coherence of columns of A.

    Parameters
    ----------
    A : array_like
        Input matrix.
    p : int, optional
        p-th norm.

    Returns
    -------
    array_like
        Mutual coherence of columns of A.
    """
    A = np.asmatrix(A)
    _, N = A.shape
    A = A * np.asmatrix(np.diag(1/norm_of_columns(A)))
    Gram_A = A.H*A
    for j in range(N):
        Gram_A[j, j] = 0
    return np.max(np.abs(Gram_A))


def asarray_1d(a, **kwargs):
    """Squeeze the input and check if the result is one-dimensional.

    Returns *a* converted to a `numpy.ndarray` and stripped of
    all singleton dimensions.  Scalars are "upgraded" to 1D arrays.
    The result must have exactly one dimension.
    If not, an error is raised.

    """
    result = np.squeeze(np.asarray(a, **kwargs))
    if result.ndim == 0:
        result = result.reshape((1,))
    elif result.ndim > 1:
        raise ValueError("array must be one-dimensional")
    return result
