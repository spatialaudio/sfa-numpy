import numpy as np
from scipy import linalg
from scipy.special import eval_legendre as legendre


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


def matdiagmul(A, b):
    """Efficient multiplication of  matrix and diagonal matrix .

    Returns the multiplication of a matrix *A* and a diagonal matrix. The
    diagonal matrix is given by the vector *b* containing its elements on
    the main diagonal.  If *b* is a matrix, it is treated as a stack of vectors
    residing in the last index and broadcast accordingly.

    Parameters
    ----------
    A : array_like
        Input matrix.
    b : array_like
        Main diagonal elements or stack of main diagonal elements.

    Returns
    -------
    array_like
        Result of matrix multiplication.
    """
    if len(b.shape) == 1:
        b = b[np.newaxis, :]
    K, N = b.shape
    M, N = A.shape

    C = np.zeros([K, M, N], dtype=A.dtype)
    for k in range(K):
        C[k, :, :] = A * b[k, :]
    return C


def db(x, *, power=False):
    """Convert *x* to decibel.

    Parameters
    ----------
    x : array_like
        Input data.  Values of 0 lead to negative infinity.
    power : bool, optional
        If ``power=False`` (the default), *x* is squared before
        conversion.

    """
    with np.errstate(divide='ignore'):
        return (10 if power else 20) * np.log10(np.abs(x))


def double_factorial(n):
    """Double factorial"""
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        return n * double_factorial(n - 2)


def maxre_sph(N):
    """max-reE modal weight for spherical harmonics expansion.

    Parameter
    ---------
    N : int
        Highest spherical harmonic order (Ambisonic order).

    """
    theta = np.deg2rad(137.9 / (N + 1.52))
    return legendre(np.arange(N + 1), np.cos(theta))


def point_spread(N, phi, modal_weight=maxre_sph, equalization='omni'):
    """Directional response of a given modal weight function and
    equalization scheme.

    Parameters
    ----------
    N : int
        Highest spherical harmonic order (Ambisonic order).
    phi : array_like
        Angular distance from the main axis in radian.
    modal_weight : callable, optional
        Modal weighting function.
    equalization : {'omni', 'diffuse', 'free'}, optional
        Equalization scheme.

    """
    a = modal_weight(N)
    if equalization == 'omni':
        pass
    elif equalization == 'diffuse':
        a *= 1 / modal_norm(a, ord=2)
    elif equalization == 'free':
        a *= 1 / modal_norm(a, ord=1)
    return np.stack([(2*n+1) * a[n] * legendre(n, np.cos(phi))
                     for n in range(N+1)])


def modal_norm(a, ord=2):
    """Norm of the coefficients in the spherical harmonics domain.
    """
    num_degree = 2 * np.arange(a.shape[-1]) + 1
    return np.sum(num_degree * a**ord, axis=-1)**(1/ord)
