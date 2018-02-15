from __future__ import division
import numpy as np
from scipy import special
from .. import util
from functools import wraps
from warnings import warn


def replace_zeros_of_radial_function_decorator(f):
    """Apply replace_zeros_of_radial_function() to output of function f.

    Also add argument flag 'replace_zeros' to f.

    CAVEAT:
    replace_zeros_of_radial_function() needs wavenumbers k or kr.
    These are taken from argument list of f,  either by key or using the 2nd
    positional argument !
    """
    @wraps(f)
    def wrapper(*args, replace_zeros=True, **kwargs):
        if 'kr' in kwargs:
            kr = kwargs['kr']
        elif 'k' in kwargs:
            # The exact values in kr do not matter, only order is important.
            # So it's okay to use k instead.
            kr = kwargs['k']
        else:
            kr = args[1]  # ATTENTION: hinges on positional argument order!
        if replace_zeros:
            return replace_zeros_of_radial_function(f(*args, **kwargs), kr)
        else:
            return f(*args, **kwargs)
    return wrapper


@replace_zeros_of_radial_function_decorator
def spherical_pw(N, k, r, setup):
    r"""Radial coefficients for a plane wave.

    Computes the radial component of the spherical harmonics expansion of a
    plane wave impinging on a spherical array.

    .. math::

        \mathring{P}_n(k) = 4 \pi i^n b_n(kr)

    Parameters
    ----------
    N : int
        Maximum order.
    k : (M,) array_like
        Wavenumber.
    r : float
        Radius of microphone array.
    setup : {'open', 'card', 'rigid'}
        Array configuration (open, cardioids, rigid).

    Returns
    -------
    bn : (M, N+1) numpy.ndarray
        Radial weights for all orders up to N and the given wavenumbers.
    """
    kr = util.asarray_1d(k*r)
    n = np.arange(N+1)

    bn = weights(N, kr, setup)
    return 4*np.pi * (1j)**n * bn


@replace_zeros_of_radial_function_decorator
def spherical_ps(N, k, r, rs, setup):
    r"""Radial coefficients for a point source.

    Computes the radial component of the spherical harmonics expansion of a
    point source impinging on a spherical array.

    .. math::

        \mathring{P}_n(k) = 4 \pi (-i) k h_n^{(2)}(k r_s) b_n(kr)

    Parameters
    ----------
    N : int
        Maximum order.
    k : (M,) array_like
        Wavenumber.
    r : float
        Radius of microphone array.
    rs : float
        Distance of source.
    setup : {'open', 'card', 'rigid'}
        Array configuration (open, cardioids, rigid).

    Returns
    -------
    bn : (M, N+1) numpy.ndarray
        Radial weights for all orders up to N and the given wavenumbers.
    """
    k = util.asarray_1d(k)
    krs = k*rs
    n = np.arange(N+1)

    bn = weights(N, k*r, setup)
    if len(k) == 1:
        bn = bn[np.newaxis, :]

    for i, x in enumerate(krs):
        hn = special.spherical_jn(n, x) - 1j * special.spherical_yn(n, x)
        bn[i, :] = bn[i, :] * 4*np.pi * (-1j) * hn * k[i]

    return np.squeeze(bn)


def weights(N, kr, setup):
    r"""Radial weighing functions.

    Computes the radial weighting functions for diferent array types
    (cf. eq.(2.62), Rafaely 2015).

    For instance for an rigid array

    .. math::

        b_n(kr) = j_n(kr) - \frac{j_n^\prime(kr)}{h_n^{(2)\prime}(kr)}h_n^{(2)}(kr)

    Parameters
    ----------
    N : int
        Maximum order.
    kr : (M,) array_like
        Wavenumber * radius.
    setup : {'open', 'card', 'rigid'}
        Array configuration (open, cardioids, rigid).

    Returns
    -------
    bn : (M, N+1) numpy.ndarray
        Radial weights for all orders up to N and the given wavenumbers.

    """
    kr = util.asarray_1d(kr)
    n = np.arange(N+1)
    bns = np.zeros((len(kr), N+1), dtype=complex)
    for i, x in enumerate(kr):
        jn = special.spherical_jn(n, x)
        if setup == 'open':
            bn = jn
        elif setup == 'card':
            bn = jn - 1j * special.spherical_jn(n, x, derivative=True)
        elif setup == 'rigid':
            if x == 0:
                # hn(x)/hn'(x) -> 0 for x -> 0
                bn = jn
            else:
                jnd = special.spherical_jn(n, x, derivative=True)
                hn = jn - 1j * special.spherical_yn(n, x)
                hnd = jnd - 1j * special.spherical_yn(n, x, derivative=True)
                bn = jn - jnd/hnd*hn
        else:
            raise ValueError('setup must be either: open, card or rigid')
        bns[i, :] = bn
    return np.squeeze(bns)


def replace_zeros_of_radial_function(A, kr):
    """
    Replace zero entries A[i, j] == 0 with A[l, j] != 0.

    where kr[l] is (the wavenumber) nearest to kr[i].

    (This function may be used to fix "forbidden frequencies" in radial
    filters before inversion.)

    Parameters
    ----------
    A : (K, N) ndarray
    kr : (K,) array_like

    Returns
    -------
    (K, N) ndarray

    """
    kr = util.asarray_1d(kr)

    if len(A.shape) == 1 and A.shape[0] == len(kr):
        # single column (mode) is fine.
        A = A[:, np.newaxis]
    elif len(A.shape) == 1 and len(kr) == 1:
        # single wavenumber is also fine.
        A = A[np.newaxis, :]
    if A.shape[0] != len(kr):
        raise ValueError("A and kr must have same len > 1,"
                         " but have {} and {}".format(A.shape[0], len(kr)))

    kr, idx, inv_idx = np.unique(kr, True, True)
    A = A[idx, :]
    zeros = np.abs(A) < 1e-300

    for i, j in zip(*np.where(zeros)):
        # for each zero value...
        kr_tmp = kr.astype(float)
        l = i
        while zeros[l, j]:
            # ...try to find replacement value
            kr_tmp[l] = np.inf
            l = np.argmin(np.abs(kr_tmp - kr[i]))
            if np.isinf(kr_tmp[l]):
                raise ValueError("Could not replace zero value in A.")
        A[i, j] = A[l, j]

    A = A[inv_idx, :]
    return np.squeeze(A)


def regularize(dn, a0, method):
    """Regularization (amplitude limitation) of radial filters.

    Amplitude limitation of radial filter coefficients, methods according
    to (cf. Rettberg, Spors : DAGA 2014)

    Parameters
    ----------
    dn : numpy.ndarray
        Values to be regularized
    a0 : float
        Parameter for regularization (not required for all methods)
    method : {'none', 'discard', 'softclip', 'Tikh', 'wng'}
        Method used for regularization/amplitude limitation
        (none, discard, hardclip, Tikhonov, White Noise Gain).

    Returns
    -------
    dn : numpy.ndarray
        Regularized values.
    hn : array_like

    """

    idx = np.abs(dn) > a0

    if method == 'none':
        hn = np.ones_like(dn)
    elif method == 'discard':
        hn = np.ones_like(dn)
        hn[idx] = 0
    elif method == 'hardclip':
        hn = np.ones_like(dn)
        hn[idx] = a0 / np.abs(dn[idx])
    elif method == 'softclip':
        scaling = np.pi / 2
        hn = a0 / abs(dn)
        hn = 2 / np.pi * np.arctan(scaling * hn)
    elif method == 'Tikh':
        a0 = np.sqrt(a0 / 2)
        alpha = (1 - np.sqrt(1 - 1/(a0**2))) / (1 + np.sqrt(1 - 1/(a0**2)))
        hn = 1 / (1 + alpha**2 * np.abs(dn)**2)
#        hn = 1 / (1 + alpha**2 * np.abs(dn))
    elif method == 'wng':
        hn = 1/(np.abs(dn)**2)
#        hn = hn/np.max(hn)
    else:
        raise ValueError('method must be either: none, ' +
                         'discard, hardclip, softclip, Tikh or wng')
    dn = dn * hn
    return dn, hn


def diagonal_mode_mat(bk):
    """Diagonal matrix of radial coefficients for all modes/wavenumbers.

    Parameters
    ----------
    bk : (M, N+1) numpy.ndarray
        Vector containing values for all wavenumbers :math:`M` and modes up to
        order :math:`N`

    Returns
    -------
    Bk : (M, (N+1)**2, (N+1)**2) numpy.ndarray
        Multidimensional array containing diagnonal matrices with input
        vector on main diagonal.
    """
    bk = repeat_n_m(bk)
    if len(bk.shape) == 1:
        bk = bk[np.newaxis, :]
    K, N = bk.shape
    Bk = np.zeros([K, N, N], dtype=complex)
    for k in range(K):
        Bk[k, :, :] = np.diag(bk[k, :])
    return np.squeeze(Bk)


def repeat_n_m(v):
    """Repeat elements in a vector .

    Returns a vector with the elements of the vector *v* repeated *n* times,
    where *n* denotes the position of the element in *v*. The function can
    be used to order the coefficients in the vector according to the order of
    spherical harmonics. If *v* is a matrix, it is treated as a stack of
    vectors residing in the last index and broadcast accordingly.

    Parameters
    ----------
    v : (,N+1) numpy.ndarray
        Input vector or stack of input vectors.

    Returns
    -------
     : (,(N+1)**2) numpy.ndarray
        Vector or stack of vectors containing repetated values.
    """
    krlist = [np.tile(v, (2*i+1, 1)).T for i, v in enumerate(v.T.tolist())]
    return np.squeeze(np.concatenate(krlist, axis=-1))


@replace_zeros_of_radial_function_decorator
def circular_pw(N, k, r, setup):
    r"""Radial coefficients for a plane wave.

    Computes the radial component of the circular harmonics expansion of a
    plane wave impinging on a circular array.

    .. math::

        \mathring{P}_n(k) = i^n b_n(kr)

    Parameters
    ----------
    N : int
        Maximum order.
    k : (M,) array_like
        Wavenumber.
    r : float
        Radius of microphone array.
    setup : {'open', 'card', 'rigid'}
        Array configuration (open, cardioids, rigid).

    Returns
    -------
    bn : (M, 2*N+1) numpy.ndarray
        Radial weights for all orders up to N and the given wavenumbers.
    """
    kr = util.asarray_1d(k*r)
    n = np.roll(np.arange(-N, N+1), -N)

    bn = circ_radial_weights(N, kr, setup)
    return 1j**n * bn


@replace_zeros_of_radial_function_decorator
def circular_ls(N, k, r, rs, setup):
    r"""Radial coefficients for a line source.

    Computes the radial component of the circular harmonics expansion of a
    line source impinging on a circular array.

    .. math::

        \mathring{P}_n(k) = \frac{-i}{4} H_n^{(2)}(k r_s) b_n(kr)

    Parameters
    ----------
    N : int
        Maximum order.
    k : (M,) array_like
        Wavenumber.
    r : float
        Radius of microphone array.
    rs : float
        Distance of source.
    setup : {'open', 'card', 'rigid'}
        Array configuration (open, cardioids, rigid).

    Returns
    -------
    bn : (M, 2*N+1) numpy.ndarray
        Radial weights for all orders up to N and the given wavenumbers.
    """
    k = util.asarray_1d(k)
    krs = k*rs
    n = np.roll(np.arange(-N, N+1), -N)

    bn = circ_radial_weights(N, k*r, setup)
    if len(k) == 1:
        bn = bn[np.newaxis, :]
    for i, x in enumerate(krs):
        Hn = special.hankel2(n, x)
        bn[i, :] = bn[i, :] * Hn
    return -1j/4 * np.squeeze(bn)


def circ_radial_weights(N, kr, setup):
    r"""Radial weighing functions.

    Computes the radial weighting functions for diferent array types

    For instance for an rigid array

    .. math::

        b_n(kr) = J_n(kr) - \frac{J_n^\prime(kr)}{H_n^{(2)\prime}(kr)}H_n^{(2)}(kr)

    Parameters
    ----------
    N : int
        Maximum order.
    kr : (M,) array_like
        Wavenumber * radius.
    setup : {'open', 'card', 'rigid'}
        Array configuration (open, cardioids, rigid).

    Returns
    -------
    bn : (M, 2*N+1) numpy.ndarray
        Radial weights for all orders up to N and the given wavenumbers.

    """
    kr = util.asarray_1d(kr)
    n = np.arange(N+1)
    Bns = np.zeros((len(kr), N+1), dtype=complex)
    for i, x in enumerate(kr):
        Jn = special.jv(n, x)
        if setup == 'open':
            bn = Jn
        elif setup == 'card':
            bn = Jn - 1j * special.jvp(n, x, n=1)
        elif setup == 'rigid':
            if x == 0:
                # Hn(x)/Hn'(x) -> 0 for x -> 0
                bn = Jn
            else:
                Jnd = special.jvp(n, x, n=1)
                Hn = special.hankel2(n, x)
                Hnd = special.h2vp(n, x)
                bn = Jn - Jnd/Hnd*Hn
        else:
            raise ValueError('setup must be either: open, card or rigid')
        Bns[i, :] = bn
    Bns = np.concatenate((Bns, (Bns*(-1)**np.arange(N+1))[:, :0:-1]), axis=-1)
    return np.squeeze(Bns)


def circ_diagonal_mode_mat(bk):
    """Diagonal matrix of radial coefficients for all modes/wavenumbers.

    Parameters
    ----------
    bk : (M, N+1) numpy.ndarray
        Vector containing values for all wavenumbers :math:`M` and modes up to
        order :math:`N`

    Returns
    -------
    Bk : (M, 2*N+1, 2*N+1) numpy.ndarray
        Multidimensional array containing diagnonal matrices with input
        vector on main diagonal.
    """
    if len(bk.shape) == 1:
        bk = bk[np.newaxis, :]
    K, N = bk.shape
    Bk = np.zeros([K, N, N], dtype=complex)
    for k in range(K):
        Bk[k, :, :] = np.diag(bk[k, :])
    return np.squeeze(Bk)
