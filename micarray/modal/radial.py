from __future__ import division
import numpy as np
from scipy import special
from .. import util


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
    for i, x in enumerate(kr):
        bn[i, :] = bn[i, :] * 4*np.pi * (1j)**n
    return bn


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
    for i, x in enumerate(krs):
        hn = special.spherical_jn(n, x) - 1j * special.spherical_yn(n, x)
        bn[i, :] = bn[i, :] * 4*np.pi * (-1j) * hn * k[i]

    return bn


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
    n = np.arange(N+1)
    bns = np.zeros((len(kr), N+1), dtype=complex)
    for i, x in enumerate(kr):
        jn = special.spherical_jn(n, x)
        if setup == 'open':
            bn = jn
        elif setup == 'card':
            bn = jn - 1j * special.spherical_jn(n, x, derivative=True)
        elif setup == 'rigid':
            jnd = special.spherical_jn(n, x, derivative=True)
            hn = jn - 1j * special.spherical_yn(n, x)
            hnd = jnd - 1j * special.spherical_yn(n, x, derivative=True)
            bn = jn - jnd/hnd*hn
        else:
            raise ValueError('setup must be either: open, card or rigid')
        bns[i, :] = bn
    return np.squeeze(bns)


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
    dn[0, 1:] = dn[1, 1:]
    dn = dn * hn
    if not np.isfinite(dn).all():
        raise UserWarning("Filter not finite")
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
    bk = _repeat_n_m(bk)
    if len(bk.shape) == 1:
        bk = bk[np.newaxis, :]
    K, N = bk.shape
    Bk = np.zeros([K, N, N], dtype=complex)
    for k in range(K):
        Bk[k, :, :] = np.diag(bk[k, :])
    return np.squeeze(Bk)


def _repeat_n_m(v):
    krlist = [np.tile(v, (2*i+1, 1)).T for i, v in enumerate(v.T.tolist())]
    return np.squeeze(np.concatenate(krlist, axis=-1))
