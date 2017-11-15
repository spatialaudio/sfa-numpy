from __future__ import division
import numpy as np
from scipy import special
from .. import util


def sht_matrix(N, azi, elev, weights=None):
    r"""Matrix of spherical harmonics up to order N for given angles.

    Computes a matrix of spherical harmonics up to order :math:`N`
    for the given angles/grid.

    .. math::

        \mathbf{Y} = \left[ \begin{array}{cccccc} 
        Y_0^0(\theta[0], \phi[0]) & Y_1^{-1}(\theta[0], \phi[0]) & Y_1^0(\theta[0], \phi[0]) & Y_1^1(\theta[0], \phi[0]) & \dots & Y_N^N(\theta[0], \phi[0])  \\
        Y_0^0(\theta[1], \phi[1]) & Y_1^{-1}(\theta[1], \phi[1]) & Y_1^0(\theta[1], \phi[1]) & Y_1^1(\theta[1], \phi[1]) & \dots & Y_N^N(\theta[1], \phi[1])  \\
        \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
        Y_0^0(\theta[Q-1], \phi[Q-1]) & Y_1^{-1}(\theta[Q-1], \phi[Q-1]) & Y_1^0(\theta[Q-1], \phi[Q-1]) & Y_1^1(\theta[Q-1], \phi[Q-1]) & \dots & Y_N^N(\theta[Q-1], \phi[Q-1])
        \end{array} \right]

    where

    .. math::

        Y_n^m(\theta, \phi) = \sqrt{\frac{2n + 1}{4 \pi} \frac{(n-m)!}{(n+m)!}} P_n^m(\cos \theta) e^{i m \phi}

    Parameters
    ----------
    N : int
        Maximum order.
    azi : (Q,) array_like
        Azimuth.
    elev : (Q,) array_like
        Elevation.
    weights : (Q,) array_like, optional
        Quadrature weights.

    Returns
    -------
    Ymn : ((N+1)**2, Q) numpy.ndarray
        Matrix of spherical harmonics.
    """
    azi = util.asarray_1d(azi)
    elev = util.asarray_1d(elev)
    if azi.ndim == 0:
        M = 1
    else:
        M = len(azi)
    if weights is None:
        weights = np.ones(M)
    Ymn = np.zeros([(N+1)**2, M], dtype=complex)
    i = 0
    for n in range(N+1):
        for m in range(-n, n+1):
            Ymn[i, :] = weights * special.sph_harm(m, n, azi, elev)
            i += 1
    return Ymn


def Legendre_matrix(N, ctheta):
    r"""Matrix of weighted Legendre Polynominals.

    Computes a matrix of weighted Legendre polynominals up to order N for
    the given angles

    .. math::

        L_n(\theta) = \frac{2n+1}{4 \pi} P_n(\theta)

    Parameters
    ----------
    N : int
        Maximum order.
    ctheta : (Q,) array_like
        Angles.

    Returns
    -------
    Lmn : ((N+1), Q) numpy.ndarray
        Matrix containing Legendre polynominals.
    """
    if ctheta.ndim == 0:
        M = 1
    else:
        M = len(ctheta)
    Lmn = np.zeros([N+1, M], dtype=complex)
    for n in range(N+1):
        Lmn[n, :] = (2*n+1)/(4*np.pi) * np.polyval(special.legendre(n), ctheta)

    return Lmn


def cht_matrix(N, pol, weights=None):
    r"""Matrix of circular harmonics up to order N for given angles.

    Computes a matrix of circular harmonics up to order :math:`N`
    for the given angles/grid.

    .. math::
        \Psi = \left[ \begin{array}{ccccccc}
        1 & \cdots & e^{i\varphi[0]} & e^{iN\varphi[0]} & e^{-iN\varphi[0]} & \cdots & e^{-i\varphi[0]} \\
        1 & \cdots & e^{i\varphi[1]} & e^{iN\varphi[1]} & e^{-iN\varphi[1]} & \cdots & e^{-i\varphi[1]} \\
        \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
        1 & \cdots & e^{i\varphi[Q-1]} & e^{iN\varphi[Q-1]} & e^{-iN\varphi[Q-1]} & \cdots & e^{-i\varphi[Q-1]}
        \end{array} \right]

    Parameters
    ----------
    N : int
        Maximum order.
    pol : (Q,) array_like
        Polar angle.
    weights : (Q,) array_like, optional
        Weights.

    Returns
    -------
    Psi : (2N+1, Q) numpy.ndarray
        Matrix of circular harmonics.
    """
    pol = util.asarray_1d(pol)
    if pol.ndim == 0:
        Q = 1
    else:
        Q = len(pol)
    if weights is None:
        weights = np.ones(Q)
    Psi = np.zeros([(2*N+1), Q], dtype=complex)
    order = np.roll(np.arange(-N, N+1), -N)
    for i, n in enumerate(order):
        Psi[i, :] = weights * np.exp(1j * n * pol)
    return Psi


def grid_equal_angle(n):
    """Equi-angular sampling points on a sphere.

    According to (cf. Rafaely book, sec.3.2)

    Parameters
    ----------
    n : int
        Maximum order.

    Returns
    -------
    azi : array_like
        Azimuth.
    elev : array_like
        Elevation.
    weights : array_like
        Quadrature weights.
    """
    azi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)
    elev, d_elev = np.linspace(0, np.pi, 2*n+2, endpoint=False, retstep=True)
    elev += d_elev/2

    weights = np.zeros_like(elev)
    p = np.arange(1, 2*n+2, 2)
    for i, theta in enumerate(elev):
        weights[i] = 2*np.pi/(n+1) * np.sin(theta) * np.sum(np.sin(p*theta)/p)

    azi = np.tile(azi, 2*n+2)
    elev = np.repeat(elev, 2*n+2)
    weights = np.repeat(weights, 2*n+2)
    weights /= n+1     # sum(weights) == 4pi
    return azi, elev, weights


def grid_gauss(n):
    """ Gauss-Legendre sampling points on sphere.

    According to (cf. Rafaely book, sec.3.3)

    Parameters
    ----------
    n : int
        Maximum order.

    Returns
    -------
    azi : array_like
        Azimuth.
    elev : array_like
        Elevation.
    weights : array_like
        Quadrature weights.
    """
    azi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)
    x, weights = np.polynomial.legendre.leggauss(n+1)
    elev = np.arccos(x)
    azi = np.tile(azi, n+1)
    elev = np.repeat(elev, 2*n+2)
    weights = np.repeat(weights, 2*n+2)
    weights *= np.pi / (n+1)      # sum(weights) == 4pi
    return azi, elev, weights


def grid_equal_polar_angle(M, phi0=0):
    """Equi-angular sampling points on a circle.

    Parameters
    ----------
    M : int
        Number of microphones.
    phi0 : float
        Angular shift

    Returns
    -------
    pol : array_like
        Polar angle.
    weights : array_like
        Weights.
    """
    pol = np.linspace(0, 2*np.pi, num=M, endpoint=False) + phi0
    weights = 1/M * np.ones(M)
    return pol, weights
