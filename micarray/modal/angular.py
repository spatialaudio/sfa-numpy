from __future__ import division
import numpy as np
from scipy import special
from .. import util
from warnings import warn
try:
    import quadpy  # only for grid_lebedev()
except ImportError:
    pass


def sht_matrix(N, azi, colat, weights=None):
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


    (Note: :math:`\mathbf{Y}` is interpreted as the inverse transform (or synthesis)
    matrix in examples and documentation.)

    Parameters
    ----------
    N : int
        Maximum order.
    azi : (Q,) array_like
        Azimuth.
    colat : (Q,) array_like
        Colatitude.
    weights : (Q,) array_like, optional
        Quadrature weights.

    Returns
    -------
    Ymn : (Q, (N+1)**2) numpy.ndarray
        Matrix of spherical harmonics.

    """
    azi = util.asarray_1d(azi)
    colat = util.asarray_1d(colat)
    if azi.ndim == 0:
        Q = 1
    else:
        Q = len(azi)
    if weights is None:
        weights = np.ones(Q)
    Ymn = np.zeros([Q, (N+1)**2], dtype=complex)
    i = 0
    for n in range(N+1):
        for m in range(-n, n+1):
            Ymn[:, i] = weights * special.sph_harm(m, n, azi, colat)
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
        1 & e^{i\varphi[0]} & \cdots & e^{iN\varphi[0]} & e^{-iN\varphi[0]} & \cdots & e^{-i\varphi[0]} \\
        1 & e^{i\varphi[1]} & \cdots & e^{iN\varphi[1]} & e^{-iN\varphi[1]} & \cdots & e^{-i\varphi[1]} \\
        \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
        1 & e^{i\varphi[Q-1]} & \cdots & e^{iN\varphi[Q-1]} & e^{-iN\varphi[Q-1]} & \cdots & e^{-i\varphi[Q-1]}
        \end{array} \right]

    (Note: :math:`\Psi` is interpreted as the inverse transform (or synthesis)
    matrix in examples and documentation.)


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
    Psi : (Q, 2N+1) numpy.ndarray
        Matrix of circular harmonics.

    """
    pol = util.asarray_1d(pol)
    if pol.ndim == 0:
        Q = 1
    else:
        Q = len(pol)
    if weights is None:
        weights = np.ones(Q)
    Psi = np.zeros([Q, (2*N+1)], dtype=complex)
    order = np.roll(np.arange(-N, N+1), -N)
    for i, n in enumerate(order):
        Psi[:, i] = weights * np.exp(1j * n * pol)
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
    colat : array_like
        Colatitude.
    weights : array_like
        Quadrature weights.
    """
    azi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)
    colat, d_colat = np.linspace(0, np.pi, 2*n+2, endpoint=False, retstep=True)
    colat += d_colat/2

    weights = np.zeros_like(colat)
    p = np.arange(1, 2*n+2, 2)
    for i, theta in enumerate(colat):
        weights[i] = 2*np.pi/(n+1) * np.sin(theta) * np.sum(np.sin(p*theta)/p)

    azi = np.tile(azi, 2*n+2)
    colat = np.repeat(colat, 2*n+2)
    weights = np.repeat(weights, 2*n+2)
    weights /= n+1     # sum(weights) == 4pi
    return azi, colat, weights


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
    colat : array_like
        Colatitude.
    weights : array_like
        Quadrature weights.
    """
    azi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)
    x, weights = np.polynomial.legendre.leggauss(n+1)
    colat = np.arccos(x)
    azi = np.tile(azi, n+1)
    colat = np.repeat(colat, 2*n+2)
    weights = np.repeat(weights, 2*n+2)
    weights *= np.pi / (n+1)      # sum(weights) == 4pi
    return azi, colat, weights


def grid_equal_polar_angle(n):
    """Equi-angular sampling points on a circle.

    Parameters
    ----------
    n : int
        Maximum order

    Returns
    -------
    pol : array_like
        Polar angle.
    weights : array_like
        Weights.
    """
    num_mic = 2*n+1
    pol = np.linspace(0, 2*np.pi, num=num_mic, endpoint=False)
    weights = 1/num_mic * np.ones(num_mic)
    return pol, weights


def grid_lebedev(n):
    """Lebedev sampling points on sphere.

    (Maximum n is 65. We use what is available in quadpy, some n may not be
    tight, others produce negative weights.

    Parameters
    ----------
    n : int
        Maximum order.

    Returns
    -------
    azi : array_like
        Azimuth.
    colat : array_like
        Colatitude.
    weights : array_like
        Quadrature weights.

    """
    def available_quadrature(d):
        """Get smallest availabe quadrature of of degree d.

        see:
        https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html
        """
        l = list(range(1, 32, 2)) + list(range(35, 132, 6))
        matches = [x for x in l if x >= d]
        return matches[0]

    if n > 65:
        raise ValueError("Maximum available Lebedev grid order is 65. "
                         "(requested: {})".format(n))

    # this needs https://pypi.python.org/pypi/quadpy
    q = quadpy.sphere.Lebedev(degree=available_quadrature(2*n))
    if np.any(q.weights < 0):
        warn("Lebedev grid of order {} has negative weights.".format(n))
    azi = q.azimuthal_polar[:, 0]
    colat = q.azimuthal_polar[:, 1]
    return azi, colat, 4*np.pi*q.weights
