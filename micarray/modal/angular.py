from __future__ import division
import numpy as np
from scipy import special
from .. import util


def sht_matrix(N, azi, elev, weights=None):
    """ (N+1)**2 x M SHT matrix"""
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
    """ (N+1) x M matrix of weighted Legendre Polynominals
        2*n+1/4*pi * P_n(ctheta)
    """
    if ctheta.ndim == 0:
        M = 1
    else:
        M = len(ctheta)
    Lmn = np.zeros([N+1, M], dtype=complex)
    for n in range(N+1):
        Lmn[n, :] = (2*n+1)/(4*np.pi) * np.polyval(special.legendre(n), ctheta)

    return Lmn


def grid_equal_angle(n):
    """ equi_angular grid on sphere.
    (cf. Rafaely book, sec.3.2)
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
    (cf. Rafaely book, sec.3.3)
    """
    azi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)
    x, weights = np.polynomial.legendre.leggauss(n+1)
    elev = np.arccos(x)
    azi = np.tile(azi, n+1)
    elev = np.repeat(elev, 2*n+2)
    weights = np.repeat(weights, 2*n+2)
    weights *= np.pi / (n+1)      # sum(weights) == 4pi
    return azi, elev, weights
