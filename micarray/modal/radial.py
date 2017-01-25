from __future__ import division
import numpy as np
from scipy import special


def spherical(N, kr, setup, plane_wave):
    if np.isscalar(kr):
        kr = np.asarray([kr])
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
        if plane_wave:
            bn = bn * 4*np.pi * (1j)**n
        bns[i, :] = bn
        bns = np.squeeze(bns)
    return bns


def regularize(dn, a0, method):
    """(cf. Rettberg, Spors : DAGA 2014)"""

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
