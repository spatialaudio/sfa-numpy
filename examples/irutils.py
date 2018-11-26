import numpy as np
from scipy.signal import remez, fftconvolve, bilinear_zpk, zpk2sos, kaiser
from scipy.special import comb


def greens_plane(npw, x, doa=False, c=343):
    """Greens function of a plane wave.

    Parameters
    ----------
    npw : (3,) array_like
        Wave vector
    x : (N, 3) array_like
        Receiver positions in the Cartesian coordinate [m]
    doa : bool
        True if npw is the propagation direction,
        False if npw is the direction of arrival.
    c : float
        Speed of sound [m/s]

    Returns
    -------
    delay : (N,) array_like
        Delay with respect to the origin [s]
    signal : (N,) array_like
        Amplitude

    """
    npw = np.array(npw) / np.linalg.norm(npw)
    x = np.array(x)
    if doa:
        npw *= -1
    if x.ndim == 1 & len(x) == 3:
        x = x[np.newaxis, :]
    return x.dot(npw) / c, np.ones(x.shape[0])


def greens_point(xs, x, c=343):
    """Greens function of a point source.

    Parameters
    ----------
    xs : (3,) array_like
        Source position in the Cartesian coordiate [m]
    x : (N, 3) array_like
        Receiver positions in the Cartesian coordinate [m]
    c : float
        Speed of sound [m/s]

    Returns
    -------
    delay : (N,) array_like
        Propagation delay [s]
    signal : (N,) array_like
        Amplitude

    """
    xs = np.array(xs)
    x = np.array(x)
    if x.ndim == 1 & len(x) == 3:
        x = x[np.newaxis, :]
    distance = np.linalg.norm(x - xs[np.newaxis, :], axis=-1)
    return distance / c, 1 / 4 / np.pi / distance


def fractional_delay(delay, signal, fs, oversample=2,
                     fdfilt_order=11, h_lpf=None):
    """Convert delay and signal into sample shift and fractional dealy filters.

    Parameters
    ----------
    delay : (N,) array_like
        Preceeding delay [s]
    signal : (N, L) array_like
        Nonzero signal
    fs : int
        Sampling frequency [Hz]
    oversample : int
        Oversampling rate
    fdfilt_order : int
        Order of the fractional delay filter
    **kwargs :
        Keyword arguments of `fir_minmax`

    Returns
    -------
    shift : (N,) array_like
        Integer delay
    signal : (N, M) array_like
        Fractional dealy
    """
    if signal.ndim == 1 and len(signal) == len(delay):
        signal = signal[:, np.newaxis]
    if h_lpf is None:
        h_lpf = fir_minmax(fs, oversample, filt_order=64, wpass=0.85, wstop=1,
                           att=-100, weight=[1, 1e-5])
    shift_fd, h_fd = lagrange_fdfilter(delay, fdfilt_order, fs=oversample * fs)
    shift_lpf = int((len(h_lpf) + 1) / 2)
    h_fd = fftconvolve(h_fd, h_lpf[np.newaxis, :])

    shift_fd -= shift_lpf
    shift = shift_fd // oversample
    res = shift_fd % oversample

    h_fd = np.column_stack((h_fd, np.zeros((len(h_fd), oversample-1))))
    for n in range(len(shift_fd)):
        h_fd[n, :] = np.roll(h_fd[n, :], res[n])
    h_fd = h_fd[:, ::oversample]
    signal = np.stack([fftconvolve(x1, x2) for x1, x2 in zip(h_fd, signal)])
    return shift, signal, fs


def lagrange_fdfilter(delay, L, fs):
    """Lagrange fractional delay filter design

    Parameters
    ----------
    delay : (N,) array_like
        Delay [s]
    L : int
        Filter length

    Returns
    -------
    shift : (N,) array_like
        Integer delay
    signal : (N, L) array_like
        Fractional delay
    """
    N = len(delay)
    shift = np.zeros(L)
    signal = np.zeros((N, L))

    d = delay * fs
    if L % 2 == 0:
        n0 = np.ceil(d).astype(int)
        Lh = int(L / 2)
    elif L % 2 == 1:
        n0 = np.round(d).astype(int)
        Lh = (np.floor(L / 2)).astype(int)
    idx = n0[:, np.newaxis] + np.arange(-Lh, -Lh + L)[np.newaxis, :]
    shift = n0 - Lh
    ii = np.arange(L)
    common_weight = comb(L - 1, ii) * (-1)**ii

    isint = (d % 1 == 0)
    signal[~isint, :] = common_weight[np.newaxis, :] / (d[~isint, np.newaxis] - idx[~isint, :])
    signal[~isint, :] /= np.sum(signal[~isint, :], axis=-1)[:, np.newaxis]
    signal[isint, Lh] = 1
    return shift, signal


def fir_minmax(fs, oversample, filt_order=64, wpass=0.85, wstop=1,
               att=-100, weight=[1, 1e-5]):
    """Low-pass filter for sampling rate conversion."""
    fpass = wpass * fs / 2
    fstop = wstop * fs / 2
    if oversample != 1:
        return remez(2 * filt_order + 1,
                     [0, fpass, fstop, oversample * fs / 2],
                     [1, 10**(att / 20)],
                     weight=weight, fs=oversample * fs)
    else:
        return np.array(1)


def ir_matrix(shift, signal):
    """
    Construct an IR matrix

    Parameters
    ----------
    shift : (N,) array_like
        Preceeding integer delay [sample]
    signal : (N, L) array_like
        Nonzero coefficients

    Returns
    -------
    overall_shift : int
        Overall shift [sample]
    h : (Nh, L) array_like
        IR matrix
    """
    N, L = signal.shape
    overall_shift = np.min(shift)
    Nh = L + np.max(shift) - overall_shift
    shift -= overall_shift
    h = np.zeros((N, Nh))
    for n in range(N):
        idx = np.arange(shift[n], shift[n] + L).astype(int)
        h[n, idx] = signal[n, :]
    return overall_shift, h.T


def bessel_poly(n):
    """Bessel polynomial coefficients"""
    beta = np.zeros(n + 1)  # n-th order polynomial has (n+1) coeffcieints
    beta[n] = 1  # This is always 1!
    for k in range(n - 1, -1, -1):  # Recurrence relation
        beta[k] = beta[k + 1] * (2 * n - k) * (k + 1) / (n - k) / 2
    return beta


def derivative_bessel_poly(n):
    """Derivative of the Bessel polynomial"""
    gamma = bessel_poly(n + 1)
    gamma[:-1] -= n * decrease_bessel_order_by_one(gamma)
    return gamma


def decrease_bessel_order_by_one(beta):
    n = len(beta)-1
    alpha = np.zeros(n)
    for k in range(n - 1):
        alpha[k] = beta[k + 1] * (k + 1) / (2 * n - k - 1)
    alpha[-1] = 1  # This is always one
    return alpha


def increase_bessel_order_by_one(beta):
    n = len(beta)
    alpha = np.zeros(n + 1)
    for k in range(n):
        alpha[k + 1] = beta[k] * (2 * n - k - 1) / (k + 1)
    alpha[0] = alpha[1]  # The first two coefficients are the same
    return alpha


def normalize_digital_filter_gain(s0, sinf, z0, zinf, fc, fs):
    """Match the digital filter gain at a control frequency.

    Parameters
    ----------
    s0 : (N,) array_like
        zeros in the Laplace domain
    sinf : (N,) array_like
        polse in the Laplace domain
    z0 : (N,) array_like
        zeros in the z-domain
    zinf : (N,) array_like
        zeros in the z-domain
    fc : float
        Control frequency in Hz
    fs : int
        Sampling frequency in Hz

    """
    k = 1
    s_c = 1j * 2 * np.pi * fc
    z_c = 1 / np.exp(1j * 2 * np.pi * fc / fs)
    k *= np.prod(s_c - s0) / np.prod(s_c - sinf)
    k *= np.prod(1 - zinf * z_c) / np.prod(1 - z0 * z_c)
    return np.abs(k)


def pre_warp(s, fs):
    return np.real(s) + 1j * 2 * fs * np.tan(np.imag(s) / 2 / fs)


def radial_filter(N, r, setup, c=343, fs=44100, pzmap='mz'):
    """Radial filter design for a plane wave.

    Parameters
    ----------
    N : int
        Maximum order.
    r : float
        Radius of microphone array
    setup : {'rigid'}
        Array configuration (e.g. rigid)
    pzmap : {'mz', 'bt'}
        Pole-zero mapping method (matched-z, bilinear transform)

    Returns
    -------
    delay : float
        Overall delay
    sos : list of (L, 6) arrays
        Second-order section filters
    """
    sos = []
    if setup is 'rigid':
        for n in range(N + 1):
            s0 = c / r * np.zeros(n)
            sinf = c / r * np.roots(derivative_bessel_poly(n)[::-1])
            if pzmap is 'mz':
                z0 = np.exp(s0 / fs)
                zinf = np.exp(sinf / fs)
            elif pzmap is 'bt':
                z0, zinf, _ = bilinear_zpk(s0, pre_warp(sinf, fs), 1, fs=fs)
                z0 = np.delete(z0, -1)
            fc = c * n / 2 / np.pi / r
            k = normalize_digital_filter_gain(s0, sinf,
                                              z0, zinf, fc, fs) * c / r
            sos.append(zpk2sos(z0, zinf, k, pairing='nearest'))
    return -r / c, sos


def modal_window(N, beta=0):
    return kaiser(2 * N + 1, beta=beta)[N:]
