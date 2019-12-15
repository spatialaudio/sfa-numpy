"""Impulse responses of linear-phase filterbank and radial filters.

    Reference
    ---------
    Franz Zotter, "A linear-phase filter-bank approach to process
    rigid spherical microphone array recordings", in Proc. IcETRAN,
    Palic, Serbia, 2018.
"""
import numpy as np
import matplotlib.pyplot as plt
from micarray.util import maxre_sph, modal_norm
from micarray.modal.radial \
    import crossover_frequencies, spherical_hn2, tf_linph_filterbank


c = 343
fs = 44100
Nfft = 2048

R = 0.049
N = 5
max_boost = 30
f_xo = crossover_frequencies(N, R, max_boost, modal_weight=maxre_sph)

fmin, fmax, numf = 0, fs/2, int(Nfft/2)+1
f = np.linspace(fmin, fmax, num=numf)
f[0] = 0.5 * f[1]
kr = 2 * np.pi * f / c * R
H_fbank = tf_linph_filterbank(f_xo, f, type='butter')

# Prototype radial filters
H_proto = np.stack([1j**(-n-1) * (kr)**2
                    * spherical_hn2(n, kr, derivative=True)
                    for n in range(N+1)])

H_radial = np.zeros_like(H_proto)
for i, Hi in enumerate(H_fbank):
    ai = maxre_sph(i)
    ai *= 1 / modal_norm(ai)
    for n, Hr in enumerate(H_proto[:i+1]):
        H_radial[n] += Hr * Hi * ai[n]
H_radial *= modal_norm(maxre_sph(N))

# inverse DFT
h_fbank = np.stack([np.fft.irfft(Hi, n=Nfft, axis=-1) for Hi in H_fbank])
h_radial = np.stack([np.fft.irfft(Hi, n=Nfft, axis=-1) for Hi in H_radial])
h_fbank = np.roll(h_fbank, int(Nfft/2), axis=-1)
h_radial = np.roll(h_radial, int(Nfft/2), axis=-1)

t = ((np.arange(Nfft) - Nfft/2) / fs) * 1000
t_R = t - R/c*1000


# Plots
def decorate_subplots(axes, **kwargs):
    for ax in axes.flat:
        if ax.is_first_col():
            ax.set_ylabel('Amplitude / dB')
        if ax.is_last_row():
            ax.set_xlabel('Time / ms')
        ax.grid(True)
        ax.set(**kwargs)


# Impulse responses
figsize = (8, 10)
gridspec_kw = {'wspace': 0.1}
tlim = -2, 2
ylim = -40, None

# each filter bank
fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=3,
                       sharex=True, sharey=True, gridspec_kw=gridspec_kw)
for i, (axi, hi) in enumerate(zip(ax.flat[:N+1], h_fbank)):
    hi *= 1 / np.max(np.abs(hi))
    axi.plot(t, hi)
    axi.set_title('Subband #{}'.format(i))
decorate_subplots(ax, xlim=tlim)
plt.savefig('./linph-filterbank-td.png')

# each order
fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=3,
                       sharex=True, sharey=True, gridspec_kw=gridspec_kw)
for i, (axi, hi) in enumerate(zip(ax.flat[:N+1], h_radial)):
    hi *= 1 / np.max(np.abs(hi))
    axi.plot(t_R, hi)
    axi.set_title('Order #{}'.format(i))
decorate_subplots(ax, xlim=tlim)
plt.savefig('./radialfilters-td.png')
