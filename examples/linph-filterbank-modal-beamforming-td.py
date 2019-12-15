"""Angular-temporal responses.

- No spatial sampling

    Reference
    ---------
    Franz Zotter, "A linear-phase filter-bank approach to process
    rigid spherical microphone array recordings", in Proc. IcETRAN,
    Palic, Serbia, 2018.
"""
import numpy as np
import matplotlib.pyplot as plt
from micarray.util import maxre_sph, point_spread, modal_norm, db
from micarray.modal.radial import crossover_frequencies, tf_linph_filterbank

c = 343
fs = 44100
N = 5
R = 0.049

Nfft = 2048
fmin, fmax, numf = 0, fs/2, int(Nfft/2)+1
f = np.linspace(fmin, fmax, num=numf, endpoint=True)

max_boost = 30
f_xo = crossover_frequencies(N, R, max_boost, modal_weight=maxre_sph)
H_fbank = tf_linph_filterbank(f_xo, f, type='butter')

# Look directions
azimin, azimax, numazi = -np.pi, np.pi, 360
azi = np.linspace(azimin, azimax, num=numazi)

# Beamformer output
Y_band = np.zeros((N+1, numazi, numf), dtype='complex')
Y_order = np.zeros((N+1, numazi, numf), dtype='complex')
for i, Hi in enumerate(H_fbank):
    ps = point_spread(i, azi, modal_weight=maxre_sph, equalization='diffuse')
    Yi = ps[:, :, np.newaxis] * Hi
    Y_order[:i+1, :] += Yi
    Y_band[i, :] = np.sum(Yi, axis=0)
normN = modal_norm(maxre_sph(N))
Y_band *= normN
Y_order *= normN
Y = np.sum(Y_order, axis=0)

y_order = np.fft.irfft(Y_order, n=Nfft, axis=-1)
y_order = np.roll(y_order, int(Nfft/2), axis=-1)
y_band = np.fft.irfft(Y_band, n=Nfft, axis=-1)
y_band = np.roll(y_band, int(Nfft/2), axis=-1)
y = np.fft.irfft(Y, n=Nfft, axis=-1)
y = np.roll(y, int(Nfft/2), axis=-1)


# Plots
def add_cbar(ax, im, pad=0.05, width=0.05, **kwarg):
    cax = plt.axes([ax.get_position().xmax + pad, ax.get_position().y0,
                    width, ax.get_position().height], **kwarg)
    plt.colorbar(im, cax=cax)


def decorate_singleplot(ax, **kwarg):
    ax.axis('tight')
    ax.set(**kwarg)


def decorate_subplots(axes, im, **kwarg):
    for axi in axes.flat:
        decorate_singleplot(axi, **kwarg)
        if axi.is_last_row():
            axi.set_xlabel('Angle / deg')
        if axi.is_first_col():
            axi.set_ylabel('Time / ms')
    add_cbar(axes[0, 1], im, pad=0.02, width=0.03, xlabel='dB')


figsize = (4, 4)
figsize_subplots = (8, 10)
azideg = np.rad2deg(azi)
azilim = azideg[0], azideg[-1]
aziticks = np.arange(np.rad2deg(azimin), np.rad2deg(azimax)+90, 90)
tau = Nfft / 2 / fs * 1000
tlim = [-3, 3]
im_kw = {'cmap': 'Blues', 'vmin': -60, 'vmax': 20,
         'extent': [azilim[0], azilim[-1], -tau, tau]}
gridspec_kw = {'wspace': 0.1}

# Beamformer impulse responses
fig, ax = plt.subplots(figsize=figsize)
im = ax.imshow(db(y.T), **im_kw)
decorate_singleplot(ax, xticks=aziticks, xlim=azilim, ylim=tlim,
                    xlabel='Azimuth / deg', ylabel='Time / ms')
add_cbar(ax, im, xlabel='dB')
plt.savefig('spatial-responses-td.png', bbox_inches='tight')

# each filterbank
fig, ax = plt.subplots(figsize=figsize_subplots, ncols=2, nrows=3,
                       sharex=True, sharey=True, gridspec_kw=gridspec_kw)
for i, (axi, yi) in enumerate(zip(ax.flat, y_band)):
    im = axi.imshow(db(yi.T), **im_kw)
    axi.set_title('Subband #{}'.format(i))
decorate_subplots(ax, im, xticks=aziticks, xlim=azilim, ylim=tlim)
plt.savefig('spatial-responses-subband-td.png', bbox_inches='tight')

# each order
fig, ax = plt.subplots(figsize=figsize_subplots, ncols=2, nrows=3,
                       sharex=True, sharey=True, gridspec_kw=gridspec_kw)
for i, (axi, yi) in enumerate(zip(ax.flat, y_order)):
    im = axi.imshow(db(yi.T), **im_kw)
    axi.set_title('Order #{}'.format(i))
decorate_subplots(ax, im, xticks=aziticks, xlim=azilim, ylim=tlim)
plt.savefig('spatial-responses-order-td.png', bbox_inches='tight')
