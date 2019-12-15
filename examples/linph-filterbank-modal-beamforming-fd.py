"""Angular-spectral responses.

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
N = 5
R = 0.049
max_boost = 30

fmin, fmax, numf = 10, 20000, 500
f = np.logspace(np.log10(fmin), np.log10(fmax), num=numf)
f_xo = crossover_frequencies(N, R, max_boost, modal_weight=maxre_sph)
H_fbank = tf_linph_filterbank(f_xo, f, type='butter')

# Look directions
azimin, azimax, numazi = -np.pi, np.pi, 361
azi = np.linspace(azimin, azimax, num=numazi, endpoint=True)

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


# Plots
def add_cbar(ax, im, pad=0.05, width=0.05, **kwarg):
    cax = plt.axes([ax.get_position().xmax + pad, ax.get_position().y0,
                    width, ax.get_position().height], **kwarg)
    plt.colorbar(im, cax=cax)


def decorate_singleplot(ax, **kwarg):
    ax.axis('tight')
    ax.set_yscale('log')
    ax.set(**kwarg)


def decorate_subplots(axes, im, **kwarg):
    for axi in axes.flat:
        decorate_singleplot(axi, **kwarg)
        if axi.is_last_row():
            axi.set_xlabel('Angle / deg')
        if axi.is_first_col():
            axi.set_ylabel('Frequency / Hz')
    add_cbar(axes[0, 1], im, pad=0.02, width=0.03, xlabel='dB')


figsize = (4, 4)
figsize_subplots = (8, 10)
azideg = np.rad2deg(azi)
azilim = azideg[0], azideg[-1]
phiticks = np.arange(np.rad2deg(azimin), np.rad2deg(azimax)+90, 90)
flim = fmin, fmax
im_kw = {'cmap': 'Blues', 'vmin': -60, 'vmax': 20}
gridspec_kw = {'wspace': 0.1}

# Beamformer output
fig, ax = plt.subplots(figsize=figsize)
im = ax.pcolormesh(azideg, f, db(Y.T), **im_kw)
ax.plot(np.zeros(N), f_xo, 'wx', alpha=0.5)
decorate_singleplot(ax, xticks=phiticks, xlabel='Azimuth / deg',
                    ylabel='Frequnecy / Hz')
add_cbar(ax, im, xlabel='dB')
plt.savefig('spatial-responses-fd.png', bbox_inches='tight')

# each filterbank
fig, ax = plt.subplots(figsize=figsize_subplots, ncols=2, nrows=3,
                       sharex=True, sharey=True, gridspec_kw=gridspec_kw)
for i, (axi, Yi) in enumerate(zip(ax.flat, Y_band)):
    im = axi.pcolormesh(azideg, f, db(Yi.T), **im_kw)
    axi.set_title('Subband #{}'.format(i))
decorate_subplots(ax, im, xticks=phiticks, xlim=azilim, ylim=flim)
plt.savefig('spatial-responses-subband-fd.png', bbox_inches='tight')

# each order
fig, ax = plt.subplots(figsize=figsize_subplots, ncols=2, nrows=3,
                       sharex=True, sharey=True, gridspec_kw=gridspec_kw)
for i, (axi, Yi) in enumerate(zip(ax.flat, Y_order)):
    im = axi.pcolormesh(azideg, f, db(Yi.T), **im_kw)
    axi.set_title('Order #{}'.format(i))
decorate_subplots(ax, im, xticks=phiticks, xlim=azilim, ylim=flim)
plt.savefig('spatial-responses-order-fd.png', bbox_inches='tight')
