"""Fourth-order Ambisonics (Eigenmike em32).

    Reference
    ---------
    Franz Zotter, "A linear-phase filter-bank approach to process
    rigid spherical microphone array recordings",
"""
import numpy as np
import matplotlib.pyplot as plt
import micarray
from scipy.special import sph_harm
from scipy.signal import unit_impulse, sosfilt, fftconvolve as conv,\
                         kaiser, freqz
from micarray.util import db
from micarray.modal.radial import crossover_frequencies, sos_radial_filter,\
                                  tf_equalized_radial_filters

c = 343
fs = 44100
Nfft = 2048

array_order = 4
azi_m, colat_m, R = np.loadtxt('em32.txt').T
R = R[0]
Ynm_m = micarray.modal.angular.sht_matrix(array_order, azi_m, colat_m)

# Incident plane wave captured by mic array
sf_order = 20
Nimp = 2048
imp = unit_impulse(Nimp)
azi_pw, colat_pw = 0 * np.pi, 0.5 * np.pi
delay, sos = sos_radial_filter(sf_order, R, fs=fs, setup='rigid')
sos_irs = np.stack([sosfilt(sos[n], imp) for n in range(sf_order+1)])
snm = np.column_stack([
        sos_irs[n] * np.conj(sph_harm(m, n, azi_pw, colat_pw))
        for n in range(sf_order+1)
        for m in range(-n, n+1)])
snm *= 4 * np.pi
Ynm_s = micarray.modal.angular.sht_matrix(sf_order, azi_m, colat_m)
s = np.real(np.squeeze(np.matmul(Ynm_s, snm[:, :, np.newaxis])))

# Radial filters
max_boost = 30
f_xo = crossover_frequencies(array_order, R, max_boost)
Nfft = 2048
f_dft = np.fft.rfftfreq(Nfft, d=1/fs)
f_dft[0] = 0.1 * f_dft[1]
H_radial = tf_equalized_radial_filters(array_order, R, f_dft, max_boost,
                                       type='butter')
h_radial = np.stack([np.fft.irfft(Hi, n=Nfft, axis=-1) for Hi in H_radial])
h_radial = np.roll(h_radial, int(Nfft/2), axis=-1)
h_radial *= kaiser(Nfft, beta=8.6)
pre_delay = -Nfft / 2 / fs

# beamforming
bf_order = array_order
N_angle = 360
azi_l, colat_l = np.linspace(-np.pi, np.pi, num=N_angle), 0.5 * np.pi
Ynm_l = micarray.modal.angular.sht_matrix(bf_order, azi_l, colat_l)
snm_hat = np.squeeze(np.matmul(np.linalg.pinv(Ynm_m), s[:, :, np.newaxis]))
ynm = np.column_stack([
        conv(h_radial[n], snm_hat[:, n**2+n+m])
        for n in range(bf_order+1)
        for m in range(-n, n+1)])
y = np.real(np.squeeze(np.matmul(Ynm_l, ynm[:, :, np.newaxis])))

# frequency responses
fmin, fmax, numf = 20, fs/2, 1000
f = np.logspace(np.log10(fmin), np.log10(fmax), num=numf, endpoint=True)
Y = np.column_stack([freqz(yi, 1, worN=f, fs=fs)[1] for yi in y.T])

# critical frequencies
f_alias = c * array_order / 2 / np.pi / R
f_sf = c * sf_order / 2 / np.pi / R


# plots
def add_cbar(ax, im, pad=0.05, width=0.05, **kwarg):
    cax = plt.axes([ax.get_position().xmax + pad, ax.get_position().y0,
                    width, ax.get_position().height], **kwarg)
    plt.colorbar(im, cax=cax)


im_kw = {'cmap': 'Blues', 'vmin': -60, 'vmax': None}
phimin, phimax = np.rad2deg(azi_l[0]), np.rad2deg(azi_l[-1])
phiticks = np.arange(phimin, phimax+90, 90)
tmin = (delay + pre_delay) * 1000
tmax = tmin + (Nfft + Nimp - 1)/fs * 1000
tlim = -1.5, 1.5
flim = fmin, fmax

# Impulse responses
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(db(y), extent=[phimin, phimax, tmin, tmax],
               origin='lower', interpolation='none', **im_kw)
ax.axis('tight')
ax.set_ylim(tlim)
ax.set_xticks(phiticks)
ax.set_xlabel('azimuth in deg')
ax.set_ylabel('time in ms')
add_cbar(ax, im, xlabel='dB')
plt.savefig('./em32-td.png', bbox_inches='tight')

# Transfer functions
fig, ax = plt.subplots(figsize=(4, 4))
phi_deg = np.rad2deg(azi_l)
im = ax.pcolormesh(phi_deg, f, db(Y), **im_kw)
ax.plot(np.zeros_like(f_xo), f_xo, 'k+')
[plt.text(0, fi, '{:0.1f} Hz'.format(fi), va='bottom', ha='left', rotation=30)
 for fi in f_xo]
ax.hlines(f_alias, phimin, phimax, color='r', linestyle='--')
ax.hlines(f_sf, phimin, phimax, color='k', linestyle='--')
ax.axis('tight')
ax.set_xticks(phiticks)
ax.set_ylim(flim)
ax.set_yscale('log')
ax.set_xlabel('azimuth in deg')
ax.set_ylabel('frequency in Hz')
add_cbar(ax, im, xlabel='dB')
plt.savefig('./em32-fd.png', bbox_inches='tight')
