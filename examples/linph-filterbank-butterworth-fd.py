"""Linear-phase filterbank.

- The target magnitude responses fo the filter-bank is designed
  by using the zero-phase Butterworth responses.
  (not to confused with typical (minphase) Butterworth filters)

    Reference
    ---------
    Franz Zotter, "A linear-phase filter-bank approach to process
    rigid spherical microphone array recordings", in Proc. IcETRAN,
    Palic, Serbia, 2018. (see Fig. 7)
"""
import numpy as np
import matplotlib.pyplot as plt
from micarray.util import maxre_sph, modal_norm, db
from micarray.modal.radial \
    import crossover_frequencies, spherical_hn2, tf_linph_filterbank

c = 343
R = 0.049
N = 4
max_boost = 30
f_xo = crossover_frequencies(N, R, max_boost, modal_weight=maxre_sph)

fmin, fmax, numf = 10, 20000, 2000
f = np.logspace(np.log10(fmin), np.log10(fmax), num=numf)
kr = 2 * np.pi * f / c * R

H_fbank = tf_linph_filterbank(f_xo, f, type='butter')
H_tot = np.sum(H_fbank, axis=0)

# Prototpye radial filters
H_proto = np.stack([
        1j**(-n-1) * (kr)**2 * spherical_hn2(n, kr, derivative=True)
        for n in range(N+1)])

# Equalized radial filters
H_radial = np.zeros_like(H_proto)
for i, Hi in enumerate(H_fbank):
    ai = maxre_sph(i)
    ai *= 1 / modal_norm(ai)
    for n, Hn in enumerate(H_proto[:i+1]):
        H_radial[n] += Hi * ai[n]
H_radial *= modal_norm(maxre_sph(N))


# Plots
# Filter-bank
fig, ax = plt.subplots()
for i, Hi in enumerate(H_fbank):
    ax.semilogx(f, db(Hi), lw=3, label='${}$'.format(i), alpha=0.5)
for fx in f_xo:
    ax.semilogx(fx, 0, 'kv')
    ax.text(fx, 0, '{:0.1f} Hz'.format(fx), rotation=30,
            horizontalalignment='left', verticalalignment='bottom')
ax.semilogx(f, db(H_tot), 'k:', label='Sum')
ax.set_xlim(fmin, fmax)
ax.set_ylim(-100, 12)
ax.set_xscale('log')
ax.grid(True)
ax.set_xlabel('frequency in Hz')
ax.set_ylabel('magnitude in dB')
ax.legend(title='subband')
plt.savefig('./linph-filterbank-fd.png', bbox_inches='tight')

# Normalized radial filters
fig, ax = plt.subplots()
for n, (Hr, Hp) in enumerate(zip(H_radial, H_proto)):
    ax.semilogx(f, db(Hp), c='k', ls=':')
    ax.semilogx(f, db(Hr * Hp), lw=3, label='${}$'.format(n), alpha=0.5)
ax.hlines(max_boost, xmin=fmin, xmax=fmax, colors='C3', linestyle='--',
          label='max. boost')
ax.set_xlim(fmin, fmax)
ax.set_ylim(-23, 33)
ax.set_xscale('log')
ax.grid(True)
ax.set_xlabel('frequency in Hz')
ax.set_ylabel('magnitude in dB')
ax.legend(title='order', loc='lower right', ncol=2)
plt.savefig('./linph-filterbank-butterworth-fd.png', bbox_inches='tight')
