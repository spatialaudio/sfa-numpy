"""Cross-over frequencies for linear-phase filterbank.

- filter bank design for Ambisonic encoding of rigid sphere signals
- determine cross-over frequencies based on pre-defined maximum boost
- exploit small argument approximation of spherical Hankel functions

    Reference
    ---------
    Franz Zotter, "A linear-phase filter-bank approach to process
    rigid spherical microphone array recordings", in Proc. IcETRAN,
    Palic, Serbia, 2018. (see Fig. 6)
"""
import numpy as np
import matplotlib.pyplot as plt
from micarray.util import maxre_sph, double_factorial, modal_norm, db
from micarray.modal.radial import crossover_frequencies, spherical_hn2

c = 343
N = 4
R = 0.049
max_boost = 30
f_xo = crossover_frequencies(N, R, max_boost, modal_weight=maxre_sph)
band_gain = [1 / modal_norm(maxre_sph(n)) for n in range(N+1)]

fmin, fmax, numf = 10, 20000, 2000
f = np.logspace(np.log10(fmin), np.log10(fmax), num=numf)
kr = 2 * np.pi * f / c * R

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
for n in range(N+1):

    # Analytic radial filter
    radfilt = band_gain[n] * (kr)**2 * spherical_hn2(n, kr, derivative=True)
    ax.semilogx(f, db(radfilt), lw=3, alpha=0.5, zorder=0,
                label='${}$'.format(n))

    if n > 0:
        fn = f_xo[n-1]
        krn = 2 * np.pi * fn / c * R

        # Low-frequency (small argument) approximation
        lf_approx = band_gain[n] * double_factorial(2*n-1) * (n+1) / (kr)**n
        gain_at_crossover = \
            band_gain[n] * (krn)**2 * spherical_hn2(n, krn, derivative=True)

        ax.semilogx(f, db(lf_approx), c='black', ls=':')
        ax.semilogx(fn, db(gain_at_crossover), 'C0o')
        ax.text(fn, db(gain_at_crossover), '{:3.1f} Hz'.format(fn),
                ha='left', va='bottom', rotation=55)
ax.hlines(max_boost, xmin=fmin, xmax=fmax,
          colors='C3', linestyle='--', label='max. boost')
ax.set_xlim(fmin, fmax)
ax.set_ylim(-10, 90)
ax.grid(True)
ax.set_xlabel('frequency in Hz')
ax.set_ylabel('magnitude in dB')
ax.legend(title='Order', ncol=2)
plt.savefig('./crossover-frequencies.png', bbox_inches='tight')
