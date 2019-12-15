"""Noise amplification.

    Reference
    ---------
    Franz Zotter, "A linear-phase filter-bank approach to process
    rigid spherical microphone array recordings", in Proc. IcETRAN,
    Palic, Serbia, 2018. (see Fig. 11)
"""
import numpy as np
import matplotlib.pyplot as plt
from micarray.util import maxre_sph, modal_norm, db
from micarray.modal.radial import spherical_hn2, tf_linph_filterbank,\
                                  crossover_frequencies

c = 343
R = 0.042
N = 4

fmin, fmax, numf = 10, 20000, 500
f = np.logspace(np.log10(fmin), np.log10(fmax), num=numf)
kr = 2 * np.pi * f / c * R

Max_boost = 0, 5, 10, 15, 20
Noise_amp = np.zeros((numf, len(Max_boost)))
Freq_xo = np.zeros((N, len(Max_boost)))

# Prototype radial filters
H_proto = np.stack([1j**(-n-1) * (kr)**2
                    * spherical_hn2(n, kr, derivative=True)
                    for n in range(N+1)])

for k, max_boost in enumerate(Max_boost):
    f_xo = crossover_frequencies(N, R, max_boost)
    Freq_xo[:, k] = f_xo
    H_fbank = tf_linph_filterbank(f_xo, f, type='butter')

    H_radial = np.zeros_like(H_proto)
    for i, Hi in enumerate(H_fbank):
        ai = maxre_sph(i)
        ai *= 1 / modal_norm(ai)
        for n, Hr in enumerate(H_proto[:i+1]):
            H_radial[n] += Hr * Hi * ai[n]
    Noise_amp[:, k] = (modal_norm(np.abs(H_radial.T)) / np.abs(H_proto[0]))**2

# Plot
fig, ax = plt.subplots()
ax.semilogx(f, db(Noise_amp, power=True))
ax.set_xlim(fmin, fmax)
ax.set_ylim(-3, 23)
ax.set_xlabel('Frequency / Hz')
ax.set_ylabel('Magnitude / dB')
ax.grid(True)
ax.legend(Max_boost, title='max. boost / dB')
plt.savefig('./noise-amplification.png')
