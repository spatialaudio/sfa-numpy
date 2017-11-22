"""
    Modal analysis and extrapolation of a monochromatic sound field
    in the cricular harmonics domain using an open circular array
"""
import numpy as np
import micarray
from micarray.util import db
import scipy.special as special
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# Constants
c = 343  # speed of sound [m/s]

# 2-dimensional grid
spacing = 0.01
x = np.expand_dims(np.arange(-1, 1, spacing), axis=0)
y = np.expand_dims(np.arange(-1, 1, spacing), axis=1)
r = np.sqrt(x**2+y**2).astype(complex)
phi = np.arctan2(y, x)

# Incident plane wave
phi_pw = 0.5*np.pi  # incoming direction
f = 1500  # temporal frequency
k = micarray.util.asarray_1d(2*np.pi*f/c)  # corresponding wave number
s0 = np.exp(1j*k*r*np.cos(phi-phi_pw))  # incident sound field

# Microphone array and modal analysis
N = 15  # maximum order
order = np.roll(np.arange(-N, N+1), -N)
threshold = 1e5  # regulaization parameter
R = 0.5  # radius
Phi, weights = micarray.modal.angular.grid_equal_polar_angle(N)  # array
p = np.exp(1j*k*R*np.cos(Phi-phi_pw))  # captured signal
bn = micarray.modal.radial.circ_radial_weights(N, k*R, setup='open')
dn, _ = micarray.modal.radial.regularize(1/bn, threshold, 'softclip')
pm = dn * np.fft.ifft(p)

# Sound field extrapolation
basis = special.jn(order[:, np.newaxis, np.newaxis], k * r[np.newaxis, :, :]) \
        * np.exp(-1j*order[:, np.newaxis, np.newaxis] * phi[np.newaxis, :, :])
s = np.tensordot(pm, basis, axes=[0, 0])


# Plots
plt.figure(figsize=(4, 4))
plt.pcolormesh(x, y, np.real(s), cmap='coolwarm')
plt.plot(R*np.cos(Phi), R*np.sin(Phi), 'k.')
plt.axis('scaled')
plt.axis([-1, 1, -1, 1])
cb = plt.colorbar(fraction=0.046, pad=0.04)
plt.clim(-1, 1)
plt.xlabel('$x$ / m')
plt.ylabel('$y$ / m')
plt.title('Extrapolated Sound Field')
plt.savefig('extrapolation_open_circ_mono.png')

plt.figure(figsize=(4, 4))
plt.pcolormesh(x, y, db(s0-s), cmap='Blues', vmin=-60)
plt.plot(R*np.cos(Phi), R*np.sin(Phi), 'k.')
plt.axis('scaled')
plt.axis([-1, 1, -1, 1])
cb = plt.colorbar(fraction=0.046, pad=0.04)
cb.set_label('dB')
plt.clim(-60, 0)
plt.xlabel('$x$ / m')
plt.ylabel('$y$ / m')
xx, yy = np.meshgrid(x, y)
cs = plt.contour(xx, yy, db(s0-s), np.arange(-60, 20, 20), colors='orange')
plt.clabel(cs, fontsize=9, inline=1, fmt='%1.0f')
plt.title('Extrapolation Error')
plt.savefig('extrapolation_error_open_circ_mono.png')
