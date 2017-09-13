"""
    Compute the plane wave decomposition for an incident broadband plane wave
    on an open spherical array using a modal beamformer of finite order.
"""

import numpy as np
import matplotlib.pyplot as plt
import micarray

N = 20  # order of modal beamformer/microphone array
pw_angle = (np.pi, np.pi/2)  # incidence angle of plane wave
azi_pwd = np.linspace(0, 2*np.pi, 91, endpoint=False)  # angles for plane wave decomposition
k = np.linspace(0.1, 20, 100)  # wavenumber vector
r = 1  # radius of array


def dot_product_sph(v, u):
    # evaluate dot-product between u and v in spherical coordinates
    return (np.cos(v[0])*np.sin(v[1])*np.cos(u[0])*np.sin(u[1]) +
            np.sin(v[0])*np.sin(v[1])*np.sin(u[0])*np.sin(u[1]) +
            np.cos(v[1])*np.cos(u[1]))


# get quadrature grid (microphone positions) of order N
azi, elev, weights = micarray.modal.angular.grid_gauss(N)
# get spherical harmonics matrix for sensors
Y_p = micarray.modal.angular.sht_matrix(N, azi, elev, weights)
# get spherical harmonics matrix for a source ensemble of azimuthal plane waves
Y_q = micarray.modal.angular.sht_matrix(N, azi_pwd, np.pi/2)
# get radial filters
bn = micarray.modal.radial.spherical_pw(N, k, r, setup='open')
dn, _ = micarray.modal.radial.regularize(1/bn, 100, 'softclip')
D = micarray.modal.radial.diagonal_mode_mat(dn)

# compute microphone signals for an incident broad-band plane wave
p = np.exp(1j * k[:, np.newaxis]*r * dot_product_sph((azi, elev), pw_angle))
# compute the plane wave dcomposition
A_pwd = np.matmul(np.matmul(np.conj(Y_q.T), D), Y_p)
q_pwd = np.squeeze(np.matmul(A_pwd, np.expand_dims(p, 2)))
q_pwd_t = np.fft.fftshift(np.fft.irfft(q_pwd, axis=0), axes=0)

# visualize plane wave decomposition (aka beampattern)
plt.figure()
plt.pcolormesh(k, azi_pwd/np.pi, 20*np.log10(np.abs(q_pwd.T)), vmin=-40)
plt.colorbar()
plt.xlabel(r'$kr$')
plt.ylabel(r'$\phi / \pi$')
plt.title('Plane wave docomposition by modal beamformer (frequency domain)')
plt.savefig('modal_open_beamformer_pwd_fd.png')

plt.figure()
plt.pcolormesh(range(2*len(k)-2), azi_pwd/np.pi, 20*np.log10(np.abs(q_pwd_t.T)), vmin=-40)
plt.colorbar()
plt.ylabel(r'$\phi / \pi$')
plt.title('Plane wave docomposition by modal beamformer (time domain)')
plt.savefig('modal_open_beamformer_pwd_td.png')
