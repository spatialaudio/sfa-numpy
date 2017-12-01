"""
    Compute the plane wave decomposition for an incident broadband plane wave
    on an rigid spherical array using a modal beamformer of finite order.
"""

import numpy as np
import matplotlib.pyplot as plt
import micarray

N = 20  # order of modal beamformer/microphone array
azi_pw = np.pi  # incidence angle of plane wave
azi_pwd = np.linspace(0, 2*np.pi, 91, endpoint=False)  # angles for plane wave decomposition
k = np.linspace(0, 20, 100)  # wavenumber
r = 1  # radius of array


# get quadrature grid (microphone positions) of order N
azi, elev, weights = micarray.modal.angular.grid_gauss(N)

# pressure on the surface of a rigid sphere for an incident plane wave
bn = micarray.modal.radial.spherical_pw(N, k, r, setup='rigid')
D = micarray.modal.radial.diagonal_mode_mat(bn)
Y_p = micarray.modal.angular.sht_matrix(N, azi, elev)
Y_pw = micarray.modal.angular.sht_matrix(N, azi_pw, np.pi/2)
p = np.matmul(np.matmul(np.conj(Y_pw.T), D), Y_p)
p = np.squeeze(p)

# plane wave decomposition using modal beamforming
Y_p = micarray.modal.angular.sht_matrix(N, azi, elev, weights)
# get SHT matrix for a source ensemble of azimuthal plane waves
azi_pwd = np.linspace(0, 2*np.pi, 91, endpoint=False)
Y_q = micarray.modal.angular.sht_matrix(N, azi_pwd, np.pi/2)
# get radial filters
bn = micarray.modal.radial.spherical_pw(N, k, r, setup='rigid')
bn = micarray.modal.radial.replace_zeros(bn, k*r)
dn, _ = micarray.modal.radial.regularize(1/bn, 100, 'softclip')
D = micarray.modal.radial.diagonal_mode_mat(dn)
# compute the PWD
A_mb = np.matmul(np.matmul(np.conj(Y_q.T), D), Y_p)
q_mb = np.squeeze(np.matmul(A_mb, np.expand_dims(p, 2)))
q_mb_t = np.fft.fftshift(np.fft.irfft(q_mb, axis=0), axes=0)


# visualize plane wave decomposition (aka beampattern)
plt.figure()
plt.pcolormesh(k*r, azi_pwd/np.pi, 20*np.log10(np.abs(q_mb.T)), vmin=-40)
plt.colorbar()
plt.xlabel(r'$kr$')
plt.ylabel(r'$\phi / \pi$')
plt.title('Plane wave docomposition by modal beamformer (frequency domain)')
plt.savefig('modal_rigid_beamformer_pwd_fd.png')

plt.figure()
plt.pcolormesh(range(2*len(k)-2), azi_pwd/np.pi, 20*np.log10(np.abs(q_mb_t.T)), vmin=-40)
plt.colorbar()
plt.ylabel(r'$\phi / \pi$')
plt.title('Plane wave docomposition by modal beamformer (time domain)')
plt.savefig('modal_rigid_beamformer_pwd_td.png')
