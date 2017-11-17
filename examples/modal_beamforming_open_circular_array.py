"""
    Compute the plane wave decomposition for an incident broadband plane wave
    on an open circular array using a modal beamformer of finite order.
"""

import numpy as np
import matplotlib.pyplot as plt
import micarray

N = 90  # order of modal beamformer/microphone array
pw_angle = 1.23 * np.pi  # incidence angle of plane wave
pol_pwd = np.linspace(0, 2*np.pi, 91, endpoint=False)  # angles for plane wave decomposition
k = np.linspace(0.1, 20, 100)  # wavenumber vector
r = 1  # radius of array

# get uniform grid (microphone positions) of order N
pol, weights = micarray.modal.angular.grid_equal_polar_angle(N)
# get circular harmonics matrix for sensors
Psi_p = micarray.modal.angular.cht_matrix(N, pol, weights)
# get circular harmonics matrix for a source ensemble of azimuthal plane wave
Psi_q = micarray.modal.angular.cht_matrix(N, pol_pwd)
# get radial filters
Bn = micarray.modal.radial.circular_pw(N, k, r, setup='open')
Dn, _ = micarray.modal.radial.regularize(1/Bn, 100, 'softclip')
D = micarray.modal.radial.circ_diagonal_mode_mat(Dn)

# compute microphone signals for an incident broad-band plane wave
p = np.exp(1j * k[:, np.newaxis]*r * np.cos(pol - pw_angle))
# compute plane wave decomposition
A_pwd = np.matmul(np.matmul(np.conj(Psi_q.T), D), Psi_p)
q_pwd = np.squeeze(np.matmul(A_pwd, np.expand_dims(p, 2)))
q_pwd_t = np.fft.fftshift(np.fft.irfft(q_pwd, axis=0), axes=0)

# visualize plane wave decomposition (aka beampattern)
plt.figure()
plt.pcolormesh(k, pol_pwd/np.pi, micarray.util.db(q_pwd.T), vmin=-40)
plt.colorbar()
plt.xlabel(r'$kr$')
plt.ylabel(r'$\phi / \pi$')
plt.title('Plane wave docomposition by modal beamformer (frequency domain)')
plt.savefig('modal_open_beamformer_pwd_fd.png')

plt.figure()
plt.pcolormesh(range(2*len(k)-2), pol_pwd/np.pi, micarray.util.db(q_pwd_t.T), vmin=-40)
plt.colorbar()
plt.ylabel(r'$\phi / \pi$')
plt.title('Plane wave docomposition by modal beamformer (time domain)')
plt.savefig('modal_open_beamformer_pwd_td.png')
