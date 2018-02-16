"""
    Compute the plane wave decomposition for an incident broadband plane wave
    on a rigid circular array using a modal beamformer of finite order.
"""

import numpy as np
import matplotlib.pyplot as plt
import micarray
from micarray.util import db

Nsf = 50  # order of the incident sound field
N = 30  # order of modal beamformer/microphone array
pw_angle = 1 * np.pi  # incidence angle of plane wave
pol_pwd = np.linspace(0, 2*np.pi, 180, endpoint=False)  # angles for PWD
k = np.linspace(0, 20, 100)  # wavenumber vector
r = 1  # radius of array

# get uniform grid (microphone positions) of order N
pol, weights = micarray.modal.angular.grid_equal_polar_angle(N)

# pressure on the surface of a rigid cylinder for an incident plane wave
Bn = micarray.modal.radial.circular_pw(Nsf, k, r, setup='rigid')
D = micarray.modal.radial.circ_diagonal_mode_mat(Bn)
Psi_p = micarray.modal.angular.cht_matrix(Nsf, pol)
Psi_pw = micarray.modal.angular.cht_matrix(Nsf, pw_angle)
p = np.matmul(np.matmul(Psi_p, D), np.conj(Psi_pw.T))
p = np.squeeze(p)

# plane wave decomposition using modal beamforming
Psi_p = micarray.modal.angular.cht_matrix(N, pol, weights)
Psi_q = micarray.modal.angular.cht_matrix(N, pol_pwd)
Bn = micarray.modal.radial.circular_pw(N, k, r, setup='rigid')
Dn, _ = micarray.modal.radial.regularize(1/Bn, 100, 'softclip')
D = micarray.modal.radial.circ_diagonal_mode_mat(Dn)
A_pwd = np.matmul(np.matmul(Psi_q, D), np.conj(Psi_p.T))
q_pwd = np.squeeze(np.matmul(A_pwd, np.expand_dims(p, 2)))
q_pwd_t = np.fft.fftshift(np.fft.irfft(q_pwd, axis=0), axes=0)

# visualize plane wave decomposition (aka beampattern)
plt.figure()
plt.pcolormesh(k, pol_pwd/np.pi, db(q_pwd.T), vmin=-40)
plt.colorbar()
plt.xlabel(r'$kr$')
plt.ylabel(r'$\phi / \pi$')
plt.title('Plane wave docomposition by modal beamformer (frequency domain)')
plt.savefig('modal_beamforming_rigid_circular_array_fd.png')

plt.figure()
plt.pcolormesh(range(2*len(k)-2), pol_pwd/np.pi, db(q_pwd_t.T), vmin=-40)
plt.colorbar()
plt.ylabel(r'$\phi / \pi$')
plt.title('Plane wave docomposition by modal beamformer (time domain)')
plt.savefig('modal_beamforming_rigid_circular_array_td.png')
