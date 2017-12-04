"""Compute the generalized sprial points on a sphere."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import micarray
from micarray.util import db


def sph2cart(alpha, beta, r):
    """Spherical to cartesian coordinates."""
    x = r * np.cos(alpha) * np.sin(beta)
    y = r * np.sin(alpha) * np.sin(beta)
    z = r * np.cos(beta)
    return x, y, z


# Microphone array
R = 0.5  # radius
N = 6  # modal bandwidth
M = 1*(N+1)**2  # number of microphones
azi, elev, _ = micarray.modal.angular.grid_generalized_spiral(M, C=3.6)
x, y, z = sph2cart(azi, elev, R)
Y = micarray.modal.angular.sht_matrix(N, azi, elev)  # synthesis matrix
Y_inv = np.linalg.pinv(Y)  # analysis matrix
k = np.linspace(0.1, 40, 100)  # wavenumber
bn = micarray.modal.radial.spherical_pw(N, k, R, setup='open')
D = micarray.modal.radial.diagonal_mode_mat(bn)
B = np.matmul(D, Y)
condnum = np.linalg.cond(B)  # Condition number


# Fig. Microphone array
fig = plt.figure(figsize=(8, 8))
ax = fig.gca(projection='3d')
ax.plot(x, y, z, c='lightgray')
ax.scatter(x, y, z)
ax.set_xlabel('$\hat{x}$')
ax.set_ylabel('$\hat{y}$')
ax.set_zlabel('$\hat{z}$')
ax.set_title('Generalized Spiral Points ($M={}$)'.format(M))

# Fig. Pseudo inverse matrix
plt.figure()
plt.pcolormesh(db(np.matmul(Y_inv, Y)))
plt.axis('scaled')
plt.colorbar(label='dB')
plt.title(r'$E = Y^\dagger Y$')

# Fig. Condition number
plt.figure()
plt.semilogy(k*R, condnum)
plt.xlabel('$kr$')
plt.ylabel('Condition number')
plt.ylim(top=10e4)
