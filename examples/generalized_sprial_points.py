"""Compute the generalized sprial points on a sphere."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import micarray

M = 700
azi, elev, _ = micarray.modal.angular.grid_generalized_spiral(M, C=3.6)

x = np.sin(elev) * np.cos(azi)
y = np.sin(elev) * np.sin(azi)
z = np.cos(elev)

fig = plt.figure(figsize=(8, 8))
ax = fig.gca(projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('$\hat{x}$')
ax.set_ylabel('$\hat{y}$')
ax.set_zlabel('$\hat{z}$')
ax.set_title('Generalized Spiral Points ($M={}$)'.format(M))
