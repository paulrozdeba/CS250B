"""
plotdist.py

Plots theta/phi distribution in 3d, as well as the plane x1+x2+x3=1.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ml_estimate as ml


# import data
ndata = np.loadtxt('data/classic/c400_n_a0p1_b10000p0_K3.dat')
qdata = np.loadtxt('data//classic/c400_q_a0p1_b10000p0_K3.dat')
#color code the true labels
truelabels = np.loadtxt('data/classic400_truelabels.dat')
truelabelcolors = []
for i in range(400):
    if (truelabels[i] ==1):
        truelabelcolors.append('r')
    elif (truelabels[i] ==2):
        truelabelcolors.append('b')
    else:
        truelabelcolors.append('g')
K = 3
alpha = 0.1*np.ones(K)
theta = ml.calc_theta(ndata,alpha)

# now plot distribution
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(theta[:,0],theta[:,1],theta[:,2],lw=0,c = truelabelcolors)

xx,yy = np.meshgrid(np.linspace(0.0,1.0,200),np.linspace(0.0,1.0,200))
zz = 1.0 - xx - yy
for i,z1 in enumerate(zz):
    for j,z2 in enumerate(z1):
        if z2 < 0:
            zz[i,j] = None

ax.plot_surface(xx,yy,zz, alpha=0.2, color='gray', linewidth=0)

ax.set_xlim((0.0,1.0))
ax.set_ylim((0.0,1.0))
ax.set_zlim((0.0,1.0))

ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
ax.set_zlabel(r'$\theta_3$')
ax.view_init(23,82)
plt.title(r'$\alpha = 0.1,\beta = 10000.0$')

plt.show()