"""
plotdist.py

Plots theta/phi distribution in 3d, as well as the plane x1+x2+x3=1.
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ml_estimate as ml

# import data
ndata = np.loadtxt('data/c400_n_a10p0_b10p0_K3.dat')
qdata = np.loadtxt('data/c400_q_a10p0_b10p0_K3.dat')

# import doc,vocab indices
doc_idx,voc_idx = scipy.io.loadmat('data/classic400.mat')['classic400'].nonzero()

K = 3
alpha = 10.0*np.ones(K)
theta = ml.calc_theta(ndata,alpha)

# now plot distribution
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(theta[:,0],theta[:,1],theta[:,2], '.', lw=0, ms=5)

# plot sheet
Npts = 200
xx,yy = np.meshgrid(np.linspace(0.0,1.0,Npts),np.linspace(0.0,1.0,Npts))

zz = 1.0 - xx - yy
for i,z1 in enumerate(zz):
    for j,z2 in enumerate(z1):
        if z2 < 0:
            zz[i,j] = None

ax.plot_surface(xx,yy,zz, alpha=0.2, color='gray', linewidth=0)

ax.set_xlim((0,1))
ax.set_ylim((0,1))
ax.set_zlim((0,1))

plt.show()
