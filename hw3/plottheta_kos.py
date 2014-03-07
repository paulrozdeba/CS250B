"""
plotdist.py

Plots theta/phi distribution in 3d, as well as the plane x1+x2+x3=1.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ml_estimate as ml

# import data
ndata = np.loadtxt('data/kos_n_a1p0_b0p5_K2.dat')
qdata = np.loadtxt('data/kos_q_a1p0_b0p5_K2.dat')

# import doc,vocab indices
kosdata_full = np.loadtxt('data/docword.kos.txt', dtype='float', skiprows=3)
kosdata_full[:,:2] -= 1  # docs and vocs were originally numbered starting with 1

M = 3430  # number of documents
V = 6906  # size of vocabulary

# select only enough documents st S is approximately 50,000
for d in range(M):
    if d == 0:
        kosdata = kosdata_full[kosdata_full[:,0]==0.0]
        c = np.sum(kosdata[:,2])
    else:
        kosdata = np.vstack((kosdata,kosdata_full[kosdata_full[:,0]==float(d)]))
        c = np.sum(kosdata[:,2])
        if c > 25000:
            break

M = int(kosdata[-1,0])  # reset number of documents

# split into counts and indices
doc_idx = kosdata[:,0]
voc_idx = kosdata[:,1]
counts  = kosdata[:,2]

S = len(counts)  # number of nonzero elements

K = 2
alpha = 1.0*np.ones(K)
theta = ml.calc_theta(ndata,alpha)

# now plot distribution
fig = plt.figure()

# 2D
ax = fig.add_subplot(111)
ax.plot(theta[:,0],theta[:,1], '.', lw=0, ms=5)

# 3D
"""
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
"""

plt.show()
