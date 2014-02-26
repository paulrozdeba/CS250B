"""
topwords.py

Gets the top words for each topic, based on phi.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import ml_estimate as ml

# load the data
# import data
ndata_b0p1 = np.loadtxt('data/c400_n_a10p0_b0p1_K3.dat')
qdata_b0p1 = np.loadtxt('data/c400_q_a10p0_b0p1_K3.dat')
ndata_b1p0 = np.loadtxt('data/c400_n_a1p0_b1p0_K3.dat')
qdata_b1p0 = np.loadtxt('data/c400_q_a1p0_b1p0_K3.dat')
ndata_b10p0 = np.loadtxt('data/c400_n_a0p1_b10p0_K3.dat')
qdata_b10p0 = np.loadtxt('data/c400_q_a0p1_b10p0_K3.dat')

# import doc,vocab indices
doc_idx,voc_idx = scipy.io.loadmat('data/classic400.mat')['classic400'].nonzero()

K = 3
V = 6205

beta0p1 = 0.1 * np.ones(V)
beta1p0 = 1.0 * np.ones(V)
beta10p0 = 10.0 * np.ones(V)
phi0p1 = ml.calc_phi(qdata_b0p1,beta0p1,voc_idx,V)
phi1p0 = ml.calc_phi(qdata_b1p0,beta1p0,voc_idx,V)
phi10p0 = ml.calc_phi(qdata_b10p0,beta10p0,voc_idx,V)

# get the top 10 words
top1 = np.sort(phi0p1[0])[-10:][::-1]
print top1

# find max probabilities
phimax0p1 = np.max(phi0p1,axis=1)
phimax1p0 = np.max(phi1p0,axis=1)
phimax10p0 = np.max(phi10p0,axis=1)

print phimax0p1
print phimax1p0
print phimax10p0

# plot each topic vector
fig = plt.figure()
ax = fig.add_subplot(111)
# beta = 0.1
ax.plot(np.sort(phi0p1[0])[::-1]/phimax0p1[0],color='red',label=r'$\beta=0.1$')
ax.plot(np.sort(phi0p1[1])[::-1]/phimax0p1[1],color='red')
ax.plot(np.sort(phi0p1[2])[::-1]/phimax0p1[2],color='red')
# beta = 1.0
ax.plot(np.sort(phi1p0[0])[::-1]/phimax1p0[0],color='green',label=r'$\beta=1.0$')
ax.plot(np.sort(phi1p0[1])[::-1]/phimax1p0[1],color='green')
ax.plot(np.sort(phi1p0[2])[::-1]/phimax1p0[2],color='green')
# beta = 10.0
ax.plot(np.sort(phi10p0[0])[::-1]/phimax10p0[0],color='blue',label=r'$\beta=10.0$')
ax.plot(np.sort(phi10p0[1])[::-1]/phimax10p0[1],color='blue')
ax.plot(np.sort(phi10p0[2])[::-1]/phimax10p0[2],color='blue')

ax.set_xlim(-100,6205)
#ax.set_ylim(0,np.max(phi))
ax.set_xlabel(r'Word $v$, by rank')
ax.set_ylabel(r'$\bar{\phi}^{(v)}$')
#ax.set_xlim(-600,6205)
#ax.set_ylim(-np.max(phi)/10,np.max(phi))
#ax.set_yscale('log')
#ax.axhline(y=0, color='black')
#ax.axvline(x=0, color='black')
plt.legend()
plt.show()
