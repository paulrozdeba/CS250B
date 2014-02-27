"""
topwords.py

Gets the top words for each topic, based on phi.
"""

import numpy as np
import matplotlib.pyplot as plt

import ml_estimate as ml

# load the data
# import data
ndata_K2 = np.loadtxt('data/kos_n_a1p0_b1p0_K2.dat')
qdata_K2 = np.loadtxt('data/kos_q_a1p0_b1p0_K2.dat')
ndata_K3 = np.loadtxt('data/kos_n_a1p0_b1p0_K3.dat')
qdata_K3 = np.loadtxt('data/kos_q_a1p0_b1p0_K3.dat')
ndata_K10 = np.loadtxt('data/kos_n_a1p0_b1p0_K10.dat')
qdata_K10 = np.loadtxt('data/kos_q_a1p0_b1p0_K10.dat')
ndata_K19 = np.loadtxt('data/kos_n_a1p0_b1p0_K19.dat')
qdata_K19 = np.loadtxt('data/kos_q_a1p0_b1p0_K19.dat')

# import the doc,vocab indices
kosdata = np.loadtxt('data/kosdata.dat')
doc_idx = kosdata[:,0]
voc_idx = kosdata[:,1]

V = 6906

beta = 0.5 * np.ones(V)
phiK2 = ml.calc_phi(qdata_K2,beta,voc_idx,V)
phiK3 = ml.calc_phi(qdata_K3,beta,voc_idx,V)
phiK10 = ml.calc_phi(qdata_K10,beta,voc_idx,V)
phiK19 = ml.calc_phi(qdata_K19,beta,voc_idx,V)

# get the top 10 words
topK2 = np.sort(phiK2,axis=1)[:,-10:][:,::-1]
topK3 = np.sort(phiK3,axis=1)[:,-10:][:,::-1]
topK10 = np.sort(phiK10,axis=1)[:,-10:][:,::-1]
topK19 = np.sort(phiK19,axis=1)[:,-10:][:,::-1]

# find max probabilities
phimaxK2 = topK2[:,0]
phimaxK3 = topK3[:,0]
phimaxK10 = topK10[:,0]
phimaxK19 = topK19[:,0]

# plot each topic vector
fig = plt.figure()
ax = fig.add_subplot(111)

# K = 2
for i,(phi,phimax) in enumerate(zip(phiK2,phimaxK2)):
    if i==0:
        ax.loglog(np.sort(phi)[::-1]/phimax,color='red',label=r'$K=2$')
    else:
        ax.loglog(np.sort(phi)[::-1]/phimax,color='red')
# K = 3
for i,(phi,phimax) in enumerate(zip(phiK3,phimaxK3)):
    if i==0:
        ax.loglog(np.sort(phi)[::-1]/phimax,color='blue',label=r'$K=3$')
    else:
        ax.loglog(np.sort(phi)[::-1]/phimax,color='blue')
# K = 10
for i,(phi,phimax) in enumerate(zip(phiK10,phimaxK10)):
    if i==0:
        ax.loglog(np.sort(phi)[::-1]/phimax,color='green',label=r'$K=10$')
    else:
        ax.loglog(np.sort(phi)[::-1]/phimax,color='green')
# K = 19
for i,(phi,phimax) in enumerate(zip(phiK19,phimaxK19)):
    if i==0:
        ax.loglog(np.sort(phi)[::-1]/phimax,color='violet',label=r'$K=19$')
    else:
        ax.loglog(np.sort(phi)[::-1]/phimax,color='violet')

ax.set_xlim(-100,6906)
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
