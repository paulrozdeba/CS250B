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
ndata_1 = np.loadtxt('data/c400_n_a0p1_b10000p0_K3.dat')
qdata_1 = np.loadtxt('data/c400_q_a0p1_b10000p0_K3.dat')
ndata_2 = np.loadtxt('data/c400_n_a0p1_b0p1_K3.dat')
qdata_2 = np.loadtxt('data/c400_q_a0p1_b0p1_K3.dat')
ndata_3 = np.loadtxt('data/c400_n_a10p0_b0p1_K3.dat')
qdata_3 = np.loadtxt('data/c400_q_a10p0_b0p1_K3.dat')
ndata_4 = np.loadtxt('data/c400_n_a1p0_b0p1_K3.dat')
qdata_4 = np.loadtxt('data/c400_q_a1p0_b0p1_K3.dat')
ndata_5 = np.loadtxt('data/c400_n_a10p0_b100p0_K3.dat')
qdata_5 = np.loadtxt('data/c400_q_a10p0_b100p0_K3.dat')

# import doc,vocab indices
c400data = scipy.io.loadmat('data/classic400.mat')
doc_idx,voc_idx = c400data['classic400'].nonzero()
# import vocabulary
vocabulary = []
for i,entry in enumerate(c400data['classicwordlist']):
    vocabulary.append(entry[0][0])

K = 3
V = 6205

beta1 = 10000.0 * np.ones(V)
beta2 = 0.1 * np.ones(V)
beta3 = 0.1 * np.ones(V)
beta4 = 0.1 * np.ones(V)
beta5 = 100.0 * np.ones(V)
phi1 = ml.calc_phi(qdata_1,beta1,voc_idx,V)
phi2 = ml.calc_phi(qdata_2,beta2,voc_idx,V)
phi3 = ml.calc_phi(qdata_3,beta3,voc_idx,V)
phi4 = ml.calc_phi(qdata_4,beta4,voc_idx,V)
phi5 = ml.calc_phi(qdata_5,beta5,voc_idx,V)

# get the top 20 words
top1 = np.sort(phi1,axis=1)[:,-20:][:,::-1]
top1_ind = np.argsort(phi1,axis=1)[:,-20:][:,::-1]
top2 = np.sort(phi2,axis=1)[:,-20:][:,::-1]
top2_ind = np.argsort(phi2,axis=1)[:,-20:][:,::-1]
top3 = np.sort(phi3,axis=1)[:,-20:][:,::-1]
top3_ind = np.argsort(phi3,axis=1)[:,-20:][:,::-1]
top4 = np.sort(phi4,axis=1)[:,-20:][:,::-1]
top4_ind = np.argsort(phi4,axis=1)[:,-20:][:,::-1]
top5 = np.sort(phi5,axis=1)[:,-20:][:,::-1]
top5_ind = np.argsort(phi5,axis=1)[:,-20:][:,::-1]

# print top words to file
f = open('data/topwords_c400_K3.txt', 'w')
f.write('a=0.1, b=10000.0:\n')
for i,plist in enumerate(top1):
    f.write('\nTopic:\n')
    for j,p in enumerate(plist):
        f.write(vocabulary[top1_ind[i,j]] + '\n')
f.write('a=0.1, b=0.1:\n')
for i,plist in enumerate(top2):
    f.write('\nTopic:\n')
    for j,p in enumerate(plist):
        f.write(vocabulary[top2_ind[i,j]] + '\n')
f.write('a=10.0, b=0.1:\n')
for i,plist in enumerate(top3):
    f.write('\nTopic:\n')
    for j,p in enumerate(plist):
        f.write(vocabulary[top3_ind[i,j]] +'\n')
f.write('a=1.0, b=0.1:\n')
for i,plist in enumerate(top4):
    f.write('\nTopic:\n')
    for j,p in enumerate(plist):
        f.write(vocabulary[top4_ind[i,j]] + '\n')
f.write('a=10.0, b=100.0:\n')
for i,plist in enumerate(top5):
    f.write('\nTopic:\n')
    for j,p in enumerate(plist):
        f.write(vocabulary[top5_ind[i,j]] + '\n')
exit(0)
"""
# find max probabilities
phimax1 = np.max(phi0p1,axis=1)
phimax2 = np.max(phi1p0,axis=1)
phimax3 = np.max(phi10p0,axis=1)

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
"""

