"""
runkos.py

Runs topic modeling on dailykos.com data set (source: UCI MLR)
"""

import numpy as np
import scipy.io
import gibbs
import sys

# Load in the data
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

# initialize hyperparameters
#K = 3
K = int(sys.argv[1])
alpha = 1.0 * np.ones(K)
beta  = 0.5 * np.ones(V)

# Now randomly initialize q,n based on data
q = np.zeros(shape=(S,K), dtype='int')
n = np.zeros(shape=(M+1,K), dtype='int')

for bi,(m,count) in enumerate(zip(doc_idx,counts)):
    # To randomly assign topics, draw (K-1) ints from the uniform distribution 
    # over the interval [0,count).  The length of each sub-interval is the topic
    # count assigned to that element of the matrix (note that this
    # automatically includes the possibility of zero counts).
    draws = np.sort(np.append(np.array([0]),np.random.randint(0, count, K-1)))
    subints = np.array([draws[i+1] - draws[i] for i in range(len(draws)-1)])
    subints = np.append(subints, np.array(count-draws[i]))
    np.random.shuffle(subints)
    for zi,subint in enumerate(subints):
        q[bi,zi] += subint
        n[m,zi] += subint

for nep in range(500):
    q,n = gibbs.gibbs_epoch(q,n,alpha,beta,doc_idx,voc_idx)

qfname = 'data/kos_q_a1p0_b0p5_K'+str(K)+'.dat'
nfname = 'data/kos_n_a1p0_b0p5_K'+str(K)+'.dat'
np.savetxt(qfname,np.array(q),fmt='%d')
np.savetxt(nfname,np.array(n),fmt='%d')
