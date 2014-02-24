"""
runclassic400.py

Run script to perform LDA sampling on Classic 400 data.
"""

import numpy as np
import scipy.io
import gibbs

# First, load in the data
datadict = scipy.io.loadmat('data/classic400.mat')

# Extract count data from dict, split into lists
classic400data = datadict['classic400']
doc_idx, voc_idx = classic400data.nonzero()  # load doc, vocab indices
counts = classic400data.data  # load counts

K = 3  # cardinality of topic space
M = 400  # number of documents
V = 6205  # size of the vocabulary
S = len(counts)  # number of nonzero elements in corpus

# Now randomly initialize q,n based on data
q = np.zeros(shape=(S,K), dtype='int')
n = np.zeros(shape=(M,K), dtype='int')

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

# initialize alpha, beta
afv = [0.1, 1.0, 10.0]
bfv = [10.0, 1.0, 0.1]
qfnamev = ['data/c400_q_a0p1_b10p0_K3.dat',
          'data/c400_q_a1p0_b1p0_K3.dat',
          'data/c400_q_a10p0_b0p1_K3.dat']
nfnamev = ['data/c400_n_a0p1_b10p0_K3.dat',
          'data/c400_n_a1p0_b1p0_K3.dat',
          'data/c400_n_a10p0_b0p1_K3.dat']

for af,bf,qfname,nfname in zip(afv,bfv,qfnamev,nfnamev):
    alpha = af*np.ones(K)
    beta = bf*np.ones(V)
    
    # now run 500 epochs
    for nep in range(500):
        q,n = gibbs.gibbs_epoch(q,n,alpha,beta,doc_idx,voc_idx)
    
    # save the results to file
    np.savetxt(qfname,np.array(q),fmt='%d')
    np.savetxt(nfname,np.array(n),fmt='%d')
