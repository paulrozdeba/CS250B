"""
kosstats.py

Generates some stats and plots to compare to dailykos stats.
"""

import numpy as np
import matplotlib.pyplot as plt

# load data
ndata_K10 = np.loadtxt('data/kos_n_a1p0_b0p5_K10.dat')
ndata_K19 = np.loadtxt('data/kos_n_a1p0_b0p5_K19.dat')

# import the doc,vocab indices
kosdata = np.loadtxt('data/kosdata.dat')
doc_idx = kosdata[:,0]
voc_idx = kosdata[:,1]
f = open('data/vocab.kos.txt', 'r')
vocabulary = f.readlines()
f.close()

# tag frequency from dailykos.com
kos19 = np.array([103048, 68570, 48527, 40034, 37733, 36722, 34292, 34257,
                  32082, 31689, 31627, 30498, 29139, 28879, 28853, 27137,
                  25101, 23684, 23037], dtype='float')

# sum over the columns
topicfreq_K10 = np.sort(np.sum(ndata_K10,axis=0))[::-1]
topicfreq_K19 = np.sort(np.sum(ndata_K19,axis=0))[::-1]
topicfreq_K10 /= topicfreq_K10[0]
topicfreq_K19 /= topicfreq_K19[0]

#kos10 = kos19[:10]*(topicfreq_K10[0]/kos19[0])
#kos19 = kos19 * (topicfreq_K19[0]/kos19[0])
kos10 = kos19[:10]/kos19[0]
kos19 /= kos19[0]

print topicfreq_K10/kos19[:10]
print topicfreq_K19/kos19

# plots histograms
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(len(topicfreq_K10)),topicfreq_K10,color='blue',label='LDA')
ax.scatter(range(len(kos10)),kos10,color='red',label='dailykos.com')
ax.set_xlabel('Topic')
ax.set_ylabel('Frequency (rescaled)')
ax.set_title(r'$K=10$')
ax.set_xlim(-1,11)
plt.legend()
plt.show()
