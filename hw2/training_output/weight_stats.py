"""
weight_stats.py

Plots weights from Collins and SGD training methods, and does some statistics.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# first load the weights from all 3 data set partitions
collins_0p25 = np.load('collins_w_0p25.npy')
collins_0p5 = np.load('collins_w_0p5.npy')
collins_0p75 = np.load('collins_w_0p75.npy')

sgd_0p25 = np.load('sgd_w_0p25.npy')
sgd_0p5 = np.load('sgd_w_0p5.npy')
sgd_0p75 = np.load('sgd_w_0p75.npy')

# now calculate pearson correlation
pc_0p25 = stats.pearsonr(collins_0p25, sgd_0p25)
pc_0p5 = stats.pearsonr(collins_0p5, sgd_0p5)
pc_0p75 = stats.pearsonr(collins_0p75, sgd_0p75)

print 'Pearson correlation:\n'
print '25\%: ',pc_0p25
print '50\%: ',pc_0p5
print '75\%: ',pc_0p75

filenames = ['scatter_0p25.pdf','scatter_0p5.pdf','scatter_0p75.pdf']
titles = [r'.25/.75 training/validation', r'.50/.50 training/validation', 
            r'.75/.25 training/validation']

# plot scatter
for fname,tit,coll,sgd in zip(filenames, titles, [collins_0p25,collins_0p5,collins_0p75],[sgd_0p25,sgd_0p5,sgd_0p75]):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(sgd, coll, s=1)
    ax.set_xlabel(r'$\bar{w}$ (SGD)')
    ax.set_ylabel(r'$\bar{w}$ (Collins)')
    ax.set_title(tit)
    fig.savefig(fname)
    plt.close(fig)
