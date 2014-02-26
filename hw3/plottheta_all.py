"""
plottheta_all.py

Plots theta/phi distribution in 3d, as well as the plane x1+x2+x3=1.
Loops over all alpha and beta values, and plots those bitches.
"""

import itertools as it
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ml_estimate as ml

# import doc,vocab indices
doc_idx,voc_idx = scipy.io.loadmat('data/classic400.mat')['classic400'].nonzero()

# set up strings & vals for hyperparameters
alphastr = ['0p1','1p0','10p0']
betastr = ['0p1','1p0','10p0','100p0','1000p0','10000p0']
alphaval = [0.1,1.0,10.0]
Kstr = ['3']
Kval = [3]

# simplex plot parameters & arrays
Npts = 200
xx,yy = np.meshgrid(np.linspace(0.0,1.0,Npts),np.linspace(0.0,1.0,Npts))
zz = 1.0 - xx - yy
for i,z1 in enumerate(zz):
    for j,z2 in enumerate(z1):
        if z2 < 0:
            zz[i,j] = None

# loop over alpha,beta,K
for (As,A),Bs,(Ks,K) in it.product(zip(alphastr,alphaval),betastr,zip(Kstr,Kval)):
    try:
        ndata = np.loadtxt('data/c400_n_a'+As+'_b'+Bs+'_K'+Ks+'.dat')
    except:
        continue
    alpha = A*np.ones(K)
    theta = ml.calc_theta(ndata,alpha)
    
    # now plot distribution
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(theta[:,0],theta[:,1],theta[:,2], '.', lw=0, ms=5)
    
    # plot sheet
    ax.plot_surface(xx,yy,zz, alpha=0.2, color='gray', linewidth=0)
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.set_zlim((0,1))

    # save figure
    fig.savefig('data/plots/c400_theta_a'+As+'_b'+Bs+'_K'+Ks+'.pdf')
    plt.close(fig)
