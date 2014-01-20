"""
sgd.py

Implements stochastic gradient descent to train a logistic model conditional 
likelihood.  Outputs the trained parameters plus, optionally, the full time 
series of parameter values and the LCL at the end of each epoch of training.
"""

import numpy as np
import matplotlib.pyplot as plt
import data_routines as dr

def main():
    # load the data and preprocess it
    train_data = np.loadtxt('../dataset/1571/train_npcomp.dat', dtype='float')
    test_data = np.loadtxt('../dataset/1571/test_npcomp.dat', dtype='float')
    N_trainex = train_data.shape[0]
    N_testex = test_data.shape[0]
    
    train_data = dr.preprocess(train_data)
    
    # define pars/hyperpars for sgd process
    D = train_data.shape[1] - 1  # don't want to include the qualifier as a dim.
    x0 = np.zeros(D)  # initial position
    Nepochs = 100  # number of epochs
    Nt = Nepochs*N_trainex
    
    lr = 0.025  # learning rate
    mu = 0.002  # regularization scale
    
    traj = [x0]  # keeps track of trajectory through parameter space
    ns = 0  # step number index
    order = np.arange(0, N_trainex, dtype='int')  # ordering of training ex's
    classifiers = train_data[:,0]
    traits = train_data[:,1:]
    
    for nep in range(Nepochs):
        np.random.shuffle(order)
        for ind_ex in order:
            y_rand = classifiers[ind_ex]
            x_rand = traits[ind_ex]
            p_rand = logistic(traj[ns], x_rand)
            step = lr*((y_rand - p_rand)*x_rand - 2.0*mu*traj[ns])
            traj.append(traj[ns] + step)
            ns += 1
    traj = np.array(traj)
    
    # plot some parameter trajectory
    print 'plotting parameter trajectory'
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(traj[:,0])
    fig.savefig('testtraj_par.png')
    
    # plot LCL trajectory
    print 'calculating LCL time series'
    LCL_ts = np.zeros(Nepochs+1)
    ns = 0
    for step in traj[::N_trainex]:
        LCL_ts[ns] = LCL(step, train_data)
        ns += 1
    
    print 'plotting LCL time series'
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(LCL_ts)
    fig.savefig('testtraj_LCL.png')
    plt.close(fig)

def LCL(beta, x):
    """
    Computes LCL for a data set x, where x has shape (# examples, # dimensions 
    + 1).  The reason it is # dimensions + 1 is that the 0th element should be 
    the classifier.  Also, don't forget that there is a "fake" trait which is 
    always 1, but this is included as one of the dimensions.
    """
    
    N_ex = x.shape[0]
    traits = x[:,1:]
    result = 0.0
    
    for i in range(N_ex):
        if x[i,0] == 1:
            result += np.log(logistic(beta,traits[i]))
        else:
            result += np.log(1.0 - logistic(beta,traits[i]))
    
    return result

def LCL_reg(beta, x, mu):
    """
    Computes the regularized LCL (with Gaussian prior on betas)
    """
    
    return LCL(beta, x) - mu * np.einsum('i,i', beta,beta)

def logistic(beta, x):
    """
    Logistic function when beta and x are vectors.
    """
    return 1.0 / (1.0 + np.exp(-np.einsum('i,i', beta,x)))

if __name__ == '__main__':
    main()