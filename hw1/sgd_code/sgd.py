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
    
    # save first half as training data, second half as validation data
    N_trainex = train_data.shape[0]
    if N_trainex%2 == 0:
        N_trainex = int(N_trainex/2)
    else:
        N_trainex = int(N_trainex/2)+1
    valid_data = train_data[N_trainex:]
    train_data = train_data[:N_trainex]
    
    N_trainex = train_data.shape[0]
    N_validex = valid_data.shape[0]
    N_testex = test_data.shape[0]
    
    train_data = dr.preprocess(train_data)
    valid_data = dr.preprocess(valid_data)
    
    # Search over values of mu (the regularization scale)
    mu_vec = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    LCL_valid_vec = np.zeros(mu_vec.shape[0])
    N_mutestiter = 1
    
    for ind_mutestiter in range(N_mutestiter):
        for ind_mu in range(mu_vec.shape[0]):
            # note: optimal product lr*mu based on empirical observation
            mu = mu_vec[ind_mu]  # regularization scale
            lr = .00005/mu  # learning rate
            
            print 'mu = ' + str(mu)
            print 'lr = ' + str(lr)
            
            # get trained parameter values
            trained_pars = sgd_train(train_data, lr, mu, full_output=False)
            # save to file
            #np.save('pars_traj.npy', pars_traj)
            
            # now compute LCL over validation set
            LCL_valid = LCL(trained_pars, valid_data)
            LCL_valid_vec[ind_mu] = LCL_valid
        
        # find optimal value, zoom in around it
        LCL_max = np.amax(LCL_valid_vec)
        LCL_max_ind = np.argmax(LCL_valid_vec)
        mu_max = mu_vec[LCL_max_ind]
        
        if LCL_max_ind == 0:
            # Optimal value is somewhere below the smallest tested mu, so look 
            # at several smaller orders of magnitude.
            mu_vec = np.array([mu_max/10^6, mu_max/10^5, mu_max/10^4, 
                               mu_max/10^3, mu_max/10^2, mu_max/10, mu_max])
        elif LCL_max_ind == (mu_vec.shape[0]-1):
            # Optimal value is somewhere above largest tested mu, so look at 
            # several larger orders of magnitude
            mu_vec = np.array([mu_max, mu_max*10, mu_max*10^2, mu_max*10^3, 
                               mu_max*10^4, mu_max*10^5, mu_max*10^6])
        else:
            # Optimal value appeared in tested array.  Zoom in, enhance.
            mu_vec = np.array([mu_max/9.0, mu_max/6.0, mu_max/3.0, mu_max, 
                               mu_max*3.0, mu_max*6.0, mu_max*9.0])
        
        print LCL_valid_vec
    
    return trained_pars, mu_max, .00005/mu_max
    
    """
    # saving and plotting arrays
    # plot some parameter trajectory
    print 'plotting parameter trajectory'
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(pars_traj[:,0])
    fig.savefig('testtraj_par.png')
    
    # plot LCL trajectory
    print 'calculating LCL time series'
    LCL_ts = np.zeros(pars_traj.shape[0]/N_trainex + 1)
    ns = 0
    for step in pars_traj[::N_trainex]:
        LCL_ts[ns] = LCL(step, train_data)
        ns += 1
    
    print 'plotting LCL time series'
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(LCL_ts)
    fig.savefig('testtraj_LCL.png')
    plt.close(fig)
    """

################################################################################

def sgd_train(train_data, lr, mu, x0=None, full_output=False):
    """
    Function sgd_train.
    
    Trains the logistic model to the training data.  Pass training data along 
    with hyperparameters lr (the learning rate) and mu (the regularization 
    scale).  x0 is the initial position in parameter space, defaults to the 
    origin.
    Setting full_output=True will return the entire trajectory; otherwise it 
    returns the final parameter values.
    """
    
    # define pars/hyperpars for sgd process
    D = train_data.shape[1] - 1  # don't want to include the qualifier as a dim.
    N_trainex = train_data.shape[0]
    if x0 == None:
        x0 = np.zeros(D)  # initial position
    
    traj = [x0]  # keeps track of trajectory through parameter space
    order = np.arange(0, N_trainex, dtype='int')  # ordering of training ex's
    classifiers = train_data[:,0]
    traits = train_data[:,1:]
    
    ns = 0  # step number index
    nep = 0  # epoch index
    
    # The procedure repeatedly runs for N_testepoch epochs, and tests for 
    # convergence by comparing the average value of the LCL over the two halves 
    # of the test set.  When the fractional difference goes below the set 
    # tolerance, the procedure terminates and returns the average of the 
    # parameter values over the last N_testepoch epochs.
    N_testepoch = 10
    converged = False
    
    while converged == False:
        # track LCL values over test sets
        LCL1 = 0.0
        LCL2 = 0.0
        # loop over first N_testepoch epochs
        for i in range(N_testepoch/2):
            np.random.shuffle(order)  # randomly shuffle test data
            for ind_ex in order:
                y_rand = classifiers[ind_ex]
                x_rand = traits[ind_ex]
                p_rand = logistic(traj[ns], x_rand)
                step = lr*((y_rand - p_rand)*x_rand - 2.0*mu*traj[ns])
                traj.append(traj[ns] + step)
                ns += 1
            LCL1 += LCL(traj[ns-1], train_data)
            nep += 1
        # loop over second N_testepoch epochs
        for i in range(N_testepoch/2):
            np.random.shuffle(order)
            for ind_ex in order:
                y_rand = classifiers[ind_ex]
                x_rand = traits[ind_ex]
                p_rand = logistic(traj[ns], x_rand)
                step = lr*((y_rand - p_rand)*x_rand - 2.0*mu*traj[ns])
                traj.append(traj[ns] + step)
                ns += 1
            LCL2 += LCL(traj[ns-1], train_data)
            nep += 1
        
        if abs((LCL2 - LCL1)/LCL1) < 1e-3:
            print 'Converged after ' + str(nep) + ' epochs!'
            print 'LCL = ' + str(LCL(traj[ns-1], train_data)) + '\n'
            converged = True
    
    traj = np.array(traj)
    trained_pars = np.mean(traj[-N_testepoch:], axis=0)
    
    if full_output==True:
        return traj
    else:
        return trained_pars

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