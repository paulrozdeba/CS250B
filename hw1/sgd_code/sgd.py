"""
sgd.py

Functions that define the LCL, as well as the SGD and grid search procedures, 
specifically implemented for the logistic probability model
"""

import numpy as np

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
            LCL1 += LCL_reg(traj[ns-1], train_data, mu)
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
            LCL2 += LCL_reg(traj[ns-1], train_data, mu)
            nep += 1
        
        if abs((LCL2 - LCL1)/LCL1) < 1e-3:
            print 'Converged after ' + str(nep) + ' epochs!'
            print 'LCL = ' + str(LCL_reg(traj[ns-1], train_data, mu)) + '\n'
            converged = True
    
    traj = np.array(traj)
    trained_pars = np.mean(traj[-N_testepoch:], axis=0)
    
    if full_output==True:
        return traj
    else:
        return trained_pars

def mu_gridsearch(train_data, valid_data, mu_vec=None, N_mutestiter=4, lrmu=.00005):
    """
    Function mu_gridsearch.
    
    Perform a search over values of mu, the regularization scale, for the value 
    which maximized the LCL of a trained model when applied to a validation 
    set.
    """
    
    # define intial mu and LCL vectors, as well as # of iterations.
    if mu_vec == None:
        mu_vec = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    LCL_valid_vec = np.zeros(mu_vec.shape[0])
    
    # Start searching over mu values.  The algorithm will continuously zoom in 
    # near optimal mu values for N_mutestiter iterations.
    for ind_mutestiter in range(N_mutestiter):
        print 'Searching over mu values:'
        print mu_vec
        
        # need an array to store trained parameter values
        trained_pars_vec = np.zeros(shape=(mu_vec.shape[0],train_data.shape[1]-1))
        
        for ind_mu in range(mu_vec.shape[0]):
            # note: optimal product lr*mu based on empirical observation
            mu = mu_vec[ind_mu]  # regularization scale
            lr = lrmu/mu  # learning rate
            
            print 'mu = ' + str(mu)
            print 'lr = ' + str(lr)
            
            # get trained parameter values
            trained_pars_vec[ind_mu] = sgd_train(train_data, lr, mu, full_output=False)
            # save to file
            #np.save('pars_traj.npy', pars_traj)
            
            # now compute LCL over validation set
            LCL_valid = LCL(trained_pars_vec[ind_mu], valid_data)
            LCL_valid_vec[ind_mu] = LCL_valid
        
        # extract optimal values
        LCL_max = np.amax(LCL_valid_vec)
        LCL_max_ind = np.argmax(LCL_valid_vec)
        mu_max = mu_vec[LCL_max_ind]
        trained_pars = trained_pars_vec[LCL_max_ind]
        
        if ind_mutestiter < N_mutestiter-1:
            if LCL_max_ind == 0:
                # Optimal value is somewhere below the smallest tested mu, so 
                # look at several smaller orders of magnitude.
                mu_vec = np.array([mu_max/10**6, mu_max/10**5, mu_max/10**4, 
                                   mu_max/10**3, mu_max/10**2, mu_max/10, mu_max])
            elif LCL_max_ind == (mu_vec.shape[0]-1):
                # Optimal value is somewhere above largest tested mu, so look at 
                # several larger orders of magnitude
                mu_vec = np.array([mu_max, mu_max*10, mu_max*10**2, mu_max*10**3, 
                                   mu_max*10**4, mu_max*10**5, mu_max*10**6])
            else:
                # Optimal value appeared in tested array.  Zoom in, enhance.
                newmu_max = mu_vec[LCL_max_ind + 1]
                newmu_min = mu_vec[LCL_max_ind - 1]
                us = (newmu_max / mu_max)
                ds = (newmu_min / mu_max)
            
                mu_vec = np.array([mu_max * ds, 
                                   mu_max * ds**(2.0/3.0), 
                                   mu_max * ds**(1.0/3.0), 
                                   mu_max, 
                                   mu_max * us**(1.0/3.0), 
                                   mu_max * us**(2.0/3.0),
                                   mu_max * us])
        
        print LCL_valid_vec
        print ''
    
    lr_max = lrmu/mu_max
    print 'Final result:'
    print '    mu = ' + str(mu_max)
    print '    lr = ' + str(lr_max)
    print '    LCL = ' + str(LCL_max) + ' (on validation set)\n'
    
    return trained_pars, mu_max, lr_max, LCL_max

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
