"""
run_sgd.py

Runs the SGD procedure over the data for CS250B hw1, including automatic 
convergence detection in SGD optimization over the training data and a grid 
search over values of the regularization scale (which optimizes the regularized 
parameters over the validation set).
"""

import numpy as np
import matplotlib.pyplot as plt
import data_routines as dr
import sgd

def main():
    N_samples = 250
    results = np.zeros(shape=(N_samples, 6))
    
    for i in range(N_samples):
        results[i] = run_sgd()
    np.save('sampling_1.npy', results)
    
    """
    # generate sample run using mu_max and lr_max, to plot sample parameter 
    # trajectory and LCL trajectory
    sample_traj = sgd.sgd_train(train_data, lr_max, mu_max, full_output=True)
    
    # plot results
    components = (0, 47, 337)
    plot_pars(sample_traj, components)
    plot_LCL(sample_traj, train_data)
    """

def run_sgd():
    # load the data and preprocess it
    train_data = np.loadtxt('../dataset/1571/train_npcomp.dat', dtype='float')
    test_data = np.loadtxt('../dataset/1571/test_npcomp.dat', dtype='float')
    
    # save a copy of the original data set, before shuffling and dividing
    train_data_orig = np.copy(train_data)
    
    # save first half as training data, second half as validation data
    D = train_data.shape[1] - 1
    N_trainex = train_data_orig.shape[0]
    # shuffle the training set before splitting it
    randstate1 = np.random.get_state()
    np.random.shuffle(train_data_orig)
    
    if N_trainex%2 == 0:
        N_trainex = int(N_trainex/2)
    else:
        N_trainex = int(N_trainex/2)+1
    valid_data = train_data_orig[N_trainex:]
    train_data = train_data_orig[:N_trainex]
    
    N_trainex = train_data.shape[0]
    N_validex = valid_data.shape[0]
    N_testex = test_data.shape[0]
    
    # preprocess the data
    train_data, tdmean, tdstd = dr.preprocess(train_data, full_output=True)
    valid_data = dr.preprocess(valid_data, rescale=False)
    test_data = dr.preprocess(test_data, rescale=False)
    
    # rescale validation and test sets same as training data
    valid_data[:,1:-1] -= np.resize(tdmean, (N_validex,D))
    valid_data[:,1:-1] /= tdstd
    test_data[:,1:-1] -= np.resize(tdmean, (N_testex,D))
    test_data[:,1:-1] /= tdstd
    
    # initiate the grid search over mu
    trained_pars, mu_max, lr_max, LCL_max = sgd.mu_gridsearch(train_data, valid_data)
    
    # calculate error rate on validation data set
    errors = 0
    for ind_ex in range(N_validex):
        if sgd.logistic(trained_pars, valid_data[ind_ex,1:]) >= 0.5:
            errors += 1 - valid_data[ind_ex,0]
        else:
            errors += valid_data[ind_ex,0]
    error_rate_valid = errors/N_validex
    print 'Error rate on validation data = ' + str(error_rate_valid*100) + '%\n'
    
    # calculate LCL on test data
    LCL_test = sgd.LCL(trained_pars, test_data)
    
    # calculate error rate on test data set
    errors = 0
    for ind_ex in range(N_testex):
        if sgd.logistic(trained_pars, test_data[ind_ex,1:]) >= 0.5:
            errors += 1 - test_data[ind_ex,0]
        else:
            errors += test_data[ind_ex,0]
    error_rate_test = errors/N_testex
    print 'Error rate on test data = ' + str(error_rate_test*100) + '%\n'
    
    return mu_max, lr_max, LCL_max, error_rate_valid, LCL_test, error_rate_test

def plot_pars(par_traj, compts, fname_ts=None, fname_ep=None):
    """
    Function plot_pars.
    
    Plots sample parameter trajectories from SGD over time steps and epochs.
    Traj should be the *entire* sample trajectory, and compts should be a tuple 
    containing the desired trajectories to plot.  The fname flags should be 
    tuples of filenames for the plots, if desired.
    """
    
    for c in compts:
        print 'plotting parameter ' + str(c) + ' (time steps)'
        fig = plt.figure()
        fig.set_tight_layout(True)
        ax = fig.add_subplot(1,1,1)
        ax.plot(par_traj[:,c])
        ax.set_xlim((0,par_traj.shape[0]))
        ax.set_title('Sample parameter trajectory')
        ax.set_xlabel('Step number')
        ax.set_ylabel(r'$\beta_' + str(c) + '$')
        if fname_ts == None:
            fname = 'beta' + str(c) + '_traj_ts.pdf'
        else:
            try:
                fname = fname_ts[c]
            except:
                fname = 'beta' + str(c) + '_traj_ts.pdf'
        fig.savefig(fname)
        
        print 'plotting parameter ' + str(c) + ' (epochs)'
        fig = plt.figure()
        fig.set_tight_layout(True)
        ax = fig.add_subplot(1,1,1)
        ax.plot(par_traj[::N_trainex,c])
        ax.set_xlim((0,par_traj[::N_trainex].shape[0]))
        ax.set_title('Sample parameter trajectory')
        ax.set_xlabel('Epoch number')
        ax.set_ylabel(r'$\beta_' + str(c) + '$')
        if fname_ep == None:
            fname = 'beta' + str(c) + '_traj_ep.pdf'
        else:
            try:
                fname = fname_ep[c]
            except:
                fname = 'beta' + str(c) + '_traj_ep.pdf'
        fig.savefig(fname)

def plot_LCL(par_traj, data, fname=None):
    """
    Function plot_LCL.
    
    Plots the LCL of the logistic model as a function of time.
    """
    
    print 'calculating LCL time series'
    LCL_ts = np.zeros(traj.shape[0] + 1)
    ns = 0
    for step in traj:
        LCL_ts[ns] = sgd.LCL(step, data)
        ns += 1

    print 'plotting LCL time series'
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(LCL_ts)
    ax.set_xlim((0,par_traj.shape[0]))
    ax.set_title('Sample LCL trajectory')
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('LCL')
    if fname == None:
        fname = 'LCL_traj_ep.pdf'
    fig.savefig(fname)

if __name__ == '__main__':
    main()





