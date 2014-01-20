"""
data_routines.py

Several routines used to manipulate & process data from the MLcomp website.
"""

import numpy as np

def make_npcomp(traindata_loc, testdata_loc, mydtype='float'):
    """
    Function make_npcomp.
    
    Takes training and test data files from the Mlcomp site, and creates 
    numpy compatible data files.  Specifically, it strips the column number and 
    colon from the original data so that it can be read by numpy.loadtxt.
    Assumes data type is float, but optionally accepts int.
    """
    
    # load file as strings
    raw_train_data = np.loadtxt(traindata_loc, dtype='str')
    raw_test_data = np.loadtxt(testdata_loc, dtype='str')
    # initialize array to store data
    train_data = np.zeros(shape=raw_train_data.shape, dtype=mydtype)
    test_data = np.zeros(shape=raw_test_data.shape, dtype=mydtype)
    
    # now loop through the raw training data and remove col #s and colons
    for ind_ex in range(raw_train_data.shape[0]):  # training example
        for ind_cm in range(raw_train_data.shape[1]):  # cmpt of example
            if ind_cm == 0:
                train_data[ind_ex,ind_cm] = float(raw_train_data[ind_ex,ind_cm])
            else:
                i = 0
                while raw_train_data[ind_ex,ind_cm][i] != ':':
                    i += 1
                i += 1
                train_data[ind_ex,ind_cm] = float(raw_train_data[ind_ex,ind_cm][i:])
    
    # do the same for the test data
    for ind_ex in range(raw_test_data.shape[0]):  # training example
        for ind_cm in range(raw_test_data.shape[1]):  # cmpt of example
            if ind_cm == 0:
                test_data[ind_ex,ind_cm] = float(raw_test_data[ind_ex,ind_cm])
            else:
                i = 0
                while raw_test_data[ind_ex,ind_cm][i] != ':':
                    i += 1
                i += 1
                test_data[ind_ex,ind_cm] = float(raw_test_data[ind_ex,ind_cm][i:])
    
    # now save the compatible arrays to file
    if mydtype == 'float':
        np.savetxt(traindata_loc+'_npcomp.dat', train_data)
        np.savetxt(testdata_loc+'npcomp.dat', test_data)
    elif mydtype == 'int':
        np.savetxt(traindata_loc+'_npcomp.dat', train_data, fmt='%i')
        np.savetxt(testdata_loc+'_npcomp.dat', test_data, fmt='%i')
    else:
        print '***ERROR: Incompatible data type passed as argument.'
        exit(0)

def preprocess(data, mydtype='float', rescale=True, full_output=False):
    """
    Function preprocess.
    
    Takes a data set from MLcomp with binary classifiers, and changes them from 
    (1,-1) to (1,0).  By default, also rescales traits to have mean=0 and var=1.
    When full_output=True, also returns mean and variance vectors of data.
    """
    
    D = data.shape[1]  # number of traits
    N_ex = data.shape[0]  # number of examples
    
    # now append a 1 to the end of each example, for the intercept parameter in
    # the logistic model
    data = np.hstack((data, np.reshape(np.ones(N_ex,dtype=mydtype), (N_ex,1))))
    
    # change classifier to (1,0)
    if mydtype == 'float':
        data[:,0] = (data[:,0] + 1.0)/2.0
    elif mydtype == 'int':
        data[:,0] = (data[:,0] + 1)/2
    
    # rescale data so that all traits in training set have mean=0, variance=1
    # don't use last trait, which is always 1, since it has no variance
    if rescale == True:
        data_mean = np.mean(data[:,1:-1], axis=0)
        data_var = np.std(data[:,1:-1], axis=0)
        data[:,1:-1] -= np.resize(data_mean, (N_ex,D-1))
        data[:,1:-1] /= data_std
    
    if full_output == True:
        return data, data_mean, data_var
    else:
        return data





