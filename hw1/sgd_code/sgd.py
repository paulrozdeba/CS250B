"""
sgd.py

Implements stochastic gradient descent to train a logistic model conditional 
likelihood.  Outputs the trained parameters plus, optionally, the full time 
series of parameter values and the LCL at the end of each epoch of training.
"""

import numpy as np

def LCL(beta, x):
    """
    Computes LCL for a data set x, where x has shape (# examples, # dimensions 
    + 1).  The reason it is # dimensions + 1 is that the 0th element should be 
    the classifier.  Also, don't forget that there is a "fake" trait which is 
    always 1, but this is included as one of the dimensions.
    """
    
    traits = x[:,1:]
    result = 0.0
    
    for i in range(N_trainex):
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