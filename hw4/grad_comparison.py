"""
grad_comparison.py

Compares the value of the derivatives calculated with backpropagation, versus 
an estimate calculated using a central difference approximation.
"""

import numpy as np
import matplotlib.pyplot as plt
from backprop import backprop,backprop_full
from training import full_j
from dataproc import format_data

def midpoint_grad(f, x, eps):
    """
    Calculates the midpoint approximation to the gradient for all directions.
    
    f - The function to take gradient of.
    x - Current position.
    eps - Step size.  Can either be a single number, or an array of the same
          shape as x.
    """
    
    return (f(x+eps) - f(x-eps)) / (2*eps)

def main():
    D = 4
    L = 2
    
    # randomly generate parameter values for testing
    np.random.seed(28031987)
    W1 = np.random.randn(D,2*D)
    b1 = np.random.randn(D)
    W2 = np.random.randn(2*D,D)
    b2 = np.random.randn(2*D)
    Wlabel = np.random.randn(L,D)
    
    # flattened arrays
    W1flat = W1.flatten()
    W2flat = W2.flatten()
    Wlabelflat = Wlabel.flatten()
    allflat = np.append(W1flat,b1)
    allflat = np.append(allflat,W2flat)
    allflat = np.append(allflat,b2)
    allflat = np.append(allflat,Wlabelflat)
    Np = len(allflat)
    
    # hyperparameters
    eps = 0.0001
    lambda_reg = 0.1
    alpha = 0.2
    
    # get the data
    neg_list,pos_list = format_data()
    neg_list = neg_list[:1]
    pos_list = pos_list[:1]
    vocab = np.random.randn(268810,D)
    
    numgrad = np.zeros(Np)
    for i in range(Np):
        print 'P ' + str(i)
        allflat[i] += eps
        fxpe = full_j(allflat,D,L,lambda_reg,alpha,neg_list,pos_list,
                      vocab,normalized=True)
        allflat[i] -= 2.0*eps
        fxme = full_j(allflat,D,L,lambda_reg,alpha,neg_list,pos_list,
                       vocab,normalized=True)
        allflat[i] += eps
        numgrad[i] = (fxpe - fxme)/(2.0*eps)
    print numgrad
    

if __name__ == '__main__':
    main()
