"""
grad_comparison.py

Compares the value of the derivatives calculated with backpropagation, versus 
an estimate calculated using a central difference approximation.
"""

import numpy as np
import matplotlib.pyplot as plt

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
    
    

if __name__ == '__main__':
    main()
