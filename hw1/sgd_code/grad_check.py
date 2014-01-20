import numpy as np
import sgd
from numpy.random import randn

def grad_check(x,beta,delta):
    """
    Takes a formatted classifier-trait pair example (x), and the current
    parameter vector (beta) along with the learning rate and a perturbation 
    scale (delta).  It first calculates the gradient for this example. A
    perturbation vector is constructed composed of normally distributed values
    with standard deviation delta. It then applies the perturbation to beta and calculates
    the difference between LCL(beta,x) and LCL(beta + delta,x).  If the gradient
    is accurate, then LCL(beta + delta,x) - LCL(beta,x) should roughly equal
    delta * grad.
    """
    
    grad = np.zeros(np.size(beta))
    y = x[0]     #classifier value
    traits = x[1:]     #trait vector
    p = sgd.logistic(beta,traits)
    x_send = x.reshape(1,np.size(x)) #formatting for LCL function
    grad = (y - p)*traits     #calculate gradient
    LCL_prior = sgd.LCL(beta,x_send)     #initial LCL
    delta_vec = delta * randn(np.size(beta)) #create perturbation vector
    beta_prime = beta + delta_vec #perturbed beta vector
    LCL_prime = sgd.LCL(beta_prime,x_send) #LCL(beta + delta,x)
    diff = np.dot(delta_vec,grad)
    del_LCL = LCL_prime - LCL_prior #calculated del:LCL(x,beta)
    return diff, del_LCL, diff/del_LCL
    
    