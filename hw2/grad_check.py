import subroutines as sr
import ffs as ff
import numpy as np
import dataproc as dp
import SGD_CRF as sgd
import math
from numpy.random import randn

def LCL_single(x,y,w):
    """
    computes the LCL for a single training example
    """
    x_info = ff.sent_precheck(x)
    
    g = sr.g(w,x)
    alpha = sr.alpha_mat(g)
    beta = sr.beta_mat(g)
    z = sr.Z(alpha,beta)
    N = len(x)
    sum_on_j = 0.0
    
    for i in range(1,N):
        trueFF = ff.metaff(y[i-1],y[i],x_info,i)
        for j in trueFF:
            sum_on_j += w[j]
    LCL = -math.log(z)
    return LCL

def LCL_loop(training_sentences,training_labels,w,pert):
    """
    for each training example, computes the difference between the LCL computed
    with weight w, and weight w+pert.  Stores all these differences as an array.
    """
    LCL_diff_array = np.zeros(len(training_labels))
    for i in range(len(training_labels)):
        LCL_diff_array[i] =  LCL_single(training_sentences[i],training_labels[i],w+pert) - LCL_single(training_sentences[i],training_labels[i],w)
    return LCL_diff_array

def delta_loop(training_sentences,training_labels,w,pert):
    """
    for each training example, computes the per-sample gradient vector, and then
    dots it with the perturbation vector to get a list of delta-LCL
    """
    dw = np.zeros(np.size(w))
    delta_array = np.zeros(len(training_labels))
    for i in range(len(training_labels)):
        dw = sgd.compute_gradient(training_sentences[i],training_labels[i],w,dw)
        delta_array[i] = np.dot(dw,pert)
    return delta_array
    
def finite_compare(training_sentences,training_labels,w0,sigma,Ntest):
    """
    computes the sample averaged difference between the finite difference of LCL
    and delta-LCL computed with the gradient.  It then averages this value for
    multiple realizations of the perturbation vector.
    """
    avg_error = 0.0
    for i in range(Ntest):
        pert = sigma*randn(np.size(w0))
        LCL_diff = LCL_loop(training_sentences,training_labels,w0,pert)
        delta_array = delta_loop(training_sentences,training_labels,w0,pert)
        error = np.mean(np.absolute(LCL_diff - delta_array))
        avg_error += error
    return avg_error / float(Ntest)

def expectation_check(training_sentences,training_labels,w):
    lhs_array = np.zeros(np.size(w))
    rhs_array = np.zeros(np.size(w))
    for i in range(len(training_labels)):
        y = training_labels[i]
        x = training_sentences[i]
        N = len(y)
        x_info = ff.sent_precheck(x)
        for k in range(1,N):
            trueFF = ff.metaff(y[k-1],y[k],x_info,k)
            for j in trueFF:
                lhs_array[j] += 1
        
        g = sr.g(w,x)
        e_g = np.exp(g)
        alpha = sr.alpha_mat(g)
        beta = sr.beta_mat(g)
        z = sr.Z(alpha,beta)
        for k in range(np.shape(g)[0]):
            for m1 in range(8):
                for m2 in range(8):
                    factor = alpha[k,m1]*beta[k+1,m2]*e_g[k,m1,m2]/z
                    #get list of non-zero (and thus =1) f_j for (i,m1,m2)
                    trueFF = ff.metaff(m1,m2,x_info,k)
                    #add the weighting factor to them
                    for j in trueFF:
                        rhs_array[j] += factor
    return lhs_array,rhs_array