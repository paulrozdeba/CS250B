"""
subroutines.py

Contains several routines 
"""

import numpy as np
import itertools as it
import ffs

def g(w, x):
    """
    Calculates g functions for each pair of tags y in a sentence x.
    
    w - List of weights associated with ffs.
    x - Sequence (sentence) over which to evaluate g.
    
    Returns: a (N-1) X M X M matrix where N is the length of the sentence, and 
    there are M possible tags for each word.
    """
    
    M = 8  # number of possible tags
    N = len(x)  # length of sentence
    __g__ = np.zeros(shape=(N-1,M,M))
    
    # preprocess the sentence
    x_info = ffs.sent_precheck(x)
    
    for i,m1,m2 in it.product(range(1,N),range(0,M),range(0,M)):
        # get the nonzero feature function indices for this tag pair
        trueFF = ffs.metaff(m1,m2,x_info,i)
        # fill in the nonzero elements of g
        for j in trueFF:
            __g__[i-1,m1,m2] += w[j]
    
    return __g__

def forward(e_g, k, v,alpha_mat):
    """
    e_g is the g array exponentiated, k is the current index in the tag sequence
    and v is the y_i
    """
    if(k==0):
        total = (v==0)
    else:
        total = 0
        for i in range(e_g.shape[1]):
            total += float(forward(e_g,k-1,i,alpha_mat))*e_g[k,i,v]
    return total

def backward(e_g, v, k,beta_mat):
    """
    e_g is the g array exponentiated, k is the current index in the tag sequence
    and v is the y_(i-1)
    """
    length = e_g.shape[0]-1
    if(k==length):
        total = (v==1)
    else:
        total = 0
        for i in range(e_g.shape[1]):
            total += float(backward(e_g,i,k+1,beta_mat))*e_g[k+1,v,i]
    return total
    
def alpha_mat(g):
    e_g = np.exp(g)
    N = e_g.shape[0] + 1
    M = e_g.shape[1]
    alpha_mat = np.zeros((N,M))
    for k in range(N):
        for v in range(M):
            if (k==0):
                alpha_mat[k,v] = float((v==0))
            else:
                alpha_mat[k,v] = np.dot(alpha_mat[k-1,:],e_g[k-1,:,v])
    return alpha_mat
     
def beta_mat(g):
    e_g = np.exp(g)
    N = e_g.shape[0] + 1
    M = e_g.shape[1]
    beta_mat = np.zeros((N,M))
    for k in range(N)[::-1]:
        for u in range(M):
            if (k==(N-1)):
                beta_mat[k,u] = float((u==1))
            else:
                beta_mat[k,u] = np.dot(beta_mat[k+1,:],e_g[k,u,:]) 
    return beta_mat

def U(g):
    """
    Calculates the matrix elements of the propagator U(k,v).
    
    g - The matrix elements of g for the entire sequence x.
    """
    
    N = g.shape[0]  # length of sentence x
    M = g.shape[1]  # number of possible tags
    __U__ = np.zeros(shape=(N,M))
    
    for i in range(N):
        __U__[i] = __U_singlek__(g,i)
    
    return __U__
        
def __U_singlek__(g, k):
    """
    Supplementary function, for use inside U(g) only.
    """
    
    # implement recursion here
    if k == 0:
        return np.amax(g[0], axis=0)
    else:
        return np.amax(__U_singlek__(g,k-1) + g[k], axis=0)

def Z(alpha_matrix,beta_matrix):
    Z = np.zeros(alpha_matrix.shape[0])
    for k in range(alpha_matrix.shape[0]):
        Z[k] = np.dot(alpha_matrix[k,:],beta_matrix[k,:])
    return np.mean(Z)

def bestlabel(U, g):
    """
    Predicts the best label for an example, based on U and g matrices.
    """
    
    N = U.shape[0]  # length of sentence
    y = []  # to store the label
    
    # first find the best tag at position n
    y_N = np.argmax(U[-1])
    y.append(y_N)
    ykbest = y_N
    
    # now get the rest of the label
    for i,(U_km1,g_k) in enumerate(zip(U[::-1],g[::-1])):
        y_km1_best = np.argmax(U_km1 + g_k[:,ykbest])
        y.append(y_km1_best)
        ykbest = y_km1_best
    
    return y[::-1]


def score(weights,validate_labels,validate_sentences):
    """
    Calculates the average word level accuracy percentage
    """
    N_validate = len(validate_labels)
    average_error = 0.0
    for i in range(N_validate):
        num_error = 0
        y = validate_labels[i]
        x = validate_sentences[i]
        g = g(weights,x)
        U = U(g)
        y_predict = bestlabel(U,g)
        for j in range(len(y)):
            if(y[j] != y_predict[j]):
                num_error += 1.0
        num_error *= 1.0/float(len(y))
        average_error += num_error
    average_error *= 1.0/float(N_validate)
    return average_error