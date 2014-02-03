"""
subroutines.py

Contains several routines 
"""

import numpy as np
import itertools as it

def g(f, w, tags, x):
    """
    Calculates g functions for each pair of tags y in a sentence x.
    
    f - The "meta" feature function.
    w - List of weights associated with ffs.
    tags - The set of possible tags.
    x - Sequence (sentence) over which to evaluate g.
    
    Returns: a (N-1) X M X M matrix where N is the length of the sentence, and 
    there are M possible tags for each word.
    """
    
    M = len(tags)  # number of possible tags
    N = len(x)  # length of sentence
    __g__ = np.zeros(shape=(N-1,M,M))
    
    for i,(k1,tag1),(k2,tag2),(j,weight) in it.product(range((1,N)),enumerate(tags),enumerate(tags),enumerate(w)):
#        if i == 1:
#            #__g__[i-1,k1,k2] += weight * f('START',tag2,x,i,j)
#            __g__[i-1,k1,k2] += weight * f(tag1,tag2,x,i,j) * (tag1=='START')
#        elif i == (N-1):
#            #__g__[i-1,k1,k2] += weight * f(tag1,'STOP',x,i,j)
#            __g__[i-1,k1,k2] += weight * f(tag1,tag2,x,i,j) * (tag2=='STOP')
#        else:
#            __g__[i-1,k1,k2] += weight * f(tag1,tag2,x,i,j)
        __g__[i-1,k1,k2] += weight * f(tag1,tag2,x,i,j)
    
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

def Z(w,x):
    Z = 0.0
    
    




