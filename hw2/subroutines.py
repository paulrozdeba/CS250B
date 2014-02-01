"""
subroutines.py

Contains several routines 
"""

import numpy as np
import itertools as it

def g(f, w, tags, x):
    """
    Calculates g functions for each pair of tags y in a sentence x.
    
    f - List of low-level feature functions over which to evaluate g.
    w - List of weights associated with ffs.
    y - The set of possible tags.
    x - Sequence (sentence) over which to evaluate g.
    
    Returns: a (N-1) X M X M matrix where N is the length of the sentence, and 
    there are M possible tags for each word.
    """
    
    M = len(tags)  # number of possible tags
    assert(M == len(f))  # just making sure things are in order
    assert(len(f) == len(w))
    
    N = len(x)  # length of sentence
    g = np.zeros(shape=(N-1,M,M))
    
    # outer loops goes over sentence
    for i in range(N-1):
        # inner loop covers all possible tag pairs
        for (j,tag1),(k,tag2),(weight,func) in it.product(enumerate(tags),enumerate(tags),zip(w,f)):
            g[i,j,k] += weight * func(tag1,tag2,x,i)
    
    return g

def forward(e_g,k,v):
    """
    e_g is the g array exponentiated, k is the current index in the tag sequence
    and v is the y_i
    """
    if(k==0):
        return(v==0)
    else:
        total = 0
        for i in range(8):
            total += float(forward(e_g,k-1,i))*e_g[k,i,v]
        return total

def backward(e_g,v,k):
    """
    e_g is the g array exponentiated, k is the current index in the tag sequence
    and v is the y_(i-1)
    """
    length = e_g.shape[0]-1
    if(k==length):
        return(v==1)
    else:
        total = 0
        for i in range(8):
            total += float(backward(e_g,i,k+1))*e_g[k+1,v,i]
        return total

def U(g):
    """
    Calculates the matrix elements of the propagator U(k,v).
    
    g - The matrix elements of g for the entire sequence x.
    """






