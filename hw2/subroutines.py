"""
subroutines.py

Contains several routines 
"""

import numpy as np
import itertools as it

def _g(f, w, tags, x):
    """
    Calculates g functions for each pair of tags y in a sentence x.
    This is NOT something the user should be calling, hence the _g designation.
    Rather, this will be called by U, alpha, beta, etc.
    
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

def U(g):
    """
    Calculates the matrix elements of the propagator U(k,v).
    
    g - The matrix elements of g for the entire sequence x.
    """






