"""
ffs.py

This module contains all of the base-level feature functions to be used in the 
punctuation prediction program.  The base-level feature functions are the 
feature functions defined over *pairs* of tags in a label y; this allows for 
the simple implementation of several algorithms, including the Viterbi 
algorithm for computing the score of a tag/sentence pair, and the algorithms 
for computing forward/backward vectors, etc.
"""

import numpy as np

def metaff(tag1, tag2, x, i, j):
    """
    This is the feature function you interact with.
    tag1, tag2 - The tag pair over which to evaluate a feature function.
    x - The sentence on which to evaluate a ff.
    i - Position in the sentence.
    j - Label for the feature function.
    """
    
    tags = ['START','STOP','SPACE','PERIOD','COMMA','COLON','QUESTION_MARK',
            'EXCLAMATION_PT']
    suffixes = ['ing','ly']
    prefixes = ['pre','post','de','inter','intra']
    M = len(tags)
    Nsuff = len(suffixes)
    Npref = len(prefixes)
    
    # single tag indicator, first tag
    if j >= 0 and j < M:
        ind = j
        return __indicator__(tag1,tags[ind])
    # single tag indicator, second tag
    elif j >= M and j < 2*M:
        ind = j-M
        return __indicator__(tag2,tags[ind])
    # tag pair indicator
    elif j >= 2*M and j < (M*M + 2*M):
        ind1 = int(j-2*M)/int(M)
        ind2 = (j-2*M)%M
        return __indicator__(tag1,tags[ind1]) * __indicator__(tag2,tags[ind2])
    # tag/prefix indicator
    elif j >= (M*M + 2*M) and j < (M*M + 2*M + M*Npref):
        ind1 = int(j-M*M-2*M)/int(M)
        ind2 = (j-M*M-2*M)%Npref
        return __indicator__(tag1,tags[ind1]) * __indicator__(x[i-1],prefixes[ind2])
    elif j >= (M*M + 2*M + M*Npref) and j < (M*M + 2*M + 2*M*Npref):
        ind1 = int(j-M*M-2*M-M*Npref)/int(M)
        ind2 = (j-M*M-2*M-M*Npref)%Npref
        return __indicator__(tag2,tags[ind1]) * __indicator__(x[i],prefixes[ind2])
    else:
        raise ValueError('Invalid feature function index number.')
        

def __indicator__(data, test):
    """
    A simple Boolean indicator function on the two arguments, testing for 
    equality.  The inputs can either be individual strings or tuples/lists 
    of strings.
    """
    
    return data == test

def __length__(x):
    """
    Computes and returns the length of a word or sentence.  If it's a word, 
    just pass it bare to the function.  If a sentence, you should pass it 
    already parsed into a list of words.
    """
    
    return len(x)

def __word_suffix__(x, suffix):
    """
    A Boolean-valued function that returns whether or not the words in the 
    input sequence "x" end in the input "suffix".
    """
    
    return [word.endswith(suffix) for word in x]

def __word_prefix__(x, prefix):
    """
    A Boolean-valued function that returns whether or not the words in the 
    input sequence "x" begin with the input "prefix".
    """
    
    return [word.startswith(prefix) for word in x]
