"""""
ffs.py

This module contains all of the base-level feature functions to be used in the 
punctuation prediction program.  The base-level feature functions are the 
feature functions defined over *pairs* of tags in a label y; this allows for 
the simple implementation of several algorithms, including the Viterbi 
algorithm for computing the score of a tag/sentence pair, and the algorithms 
for computing forward/backward vectors, etc.
"""

import numpy as np

def sent_precheck(x):
    """
    Pre-checks a sentence x for word-level features.
    
    x - The ENTIRE sentence, as a list of words.
    
    Returns: x_info.
    x_info is a list containing information about the sentence.
    The zeroth entry is a NxD numpy array containing indices of True 
    comparisons against dictionaries, like conjunctions, prefixes, suffixes, 
    etc.
    The first entry is the length of the sentence.
    The second entry is "begins with interrogative" indicator.
    """
    
    x_info = {}
    
    # Length of sentence.
    x_info['length'] = length(x)
    
    # First check: capitalization.
    # BLAH BLAH BLAH
    
    # now lowercase this shit
    x = [word.lower for word in x]
    
    # First, define all word dictionaries.
    conjunctions = [None, 'after', 'although', 'and', 'as', 'because', 'before', 
                    'but', 'except', 'if', 'like', 'nor', 'now', 'once', 'or', 
                    'since', 'so', 'than', 'that', 'though', 'unless', 'until', 
                    'when', 'where', 'whether', 'while']
    #suffixes = [None, 'ing', 'ly']
    #prefixes = [None, 'pre', 'post', 'de', 'inter', 'intra']
    
    # Start the dictionary checks.
    D = 1  # number of dictionaries.
    x_yes = np.array(shape=(N,D))
    
    for i,word in enumerate(x):
        # check conjunctions
        if word in conjunctions:
            x_yes[i,0] = x.index(word)
        else:
            x_yes[i,0] = 0
    
    x_info['dictchecks'] = x_yes
    
    return x_info

def metaff(m1, m2, x_info, i):
    """
    This is the feature function you interact with.
    
    m1, m2 - The indices of the tag pair over which to evaluate the ff's.
    x_info - Information about the sentence.
    i - Position in sentence.
    
    Returns a list of indices of the TRUE feature functions.
    """
    
    # define the size of some spaces
    M = 8  # number of possible tags
    Np = x_info['Np']  # number of prefixes
    Ns = x_info['Ns']  # number of suffixes
    Nc = x_info['Nc']  # number of conjunctions
    
    trueFF = []  # container list for true FF meta-indices
    
    ############
    # SINGLE TAG INDICATORS (STI)
    STI1 = [m1]
    STI2 = [m2]
    
    ############
    # WORD-LEVEL DICTIONARY INDICATORS
    PREFIX = x_info['pref_true']
    SUFFIX = x_info['suff_true']
    CONJUNCTION = x_info['conj_true']
    CAPITALIZED = x_info['cap_true']
    
    ############################################################################
    # Now let's look at pairwise-indicators
    ############
    # TAG PAIR INDICATOR (TPI)
    TPI = [m1, m2]
    ############
    # SINGLE TAG, SINGLE PREFIX (STSPRE)
    STSPRE1 = [m1, PREFIX[0]]
    STSPRE2 = [m2, PREFIX[1]]
    # SINGLE TAG, SINGLE SUFFIX (STSSUFF)
    STSSUFF1 = [m1, SUFFIX[0]]
    STSSUFF2 = [m2, SUFFIX[1]]

def indicator(data, test):
    """
    A simple Boolean indicator function on the two arguments, testing for 
    equality.  The inputs can either be individual strings or tuples/lists 
    of strings.
    """
    
    return data == test

def word_suffix(x, suffix):
    """
    A Boolean-valued function that returns whether or not the words in the 
    input sequence "x" end in the input "suffix".
    """
    
    return [word.endswith(suffix) for word in x]

def word_prefix(x, prefix):
    """
    A Boolean-valued function that returns whether or not the words in the 
    input sequence "x" begin with the input "prefix".
    """
    
    return [word.startswith(prefix) for word in x]

def length(x):
    """
    Computes and returns the length of a word or sentence.  If it's a word, 
    just pass it bare to the function.  If a sentence, you should pass it 
    already parsed into a list of words.
    """
    
    return len(x)
