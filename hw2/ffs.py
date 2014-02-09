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
import itertools as it

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
    x_info['length'] = len(x)
    
    # First, define all word dictionaries.
    interrogatives = ['which','what','when','who','where','why','whom','whose',
                    'how','whether']
    conjunctions = [None, 'after', 'although', 'and', 'as', 'because', 'before', 
                    'but', 'except', 'if', 'like', 'nor', 'now', 'once', 'or', 
                    'since', 'so', 'than', 'that', 'though', 'unless', 'until', 
                    'when', 'where', 'whether', 'while']
    suffixes = [None,'able','ible','ial','al','ed','en','er','est','ful','ic','ing',
            'ation','ition','tion','ion','ity','ty','itive','ative','ive','less'
            ,'ly','ment','ness','ious','eous','ous','es']
    prefixes = [None,'anti','de','dis','en','em','fore','intra','inter','in','im','il','ir',
            'mid','mis','non','over','pre','re','semi','sub','super','trans'
            ,'under','un','post']
    #store the size of these dictionaries for later book keeping
    x_info['num_conjunctions'] = len(conjunctions)
    x_info['num_suffixes'] = len(suffixes)
    x_info['num_prefixes'] = len(prefixes)
    
    #allocate arrays for word level checks
    x_conjunctions = np.zeros(len(x))
    x_suffixes = np.zeros(len(x))
    x_prefixes = np.zeros(len(x))
    x_capitals = np.zeros(len(x))
    
    # First check: capitalization.
    for i,word in enumerate(x):
        if word[0].isupper():
            x_capitals[i] = 1
        else:
            x_capitals[i] = 0
    x_info['capitals'] = x_capitals

    
    # now lowercase this shit
    x = [word.lower() for word in x]
    
    #check for interrogative
    if (x[1] in interrogatives):
        interrogative_test = 1
    else:
        interrogative_test = 0
    
    x_info['begins_with_interrogative'] = interrogative_test
    
    for i,word in enumerate(x):
        # check conjunctions
        if word in conjunctions:
            x_conjunctions[i] = x.index(word)
        else:
            x_conjunctions[i] = 0
    x_info['conjunctions'] = x_conjunctions
    
    #check prefixes
    for i,word in enumerate(x):
        for j in range(1,len(prefixes)):
            if (word.startswith(prefixes[j])):
                x_prefixes[i] = j
                break
    
    x_info['prefixes'] = x_prefixes
    
    #check suffixes
    for i,word in enumerate(x):
        for j in range(1,len(suffixes)):
            if (word.endswith(suffixes[j])):
                x_suffixes[i] = j
                break
    
    x_info['suffixes'] = x_suffixes
    
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
    Np = x_info['num_prefixes']  # number of prefixes
    Ns = x_info['num_suffixes']  # number of suffixes
    Nc = x_info['num_conjunctions']  # number of conjunctions
    Ncap = 2
    Ninterr = 2
    
    CLASS_SIZES = [Np, Ns, Nc, Ncap, Ninterr]
    
    ############
    # SINGLE TAG INDICATORS (STI)
    STI1 = m1
    STI2 = m2
    ############
    # WORD-LEVEL DICTIONARY INDICATORS
    PREFIX = x_info['prefixes'][i-1:i+1]
    SUFFIX = x_info['suffixes'][i-1:i+1]
    CONJUNCTION = x_info['conjunctions'][i-1:i+1]
    CAPITALIZED = x_info['capitals'][i-1:i+1]
    ############
    # SENTENCE-LEVEL INDICATORS
    INTERROGATIVE = x_info['begins_with_interrogative']
    ############
    
    # now start filling out whole list of nonzero feature functions
    TAGS = [STI1, STI2]
    ALLIND = [PREFIX, SUFFIX, CONJUNCTION, CAPITALIZED, [INTERROGATIVE]]
    ALLIND_flat = [item for sublist in ALLIND for item in sublist]
    
    N_wordlevel = 4  # number of word-level classes
    
    trueFF = []
    
    # first do the single-word shit
    # priors on tags
    nstart = 0
    trueFF = [m1]
    nstart += M
    trueFF.append(m2)
    nstart += M
    
    # now single tag single word indicators
    # THIS IS WHERE SHIT GETS REAL
    for j,(k,l) in it.product(TAGS,enumerate(ALLIND_flat)):
        trueFF.append(int(nstart + j + l*M))
        if k < N_wordlevel:
            nstart += int(M * CLASS_SIZES[int(k/2)])
        else:
            nstart += int(M * CLASS_SIZES[int(N_wordlevel-1 + int(k/2)-N_wordlevel)])
    
    J = nstart
    return trueFF, J






