"""
gibbs.py

Gibbs sampling code for LDA problem.
Code is split up so individual epochs can be run over the data set.
"""

import numpy as np
import time

def gibbs_epoch(q, n, alpha, beta, doc_idx, voc_idx):
    """
    Runs one epoch of Gibbs sampling.  This means that the topic distribution 
    is updated once for each word in the corpus of documents.
    
    z - List of lists of topic assignments.  This is organized as:
            0-th index corresponds to a vocab word in a document.
            Each entry is a list of topic assignment counts.  This list should 
            sum to the total count for that vocab word, in that document.
    q - List of lists of topic counts, by vocab word.  Each entry is a list of 
        topic assignment counts, in order of topics.
    n - List of lists of topic assignments, by document.  Each entry is a list 
        of topic counts, in order of topics.
    alpha, beta - Prior Dirichlet distritbution parameters.
    doc_idx - List of document indices per 0-th entry of q.
    voc_idx - List of vocab indices per 0-th entry of q.
    """
    
    K = len(q[0])  # cardinality of the topic space
            den1 = sum(beta)
        for vi,elem in zip(voc_idx,q):
            den1 += elem[j]
            if vi == v:
                num1 += elem[j]
        
        num2 = alpha[j] + n[m][j]
        den2 = sum(alpha)
        for doc in n:
            den2 += doc[j]
        
        num1 = float(num1)
        num2 = float(num2)
        den1 = float(den1)
        den2 = float(den2)
        pvec.append(num1 * num2 / den1 / den2)
    
    return pvec
