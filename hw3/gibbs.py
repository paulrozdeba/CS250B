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
    
    #K = len(q[0])  # cardinality of the topic space
    K = q.shape[1]  # cardinality of the topic space
    
    #pvec = np.zeros(K)  # vector of probabilities
    
    # loop through the topic vector for entire corpus
    print 'Starting an epoch...'
    t0 = time.time()
    for m,v,(bi,word) in zip(doc_idx,voc_idx,enumerate(q)):
        for zi,count in enumerate(word):
            if count == 0.0:
                continue
            else:
                for wi in range(count):
                    pvec = prob_vec(K, zi, q, n, alpha, beta, voc_idx, m, v, bi)
                    psum = np.sum(pvec)
                    
                    # now draw a number btwn 0 and 1, and draw based on newp
                    draw = np.random.random_sample() * psum
                    int_hi = 0.0
                    for topic,topic_prob in enumerate(pvec):
                        int_hi += topic_prob
                        if draw < int_hi:
                            znew = topic
                            break
                    
                    # update q,n
                    # Note that the -=1 steps are unnecessary, since this
                    # already happens inside of prob_vec().
                    #q[bi][zi] -= 1
                    q[bi,znew] += 1.0
                    #n[m][zi] -= 1
                    n[m,znew] += 1.0
    print 'Epoch finished.  Time = ' + str(time.time()-t0)
    return q, n

def prob_vec(K, zi, q, n, alpha, beta, voc_idx, m, v, b):
    """
    Calculates the probability of all topics belonging to word i in document m.
    Calculation carried out term by term a la
    
    num1  *  num2
    -------------
    den1  *  den2
    
    K - Cardinality of the topic space.
    zi - Topic assignment for this word.
    q - List of lists of topic counts, by vocab word.  Each entry is a list of 
        topic assignment counts, listed by topic.
    n - List of lists of topic assignments, by document.  Each entry is a list 
        of topic counts, listed by topic.
    alpha, beta - Prior Dirichlet distritbution parameters.
    voc_idx - List of vocab indices per 0-th entry of q.
    m - This document's index.
    v - This word's vocabulary index.
    b - Bag index for q.  The bag index is the index in the bag-of-words
        vector for the *entire* corpus.
    """
    
    """
    # first pop the q and n entries for *this* word and *this* document
    # If the m,zi element of n is not zero, then at least this word was assigned 
    # with topic z.
    if n[m,zi] > 0:
        n[m,zi] -= 1
    # Same logic here, for q.
    if q[b,zi] > 0:
        q[b,zi] -= 1
    """
    # first pop the q and n entries for *this* word and *this* document
    n[m,zi] -= 1.0
    q[b,zi] -= 1.0
    
    num1 = np.resize(beta[v],K) + np.sum(q[voc_idx==v],axis=0)
    num2 = alpha + n[m]
    den1 = np.resize(np.sum(beta),K) + np.sum(q,axis=0)
    den2 = np.resize(np.sum(alpha),K) + np.sum(n,axis=0)
    
    """
    # start calculating
    for j in range(K):
        num1 = beta[v]
        den1 = sum(beta)
        for vi,elem in zip(voc_idx,q):
            den1 += elem[j]
            if vi == v:
                num1 += elem[j]
        
        num2 = alpha[j] + n[m][j]
        den2 = sum(alpha)
        for doc in n:
            den2 += doc[j]
    """    
    pvec = num1 * num2 / den1 / den2
    
    return pvec

### THE FUNCTION BELOW IS BROKEN, DO NOT USE ###
def prob(j, zi, q, n, alpha, beta, voc_idx, m, v, b):
    """
    Calculates the probability of topic j belonging to word i in document m.
    Calculation carried out term by term a la
    
    num1  *  num2
    -------------
    den1  *  den2
    
    j - Proposed topic index for this word.
    zi - Topic assignment for this word.
    q - List of lists of topic counts, by vocab word.  Each entry is a list of 
        topic assignment counts, listed by topic.
    n - List of lists of topic assignments, by document.  Each entry is a list 
        of topic counts, listed by topic.
    alpha, beta - Prior Dirichlet distritbution parameters.
    voc_idx - List of vocab indices per 0-th entry of q.
    m - This document's index.
    v - This word's vocabulary index.
    b - Bag index for q.  The bag index is the index in the bag-of-words
        vector for the *entire* corpus.
    """
    
    """
    # first pop the q and n entries for *this* word and *this* document
    # If the m,zi element of n is not zero, then at least this word was assigned 
    # with topic z.
    if n[m,zi] > 0:
        n[m,zi] -= 1
    # Same logic here, for q.
    if q[b,zi] > 0:
        q[b,zi] -= 1
    """
    
    # start calculating
    num1 = beta[v]
    den1 = sum(beta)
    for vi,elem in zip(voc_idx,q):
        den1 += elem.count(j)
        if vi == v:
            num1 += elem.count(j)
    
    num2 = alpha[j] + n[m].count(j)
    den2 = sum(alpha)
    for doc in n:
        den2 += doc.count(j)
    
    return (num1 * num2 / den1 / den2)
