"""
gibbs.py

Gibbs sampling code for LDA problem.
Code is split up so individual epochs can be run over the data set.
"""

import numpy as np

def gibbs_epoch(q, n, alpha, beta, doc_idx, voc_idx):
    """
    Runs one epoch of Gibbs sampling.  This means that the topic distribution 
    is updated once for each word in the corpus of documents.
    
    z - List of lists of topic assignments.  This is organized as:
            0-th index corresponds to a vocab word in a document.
            Each entry is a list of topic assignment counts.  This list should 
            sum to the total count for that vocab word, in that document.
    q - List of lists of topic counts, by vocab word.  Each entry is a list of 
        topic assignment counts, listed by topic.
    n - List of lists of topic assignments, by document.  Each entry is a list 
        of topic counts, listed by topic.
    alpha, beta - Prior Dirichlet distritbution parameters.
    doc_idx - List of document indices per 0-th entry of q.
    voc_idx - List of vocab indices per 0-th entry of q.
    """
    
    K = len(q[0])  # cardinality of the topic space
    
    # loop through the topic vector for entire corpus
    for bi,word in enumerate(q):
        for zi,count in enumerate(word):
            for wi in range(count):
                m = doc_idx[bi]
                v = voc_idx[bi]
                newp = prob_vec(K, zi, q, n, alpha, beta, doc_idx, voc_idx, 
                                m, v, bi)
                
                # now draw a number btwn 0 and 1, and draw based on newp
                draw = np.random.rand()
                int_hi = 0.0
                for topic,topic_prob in enumerate(newp):
                    int_hi += topic_prob
                    if draw < int_hi:
                        znew = topic
                        break
                
                # update q,n
                q[bi,zi] -= 1
                q[bi,znew] += 1
                n[m,zi] -= 1
                n[m,znew] += 1
    
    return q, n

def prob(j, zi, q, n, alpha, beta, doc_idx, voc_idx, m, v, b):
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
    doc_idx - List of document indices per 0-th entry of q.
    voc_idx - List of vocab indices per 0-th entry of q.
    m - This document's index.
    v - This word's vocabulary index.
    b - Bag index for q.  The bag index is the index in the bag-of-words
        vector for the *entire* corpus.
    """
    
    # first pop the q and n entries for *this* word and *this* document
    # If the m,zi element of n is not zero, then at least this word was assigned 
    # with topic z.
    if n[m,zi] > 0:
        n[m,zi] -= 1
    # Same logic here, for q.
    if q[b,zi] > 0:
        q[b,zi] -= 1
    
    # start calculating
    num1 = 0
    den1 = 0
    for vi,elem in zip(voc_idx,q):
        den1 += elem.count(j) + beta[vi]
        if vi == v:
            num1 += elem.count(j) + beta[vi]
    
    num2 = 0
    den2 = 0
    for mi,doc in zip(doc_idx,n):
        den2 += doc.count(j) + alpha[mi]
        if mi == m:
            num2 += doc.count(j) + alpha[mi]
    
    return (num1*num2) / (den1*den2)

def prob_vec(K, zi, q, n, alpha, beta, doc_idx, voc_idx, m, v, b):
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
    doc_idx - List of document indices per 0-th entry of q.
    voc_idx - List of vocab indices per 0-th entry of q.
    m - This document's index.
    v - This word's vocabulary index.
    b - Bag index for q.  The bag index is the index in the bag-of-words
        vector for the *entire* corpus.
    """
    
    # first pop the q and n entries for *this* word and *this* document
    # If the m,zi element of n is not zero, then at least this word was assigned 
    # with topic z.
    if n[m,zi] > 0:
        n[m,zi] -= 1
    # Same logic here, for q.
    if q[b,zi] > 0:
        q[b,zi] -= 1
    
    pvec = []
    # start calculating
    for j in range(K):
        num1 = 0
        den1 = 0
        for vi,elem in zip(voc_idx,q):
            den1 += elem.count(j) + beta[vi]
            if vi == v:
                num1 += elem.count(j) + beta[vi]
        
        num2 = 0
        den2 = 0
        for mi,doc in zip(doc_idx,n):
            den2 += doc.count(j) + alpha[mi]
            if mi == m:
                num2 += doc.count(j) + alpha[mi]
        
        pvec.append(num1 * num2 / den1 / den2)
    
    return pvec
