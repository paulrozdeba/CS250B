import gibbs
import dataproc
import numpy as np

"""
ml_estimate.py

This module computes an estimate of the topic distribution parameters for each m,
and the word distribution parameters for each topic.
"""

def calc_theta(n,c_vec):
    """
    calculates the the multinomial parameter vector for each document's topic
    distribution
    
    n - MxK array of topic assignments.  Each row is a document, each column of
        a row is the count of topic assignments for a specific topic in that
        document
    c_vec - Vector of size K (cardinality of topic set) that sets the pseudcount
        for each topic.  Constant across document.  The sum across the elements
        should be much smaller than the size of any document. Is also known as
        alpha
    """
    
    M = np.shape(n)[0] #Number of documents
    K = np.shape(n)[1] #Number of topics
    c_sum = np.sum(c_vec)
    
    #theta_mat = np.zeros((M,K))
    theta_mat = np.copy(n)
    T_vec = np.sum(theta_mat,axis=1)
        
    #normalize
    for m in range(M):
        theta_mat[m,:] += c_vec
        theta_mat[m,:] /= T_vec[m] + c_sum
    return theta_mat

def calc_phi(q,c_vec,voc_idx,V):
    """
    calculates the the multinomial parameter vector for each topic's word
    distribution
    
    q - SxK word to topic assignments.  Each row is a different word in a 
        different document.
    c_vec - Vector of size V (cardinality of vocabulary) that sets the pseudcount
        for each word.  Constant across topic.  The sum across the elements
        should be much smaller than the size of any document.  Is also the beta
        parameter vector.
    V - the cardinality of the vocabulary
    """
    S = len(voc_idx)
    K = np.shape(q)[1] #number of topics
    phi_mat = np.zeros(K,V)
    c_sum = np.sum(c_vec)
    
    #fill out the phi-matrix
    for s in range(S):
        v = voc_idx[s]
        for k in range(K):
            phi_mat[k,v] += q 
    
    topic_totals = np.sum(phi_mat,axis=1)
    
    #normalize
    for k in range(K):
        phi_mat[k,:] += c_vec
        phi_mat[k,:] /= topic_totals[k] + c_sum
        
    return phi_mat
    
    
    