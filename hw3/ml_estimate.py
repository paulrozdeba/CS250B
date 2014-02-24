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
    
    n - List of lists of topic assignments, by document.  Each entry is a list 
        of topic counts, listed by topic.
    c_vec - Vector of size K (cardinality of topic set) that sets the pseudcount
        for each topic.  Constant across document.  The sum across the elements
        should be much smaller than the size of any document.
    """
    
    M = len(n) #Number of documents
    K = len(n[0]) #Number of topics
    c_sum = np.sum(c_vec)
    
    theta_mat = np.zeros((M,K))
    for m in range(M):
        for k in range(K):
            theta_mat[m,k] = n[m][k]
    T_vec = np.sum(theta_mat,axis=1)
        
    #normalize
    for m in range(M):
        theta_mat[m,:] += c_vec
        theta_mat[m,:] /= T_vec[m] + c_sum
    return theta_mat

def calc_phi(q,c_vec,M):
    """
    calculates the the multinomial parameter vector for each topic's word
    distribution
    
    q - List of lists of topic counts, by vocab word.  Each entry is a list of 
        topic assignment counts, listed by topic.
    c_vec - Vector of size V (cardinality of vocabulary) that sets the pseudcount
        for each word.  Constant across topic.  The sum across the elements
        should be much smaller than the size of any document.
    M - the number of documents
    """
    V = len(q)/M #size of the vocabulary
    K = len(q[0]) #number of topics
    phi_mat = np.zeros(K,V)
    c_sum = np.sum(c_vec)
    
    #fill out the phi-matrix
    for m in range(M):
        for v in range(V):
            sub_list = q[m*M + v]
            for k in range(K):
                phi_mat[k,v] += sub_list[k]
    
    topic_totals = np.sum(phi_mat,axis=1)
    
    #normalize
    for k in range(K):
        phi_mat[k,:] += c_vec
        phi_mat[k,:] /= topic_totals[k] + c_sum
        
    return phi_mat
    
    
    