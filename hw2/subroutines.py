"""
subroutines.py

Contains several routines 
"""

import numpy as np
import itertools as it
import ffs

def g(w, x):
    """
    Calculates g functions for each pair of tags y in a sentence x.
    
    w - List of weights associated with ffs.
    x - Sequence (sentence) over which to evaluate g.
    
    Returns: a (N-1) X M X M matrix where N is the length of the sentence, and 
    there are M possible tags for each word.
    """
    
    M = 8  # number of possible tags
    N = len(x)  # length of sentence
    
    __g__ = np.zeros(shape=(N-1,M,M))
    
    # preprocess the sentence
    x_info = ffs.sent_precheck(x)
    
    for i,m1,m2 in it.product(range(1,N),range(0,M),range(0,M)):
        # get the nonzero feature function indices for this tag pair
        trueFF = ffs.metaff(m1,m2,x_info,i)
        # fill in the nonzero elements of g
        for j in trueFF:
            __g__[i-1,m1,m2] += w[j]
    
    return __g__

def forward(e_g, k, v,alpha_mat):
    """
    e_g is the g array exponentiated, k is the current index in the tag sequence
    and v is the y_i
    """
    if(k==0):
        total = (v==0)
    else:
        total = 0
        for i in range(e_g.shape[1]):
            total += float(forward(e_g,k-1,i,alpha_mat))*e_g[k,i,v]
    return total

def backward(e_g, v, k,beta_mat):
    """
    e_g is the g array exponentiated, k is the current index in the tag sequence
    and v is the y_(i-1)
    """
    length = e_g.shape[0]-1
    if(k==length):
        total = (v==1)
    else:
        total = 0
        for i in range(e_g.shape[1]):
            total += float(backward(e_g,i,k+1,beta_mat))*e_g[k+1,v,i]
    return total
    
def alpha_mat(g):
    e_g = np.exp(g)
    N = e_g.shape[0] + 1
    M = e_g.shape[1]
    alpha_mat = np.zeros((N,M))
    for k in range(N):
        for v in range(M):
            if (k==0):
                alpha_mat[k,v] = float((v==0))
            else:
                alpha_mat[k,v] = np.dot(alpha_mat[k-1,:],e_g[k-1,:,v])
    return alpha_mat
     
def beta_mat(g):
    e_g = np.exp(g)
    N = e_g.shape[0] + 1
    M = e_g.shape[1]
    beta_mat = np.zeros((N,M))
    for k in range(N)[::-1]:
        for u in range(M):
            if (k==(N-1)):
                beta_mat[k,u] = float((u==1))
            else:
                beta_mat[k,u] = np.dot(beta_mat[k+1,:],e_g[k,u,:]) 
    return beta_mat

def U(g):
    """
    Calculates the matrix elements of the propagator U(k,v).
    
    g - The matrix elements of g for the entire sequence x.
    """
    
    N = g.shape[0]  # number of pairs in sentence
    M = g.shape[1]  # number of possible tags
    assert(M == g.shape[2])  # just a consistency check on the shape
    __U__ = np.zeros(shape=(N,M))
    
    for i in range(N):
        __U__[i] = __U_singlek__(g,i)
    
    return __U__
        
def __U_singlek__(g, k):
    """
    Supplementary function, for use inside U(g) only.
    """
    
    # implement recursion here
    if k == 0:
        #return np.amax(g[0], axis=0)
        return g[0,0]
    else:
        return np.amax(__U_singlek__(g,k-1) + g[k], axis=0)

def Z(alpha_matrix,beta_matrix):
    Z = np.zeros(alpha_matrix.shape[0])
    for k in range(alpha_matrix.shape[0]):
        Z[k] = np.dot(alpha_matrix[k,:],beta_matrix[k,:])
    return np.mean(Z)

def bestlabel(U, g):
    """
    Predicts the best label for an example, based on U and g matrices.
    """
    
    N = U.shape[0]  # length of sentence
    assert(g.shape[0] == N)
    y = []  # to store the label
    
    # first find the best tag at position n
    y_N = np.argmax(U[-1])
    y.append(y_N)
    ykbest = y_N
    
    # now get the rest of the label
    for U_km1,g_k in zip(U[::-1][1:],g[::-1][:-1]):
        y_km1_best = np.argmax(U_km1 + g_k[:,ykbest])
        y.append(y_km1_best)
        ykbest = y_km1_best
    y.append(0)
    
    return y[::-1]

def dummy_predict(x):
    y=[]
    for i in range(len(x)):
        if(i==0):
            y.append(0)
        elif(i==(len(x)-2)):
            y.append(4)
        elif(i==(len(x)-1)):
            y.append(1)
        else:
            y.append(2)
    return y_dummy

def score_by_word(weights,score_labels,score_sentences,dummy=0):
    """
    Calculates the percentage of words properly punctuated
    """
    N_validate = len(score_labels)
    N_words = 0.0
    Num_correct = 0.0
    for i in range(N_validate):
        y = score_labels[i]
        x = score_sentences[i]
        N_words += len(x)
        g_test = g(weights,x)
        #print g_test
        U_test = U(g_test)
        if(dummy==0):
            y_predict = bestlabel(U_test,g_test)
        else:
            y_predict = dummy_predict(x)
        for j in range(len(y)):
            if(y[j] == y_predict[j]):
                Num_correct += 1.0
    return Num_correct/N_words

def score_by_sentence(weights,score_labels,score_sentences,dummy):
    """
    Returns the percentage of sentences that are completely accurately predicted
    """
    N_validate = len(score_labels)
    num_correct = 0.0
    for i in range(N_validate):
        y = score_labels[i]
        x = score_sentences[i]
        g_test = g(weights,x)
        U_test = U(g_test)
        if(dummy==0):
            y_predict = bestlabel(U_test,g_test)
        else:
            y_predict = dummy_predict(x)
        if(y==y_predict):
            num_correct += 1.0
    return num_correct / float(N_validate)

def score_by_mark(weights,score_labels,score_sentences,dummy):
    """
    Returns several things
    """
    N_validate = len(score_labels)
    score_mat = np.zeros((8,8))
    predict_totals = np.zeros(8)
    true_totals = np.zeros(8)
    percentage_mat = np.zeros(8)
    accuracy_vec = np.zeros(8)
    for i in range(N_validate):
        y = score_labels[i]
        x = score_sentences[i]
        g_test = g(weights,x)
        U_test = U(g_test)
        if(dummy==0):
            y_predict = bestlabel(U_test,g_test)
        else:
            y_predict = dummy_predict(x)
        for i in range(len(y)):
            score_mat[y[i],y_predict[i]] += 1.0
    predict_totals = np.sum(score_mat,axis=0)
    true_totals = np.sum(score_mat,axis=1)
    for i in range(8):
        percentage_mat[i,:] = score_mat / true_totals[i]
        accuracy_vec[i] = percentage_mat[i,i]
    return percentage_mat,accuracy_vec

def general_score(weights,score_labels,score_sentences,method,dummy):
    if (method=='word'):
        return score_by_word(weights,score_labels,score_sentences,dummy)
    elif (method=='sentence'):
        return score_by_sentence(weights,score_labels,score_sentences,dummy)