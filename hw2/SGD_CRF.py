"""
This is where the stochastic gradient descent training algorithm is implemented
"""

import subroutines as sr
import ffs as ff
import numpy as np
import dataproc as dp
import collins as collins
import time

def SGD_train(train_labels,train_sentences,max_epoch,validate_labels,validate_sentences,method):
    """
    The main subroutine for training by SGD training, train_labels and
    train_sentence are list of lists of corresponding (y,x) pairs.
    validate_labels and validate_sentences are similar list of lists for doing 
    early stopping regularization.
    """
    #max_epoch = 5
    converged = 0
    lr = 1.0
    
    num_training = len(train_sentences)
    order = np.arange(0, num_training, dtype='int') # ordering of training ex's
    
   
    J = ff.calcJ()
    #two weightvectors for bookkeeping, and a dw vector for updates, we initialize
    #the weightvector at zero
    #w_traj = np.zeros(shape=(max_epoch+1,J))
    weights = np.zeros(J)
    #w_traj[0,:] = weights
    old_weights = np.zeros(J)
    dw = np.zeros(J)
    score = 0.0
    
    for epoch in range(max_epoch):
        #print "epoch number {}".format(epoch)
        i = 0
        #put something in here about learning rate?
        np.random.shuffle(order)  # randomly shuffle test data
        old_weights = np.copy(weights)
        for ind_ex in order:
            #print "Now processing sample {} of {}".format(i,num_training)
            x = train_sentences[ind_ex]
            y = train_labels[ind_ex]
            dw = compute_gradient(x,y,weights,dw)
            weights += lr*dw 
            i += 1
        #convergence test, remove this commentary when we have a score function
        #in collins module
        new_score = sr.general_score(weights,validate_labels,validate_sentences,method,0)
        if (new_score > score):
            #the validation score has increased
            score = new_score
        else:
            #validation error has increased, early stopping dictates we stop
            #training and use the old weights
            converged = 1
            weights = np.copy(old_weights)
            break
    
    #score = 0.0
    return weights,score,epoch

def compute_gradient(x,y,w,dw):
    """
    This term computes the gradient vector
    """
    dw *= 0.0
    
    
    #get info about this training example
    x_info = ff.sent_precheck(x)
    N = len(x)
    
    #compute some necessary arrays of factors from the subroutines module
    g = sr.g(w,x)
    e_g = np.exp(g)
    alpha = sr.alpha_mat(g)
    beta = sr.beta_mat(g)
    z = sr.Z(alpha,beta)
    #iterate over position in sentence and tag values, getting a list of indices
    # of feature functions to update at for each position and tag pair value.
    for i in range(np.shape(g)[0]):
        for m1 in range(8):
            for m2 in range(8):
                factor = alpha[i,m1]*beta[i+1,m2]*e_g[i,m1,m2]/z
                #get list of non-zero (and thus =1) f_j for (i,m1,m2)
                #print m1,m2,x_info,i
                trueFF = ff.metaff(m1,m2,x_info,i+1)
                #add the weighting factor to them
                for j in trueFF:
                    dw[j] -= factor
                
    
    #now I go through and use data from y to compute the "true" value of F_J,
    #once more iterating over i, but not m1,m2 (instead getting those values
    #from the supplied y
    for i in range(1,N):
        trueFF = ff.metaff(y[i-1],y[i],x_info,i)
        for j in trueFF:
                    dw[j] += 1
    return dw