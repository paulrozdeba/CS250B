"""
collins.py

This is where the Collins perceptron algorithm is implemented.
"""

import numpy as np
import subroutines as sr
import dataproc as dp
import ffs
import time

def main():
    # first import the training data
    train_labels = dp.labels_as_ints('dataset/trainingLabels.txt')
    train_sentences = dp.import_sentences('dataset/trainingSentences.txt')
    
    # get J, the total number of feature functions
    J = ffs.calcJ()
    
    # now run it
    w = collins_epoch(train_labels[:100], train_sentences[:100], np.zeros(J))
    
    # make a prediction on a dummy sentence
    dummy = ['FIRSTWORD','Hello','James','LASTWORD']
    g_dummy = sr.g(w[-1],dummy)
    U_dummy = sr.U(g_dummy)
    y_best = sr.bestlabel(U_dummy,g_dummy)
    print dummy, y_best
    
    exit(0)

def collins_epoch(train_labels, train_sentences, w0):
    """
    This function is a single epoch of the Collins perceptron.  An epoch ends 
    after every example in the (shuffled) training data has been visited once.
    
    train_labels - A list containing ALL of the labels in the training data.
    train_sentences - A list of ALL sentences in the training data.
    w0 - The initial parameter values, at the start of the epoch.
    """
    
    Ntrain = len(train_sentences)  # number of training examples
    assert(Ntrain == len(train_labels))
    J = len(w0)  # number of parameters, equal to number of feature functions
    w = np.zeros(shape=(Ntrain+1,J))  # to store the parameter trajectory
    w[0] = w0
    
    for nex,(sentence,label) in enumerate(zip(train_sentences,train_labels)):
        print nex
        # first, calculate g
        g_ex = sr.g(w0,sentence)
        
        # now U
        U_ex = sr.U(g_ex)
        
        # find the best label
        y_best = sr.bestlabel(U_ex,g_ex)
        
        # update the weight
        x_info = ffs.sent_precheck(sentence)
        w[nex+1] = w[nex]
        
        for i,(m1,m2,b1,b2) in enumerate(zip(label[0:-2],label[1:],y_best[0:-2],y_best[1:])):
            trueFF = ffs.metaff(m1,m2,x_info,i+1)
            bestFF = ffs.metaff(b1,b2,x_info,i+1)
            for j in trueFF:
                w[nex+1,j] += 1
            for j in bestFF:
                w[nex+1,j] -= 1
    
    return w
        

################################################################################
if __name__ == '__main__':
    main()




