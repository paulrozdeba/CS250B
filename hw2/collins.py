"""
collins.py

This is where the Collins perceptron algorithm is implemented.
"""

import numpy as np
import subroutines as sr
import dataproc as dp
import ffs
import time

def collins(train_labels_f, train_sentences_f, test_labels_f, test_sentences_f, 
            pct_train=0.5, Nex=None):
    """
    Runs the Collins perceptron training on the input training data.
    
    labels - Path to the file containing the training labels.
    sentences - Path to the file containing training sentences.
    pct_train - Percentage of examples from data set to use as training data.
             The rest are used as validation data.
    """
    
    # load shuffled data sets
    train_labels, train_sentences, validation_labels, validation_sentences = sr.shuffle_examples(train_labels_f, train_sentences_f, pct_train)
    
    # load dictionary of tags <--> ints
    tag_dict = dp.export_dict()
    tag_dict['START'] = 0
    tag_dict['STOP'] = 1
    tag_dict_inverse = dp.export_dict_inverse()
    tag_dict_inverse[0] = 'START'
    tag_dict_inverse[1] = 'STOP'
    
    # get J, the total number of feature functions
    J = ffs.calcJ()
    print 'J = ',J
    
    # now run it
    w0 = np.zeros(J)
    # run until converged, according to 
    w1 = collins_epoch(train_labels[:1000], train_sentences[:1000], np.zeros(J))
    
    """
    # make a prediction on a dummy sentence
    #dummy = ['FIRSTWORD','I','like','cheese','but','I','also','like','bread','LASTWORD']
    dummy = ['FIRSTWORD','Do','you','like','cheese','LASTWORD']
    g_dummy = sr.g(w,dummy)
    U_dummy = sr.U(g_dummy)
    y_best = sr.bestlabel(U_dummy,g_dummy)
    """
    
    print y_best
    
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
    #w = np.zeros(shape=(Ntrain+1,J))  # to store the parameter trajectory
    #w[0] = w0
    
    # pick out a random subset of training examples
    sentences_self = train_sentences[:100]
    labels_self = train_labels[:100]
    
    # track average number of true feature functions
    av_true = 0.0
    nevals = 0.0
    
    for nex,(sentence,label) in enumerate(zip(train_sentences,train_labels)):
        if (nex+1)%100 == 0:
            print nex + 1
        
        # first, calculate g
        g_ex = sr.g(w0,sentence)
        
        # now U
        U_ex = sr.U(g_ex)
        
        # find the best label
        y_best = sr.bestlabel(U_ex,g_ex)
        
        # update the weight
        #w[nex+1] = w[nex]
        x_info = ffs.sent_precheck(sentence)
        
        for i,(m1,m2,b1,b2) in enumerate(zip(label[:-1],label[1:],y_best[:-1],y_best[1:])):
            trueFF = ffs.metaff(m1,m2,x_info,i+1)
            bestFF = ffs.metaff(b1,b2,x_info,i+1)
            
            av_true += float(len(trueFF))
            nevals += 1.0
            
            for j in trueFF:
                #w[nex+1,j] += 1
                w0[j] += 1
            for j in bestFF:
                #w[nex+1,j] -= 1
                w0[j] -= 1
                #continue
    
    print 'Average number of true FF\'s: ',(av_true/nevals)
    
    return w0
        

################################################################################
if __name__ == '__main__':
    main()




