"""
collins.py

This is where the Collins perceptron algorithm is implemented.
"""

import numpy as np
import subroutines as sr
import dataproc as dp
import ffs
import time

def collins(train_labels, train_sentences, validation_labels, 
            validation_sentences, pct_train=0.5, Nex=None):
    """
    Runs the Collins perceptron training on the input training data.
    
    labels - All training, validation labels.
    sentences - All training, validation sentences.
    pct_train - Percentage of examples from data set to use as training data.
             The rest are used as validation data.
    """
    
    # get J, the total number of feature functions
    J = ffs.calcJ()
    print 'J = ',J
    
    # now run it
    scores = []
    w0 = np.zeros(J)
    print 'Calculating initial score...'
    scores.append(sr.score_by_word(w0,validation_labels,validation_sentences))
    print 'Done!\n'
    # run until converged, according to score on validation set
    nep = 1
    epoch_time = []
    
    print 'Initiating Collins perceptron training.'
    while True:
        print 'Epoch #',nep,'...'
        t0 = time.time()
        # get the new weights & score
        print 'Training...'
        w1 = collins_epoch(train_labels, train_sentences, w0)
        print 'Done.\n'
        epoch_time.append([time.time() - t0])
        
        t0 = time.time()
        print 'Calculating new score...'
        scores.append(sr.general_score(w1,validation_labels,validation_sentences,'word',0))
        print 'Done.\n'
        epoch_time[nep-1].append(time.time() - t0)
        
        # decide if converged
        if scores[nep] < scores[nep-1]:
            break
        else:
            w0 = w1
        nep += 1
        
    print 'Training complete!\n'
    
    """
    # make a prediction on a dummy sentence
    #dummy = ['FIRSTWORD','I','like','cheese','but','I','also','like','bread','LASTWORD']
    dummy = ['FIRSTWORD','Do','you','like','cheese','LASTWORD']
    g_dummy = sr.g(w,dummy)
    U_dummy = sr.U(g_dummy)
    y_best = sr.bestlabel(U_dummy,g_dummy)
    """
    
    # now return final weights, score time series, and epoch timing
    return w0, scores, epoch_time

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
    #sentences_self = train_sentences[:100]
    #labels_self = train_labels[:100]
    
    # track average number of true feature functions
    av_true = 0.0
    nevals = 0.0
    
    for nex,(sentence,label) in enumerate(zip(train_sentences,train_labels)):
        if (nex+1)%1000 == 0:
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




