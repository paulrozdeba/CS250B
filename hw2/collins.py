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
    #J = 2832
    
    """
    # test for last tag indicator
    testsent = ['FIRSTWORD','Boom','LASTWORD']
    testlab = [0, 2, 1]
    x_info = ffs.sent_precheck(testsent)
    ffs.metaff(testlab[-2],testlab[-1],x_info,2)
    for i,(m1,m2) in enumerate(zip(testlab[0:-2],testlab[1:])):
        trueFF = ffs.metaff(m1,m2,x_info,i+1)
    """
    
    # now run it
    w = collins_epoch(train_labels[:1000], train_sentences[:1000], np.zeros(J))
    print w[-1]
    
    # make a prediction on a dummy sentence
    dummy = ['FIRSTWORD','How','are','you','doing','LASTWORD']
    g_dummy = sr.g(w[-1],dummy)
    U_dummy = sr.U(g_dummy)
    y_best = sr.bestlabel(U_dummy,g_dummy)
    
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
    w = np.zeros(shape=(Ntrain+1,J))  # to store the parameter trajectory
    w[0] = w0
    
    # track average number of true feature functions
    av_true = 0
    nevals = 0
    
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
        w[nex+1] = w[nex]
        x_info = ffs.sent_precheck(sentence)
        
        for i,(m1,m2,b1,b2) in enumerate(zip(label[:-1],label[1:],y_best[:-1],y_best[1:])):
            trueFF = ffs.metaff(m1,m2,x_info,i+1)
            bestFF = ffs.metaff(b1,b2,x_info,i+1)
            
            av_true += len(trueFF)
            nevals += 1
            
            for j in trueFF:
                w[nex+1,j] += 1
            for j in bestFF:
                w[nex+1,j] -= 1
                #continue
    
    print 'Average number of true FF\'s: ',(av_true/nevals)
    
    return w
        

################################################################################
if __name__ == '__main__':
    main()




