"""
collins.py

This is where the Collins perceptron algorithm is implemented.
"""

import numpy as np
import subroutines as sr
import dataproc as dp
from ffs import metaff

def main():
    # first import the training data
    train_labels = dp.import_labels('dataset/trainingLabels.txt')
    train_sentences = dp.import_sentences('dataset/trainingSentences.txt')
    
    # now run it
    collins_epoch(train_labels, train_sentences, np.zeros(160))

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
    w = np.zeros(shape=(Ntrain,J))  # to store the parameter trajectory
    w[0] = w0
    
    # need all possible tags
    tags = ['START','STOP','SPACE','PERIOD','COMMA','COLON','QUESTION_MARK',
            'EXCLAMATION_PT']
    
    for nex,example in enumerate(train_sentences):
        # first, calculate g
        g_ex = sr.g(metaff, w[nex], tags, example)
        
        print g_ex.shape

if __name__ == '__main__':
    main()
