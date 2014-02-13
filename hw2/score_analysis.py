"""
score_analysis.py

Calculates the word-, sentence-, and mark-level scores using Collins- and SGD-
trained predictions, as well as "dumb guessing".
"""

import sys
import numpy as np
import dataproc as dp
import subroutines as sr

# get command line arguments for training method, score type
method, scoretype = sys.argv[1:3]
if method == 'dummy':
    dummy = 1
else:
    dummy = 0

# load weights
if method != 'dummy':
    weights_0p25 = np.load('training_output/'+method+'_w_0p25.npy')
    weights_0p5 = np.load('training_output/'+method+'_w_0p5.npy')
    weights_0p75 = np.load('training_output/'+method+'_w_0p75.npy')
else:
    weights_0p25 = np.load('training_output/collins_w_0p25.npy')
    weights_0p5 = np.load('training_output/collins_w_0p5.npy')
    weights_0p75 = np.load('training_output/collins_w_0p75.npy')
weights = [weights_0p25, weights_0p5, weights_0p75]

# load the test data
test_labels = dp.labels_as_ints('dataset/testLabels.txt')
test_sentences = dp.import_sentences('dataset/testSentences.txt')

# calculate the scores
score = []

for w in weights:
    if scoretype == 'word':
        score.append(sr.score_by_word(w,test_labels,test_sentences,dummy))
    elif scoretype == 'sentence':
        score.append(sr.score_by_sentence(w,test_labels,test_sentences,dummy))
    elif scoretype == 'mark':
        score.append(sr.score_by_mark(w,test_labels,test_sentences,dummy))
    else:
        print 'Not a valid method!\n'
        exit(0)

# save to file
f = open('scores/'+method+scoretype+'.txt', 'w')
f.write('Method: ' + method + ', Scoretype: ' + scoretype + '\n')

if method != 'mark':
    f.write('25\%: ' + str(score[0]) + '\n')
    f.write('50\%: ' + str(score[1]) + '\n')
    f.write('75\%: ' + str(score[2]) + '\n')
    f.close()
else:
    f.write('25\%:\n')
    for row in score[0,0]:
        for col in row:
            f.write(str(col) + '\t')
        f.write('\n')
    
    f.write('50\%:\n')
    for row in score[1,0]:
        for col in row:
            f.write(str(col) + '\t')
        f.write('\n')
    
    f.write('75\%:\n')
    for row in score[2,0]:
        for col in row:
            f.write(str(col) + '\t')
        f.write('\n')
