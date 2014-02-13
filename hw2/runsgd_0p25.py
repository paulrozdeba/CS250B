import SGD_CRF
import subroutines as sr
import dataproc as dp
import numpy as np

# first load and shuffle the data
train_labels_og = './dataset/trainingLabels.txt'
train_sentences_og = './dataset/trainingSentences.txt'

train_labels, train_sentences, validation_labels, validation_sentences = sr.shuffle_examples(train_labels_og, train_sentences_og, pct_train=0.25)

weights, scores, epoch, ept_avg = SGD_CRF.SGD_train(train_labels, train_sentences, 20,
                                                    validation_labels, validation_sentences, 'word')

np.save('sgd_w_0p25.npy', np.array(weights))
np.save('sgd_s_0p25.npy', np.array(scores))
np.save('sgd_ep_0p25.npy', np.array([epoch,ept_avg]))
