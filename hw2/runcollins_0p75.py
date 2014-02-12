import collins
import subroutines as sr
import dataproc as dp

# first load and shuffle the data
train_labels_og = './dataset/trainingLabels.txt'
train_sentences_og = './dataset/trainingSentences.txt'

train_labels, train_sentences, validation_labels, validation_sentences = sr.shuffle_examples(train_labels_og, train_sentences_og, pct_train=0.75)

weights, scores, epoch_times = collins.collins(train_labels, train_sentences,
                                               validation_labels, validation_sentences)

np.save('collins_w_0p75.npy', np.array(weights))
np.save('collins_s_0p75.npy', np.array(scores))
np.save('collins_ept_0p75.npy', np.array(epoch_times))
