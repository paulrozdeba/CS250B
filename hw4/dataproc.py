import scipy.io
import numpy as np

def format_data():
    rtpol_neg_binarized = scipy.io.loadmat('codeDataMoviesEMNLP/data/rt-polaritydata/rt-polarity_neg_binarized.mat')
    rtpol_pos_binarized = scipy.io.loadmat('codeDataMoviesEMNLP/data/rt-polaritydata/rt-polarity_pos_binarized.mat')
    neg_list = []
    pos_list = []
    for i in range(5331):
        pos_sentence = list(rtpol_pos_binarized['allSNum'][0,i][0])
        pos_list.append(pos_sentence)
        neg_sentence = list(rtpol_neg_binarized['allSNum'][0,i][0])
        neg_list.append(neg_sentence)
    return neg_list,pos_list