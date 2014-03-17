import numpy as np
import backprop as bp
import training as tr
import tree_maker as tm
import math

def compare_predict(W1,b1,W2,b2,Wlabel,pos_neg,example,vocab,normalized):
    if(pos_neg == 1):
        label = np.array([1.0,0.0])
    else:
        label = np.array([0.0,1.0])
    
    data = tm.build_tree(example,label,vocab,W1,W2,b1,b2,Wlabel,normalized)
    predictions = data[3]
    top_predict = predictions[-1]
    
    accurate = np.dot(label,top_predict)
    if(accurate >= 0.5):
        is_correct = 1.0
    else:
        is_correct = 0.0
    
    return is_correct


def validation_test(W1,b1,W2,b2,Wlabel,validation_neg,validation_pos,vocab,normalized):
    validation_accuracy = 0.0
    for neg_example in validation_neg:
        validation_accuracy += compare_predict(W1,b1,W2,b2,Wlabel,-1,neg_example,vocab,normalized) 
    for pos_example in validation_pos:
        validation_accuracy += compare_predict(W1,b1,W2,b2,Wlabel,1,pos_example,vocab,normalized)
    
    validation_accuracy /= float(len(validation_neg) + len(validation_pos))
    return validation_accuracy
    

def single_fold(neg_train,validation_neg,pos_train,validation_pos,label_size,alpha,lam_reg,vocab,normalized):
    d = np.size(vocab[0,:])
    
    (W1,b1,W2,b2,Wlabel,training_score) = tr.training_iterate(d,label_size,lam_reg,alpha,neg_train,pos_train,vocab,normalized)
    accuracy = validation_test(W1,b1,W2,b2,Wlabel,validation_neg,validation_pos,vocab,normalized)
    return accuracy

def k_fold(neg_list,pos_list,k,label_size,alpha,lam_reg,vocab,normalized):
    """
    neg_list -- all the negative examples after they've been formatted as in data_proc
    pos_list -- self-explanatory
    k -- how many folds of cross validation to do
    label_size-- the dimension of the output label
    alpha -- the value of the alpha parameter
    lam_reg -- the value for the regularization parameter
    vocab -- the Vxd matrix of word meanings
    normalized -- boolean describing whether or not to normalize the words
    """
    accuracy_array = np.zeros(k)
    chunk = len(neg_list)/k
    for i in range(k):
        lower = i*chunk
        upper = (i+1)*chunk
        validation_neg = neg_list[lower:upper]
        validation_pos = pos_list[lower:upper]
        neg_train = neg_list[:lower] + neg_list[upper:]
        pos_train = pos_list[:lower] + pos_list[upper:]
        accuracy_array[i] = single_fold(neg_train,validation_neg,pos_train,validation_pos,label_size,alpha,lam_reg,vocab,normalized)
    mean = np.mean(accuracy_array)
    SEM = math.sqrt(np.var(accuracy_array)/float(k))
    return mean,SEM
    
    
    