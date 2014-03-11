import numpy as np
#import backprop as bp
from scipy.optimize import lbfgsb

def unfold(W1,b1,W2,b2,Wlabel):
    return np.hstack((W1.flatten(),b1.flatten(),W2.flatten(),b2.flatten(),Wlabel.flatten()))

def fold_up(d,label_size,flattened_array):
    W1 = flattened_array[:2*d*d].reshape((d,2*d))
    b1 = flattened_array[2*d*d:(2*d*d)+d]
    W2 = flattened_array[(2*d*d)+d:d+(4*d*d)].reshape((2*d,d))
    b2 = flattened_array[d+(4*d*d):(2*d)+(4*d*d)]
    Wlabel = flattened_array[(2*d)+(4*d*d):].reshape((label_size,d))
    return W1,b1,W2,b2,Wlabel

def compute_grad(flattened_array,d,label_size,lam_reg,alpha,neg_list,pos_list):
    """
    Oh god this shit is getting complicated
    """
    return flattened_grad