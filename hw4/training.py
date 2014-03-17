import numpy as np
import tree_maker as tm
#import backprop as bp
from scipy.optimize import fmin_l_bfgs_b

def unfold(W1,b1,W2,b2,Wlabel):
    return np.hstack((W1.flatten(),b1.flatten(),W2.flatten(),b2.flatten(),Wlabel.flatten()))

def fold_up(d,label_size,flattened_array):
    W1 = flattened_array[:2*d*d].reshape((d,2*d))
    b1 = flattened_array[2*d*d:(2*d*d)+d]
    W2 = flattened_array[(2*d*d)+d:d+(4*d*d)].reshape((2*d,d))
    b2 = flattened_array[d+(4*d*d):(3*d)+(4*d*d)]
    Wlabel = flattened_array[(3*d)+(4*d*d):].reshape((label_size,d))
    return W1,b1,W2,b2,Wlabel

def compute_grad(flattened_array,d,label_size,lam_reg,alpha,neg_list,pos_list,vocab,normalized):
    """
    Oh G-d this shit is getting complicated
    """
    (W1,b1,W2,b2,Wlabel) = fold_up(d,label_size,flattened_array)
    (W1_grad,b1_grad,W2_grad,b2_grad,Wlabel_grad) = bp.full_grad(W1,b1,W2,b2,Wlabel,lam_reg,alpha,neg_list,pos_list,vocab,normalized)
    flattened_grad = unfold(W1_grad,b1_grad,W2_grad,b2_grad,Wlabel_grad)
    return flattened_grad
    
def full_j(flattened_array,d,label_size,lam_reg,alpha,neg_list,pos_list,vocab,normalized):
    
    scale = 1.0/float(len(neg_list) + len(pos_list))
    (W1,b1,W2,b2,Wlabel) = fold_up(d,label_size,flattened_array)
    big_j = np.linalg.norm(flattened_array)**2
    big_j *= 0.5*lam_reg
    (whole_rec,whole_pred) = tm.whole_error(neg_list,pos_list,vocab,W1,b1,W2,b2,Wlabel,normalized)
    big_j += scale*((alpha*whole_rec)+((1.0-alpha)*whole_pred))
    return big_j

def training_iterate(d,label_size,lam_reg,alpha,neg_train,pos_train,vocab,normalized):
    """
    My eyes are bleeding!
    """
    parameter_size = (4*d*d) + (2*d) + (d*label_size)
    x0 = np.random.randn(parameter_size)
    myargs = (d,label_size,lam_reg,alpha,neg_train,pos_train,vocab,normalized)
    result =  fmin_l_bfgs_b(full_j,x0,compute_grad,args=myargs)
    (W1,b1,W2,b2,Wlabel) = fold_up(d,label_size,result[0])
    return W1,b1,W2,b2,Wlabel,result[1]
    
    
