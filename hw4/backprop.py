"""
backprop.py

Calculates derivatives of loss function in a neural network, using the back-
propagation algorithm.
"""

import numpy as np

def backprop(tree, treeinfo, t, h, Dh, g, Dg, pars, DER, DEL, alpha=0.2):
    """
    tree - Array of meanings for each node.
    treeinfo - Parameters describing tree structure.
    t - True label.
    h - Function i,j -> k.
    Dh - Derivative of h.
    g - Function meaning -> label.
    Dg - Derivative of g.
    pars - Tuple of parameters in the order W,U,V.
    DER_U,DEL_V - Derivatives of rec. & label error functions wrt U and V.
    DER_W,DEL_W - Derivatives of rec. & label error functions wrt W.
    alpha - Hyperparameter for relative importance of E1, E2.
    """
    
    W,U,V = pars  # extract parameter arrays
    NDM = tree.shape[1]  # number of meaning dimensions
    NDL = V.shape[0]  # number of label dimensions
    Wleft = W[:,:NDM]
    Wright = W[:,NDM:]
    
    # need arrays for derivatives wrt parameters
    DW1 = np.zeros(shape=W.shape)
    DW2 = np.zeros(shape=W.shape)
    DU = np.zeros(shape=U.shape)
    DV = np.zeros(shape=V.shape)
    
    # array for outputs and output delta values
    N = (tree.shape[0] + 1)/2  # length of sentence
    deltaR_out = np.zeros(shape=(2,NDM))  # for reconstruction errors
    deltaL_out = np.zeros(shape=(1,NDL))  # for labeling errors
    
    # loop over parent nodes
    for i,(info,phrase) in enumerate(zip(treeinfo,tree)):
        if info[0] == (2*N-1) and info[1] == (2*N-1):
            continue  # is a leaf
        
        meaning = tree[i]
        lchild = tree[info[0]]
        rchild = tree[info[1]]
        childvec = np.append(lchild,rchild)
        
        # calculate labels and reconstructions
        p = g(meaning,V)  # predicted label
        z = np.einsum('ij,j',U,meaning)  # reconstruction
        
        # calculate deltas for W,U,V
        deltaW1 = np.einsum('i,ij',(z-childvec),U) * Dh(childvec,W)
        deltaW2 = -np.einsum('i,i,ij',(t/p),Dg(meaning,V),V) * Dh(childvec,W)
        deltaU = z - childvec
        deltaV = -(t/p) * Dg(meaning,V)
        
        # add to derivatives over parameters
        DW += alpha * np.outer(deltaW1,childvec) + \
              (1.0-alpha) * np.outer(deltaW2,childvec)
        DU += alpha * np.outer(deltaU,meaning)
        DV += (1.0-alpha) * np.outer(deltaV,meaning)
        
        # propagate to the hidden node
        deltaU_n = Dh(a) * np.einsum('i,ij',deltaU,U)
        deltaV_n = Dh(a) * np.einsum('i,ij',deltaV,V)
        deltaW1_n = Dh(a) * np.einsum('i,ij',deltaW1,WR)
        deltaW2_n = Dh(a) * np.einsum('i,ij',deltaW2,WR)
        
        # now propagate down through the children
        Lidx = info[0]
        Ridx = info[1]
        while True:
            if Lidx == (2*N-1) and Ridx == (2*N-1):
                break  # we've hit a leaf
            a = np.einsum('ij,j',W,childvec)
            deltaU_L
