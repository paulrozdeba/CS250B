"""
backprop.py

Calculates derivatives of loss function in a neural network, using the back-
propagation algorithm.
"""

import numpy as np
import time

def backprop(tree, treeinfo, t, h, Dh, g, Dg, pars, alpha=0.2):
    """
    tree - Array of meanings for each node. Has shape=(d,num_nodes) as per
           output of build_tree. It is transposed in this function so that
           we can loop over nodes.
    treeinfo - Parameters describing tree structure. Has shape=(4,num_nodes)
               as per output of build_tree.  It is transposed in this function
               so that we can loop over information about individual nodes.
    t - True label.
    h - Function i,j -> k. Should take (childvec,W) as inputs.
    Dh - Derivative of h. Should take (childvec,W) as inputs.
    g - Function meaning -> label. Should take (meaning,V) as inputs.
    Dg - Derivative of g. Should take (meaning,V) as inputs.
    pars - List of parameter arrays in the order W,U,V.
    alpha - Hyperparameter for relative importance of E1, E2. Defaults to
            Socher's value of 0.2.
    """

    # transpose tree and treeinfo
    tree = tree.T
    treeinfo = treeinfo.T
    
    W,U,V = pars  # extract parameter arrays
    NDM = tree.shape[1]  # number of meaning dimensions
    NDL = V.shape[0]  # number of label dimensions
    Wleft = W[:,:NDM]
    Wright = W[:,NDM:]
    
    # need arrays for derivatives wrt parameters
    DW = np.zeros(shape=W.shape)
    DU = np.zeros(shape=U.shape)
    DV = np.zeros(shape=V.shape)
    
    # array for outputs and output delta values
    N = (tree.shape[0] + 1)/2  # length of sentence
    
    # loop over parent nodes
    for i,(info,phrase) in enumerate(zip(treeinfo,tree)):
        if info[0] == (2*N-1) and info[1] == (2*N-1):
            continue  # is a leaf
        
        # this next array rescales the reconstruction error
        NRec = np.eye(2*NDM,2*NDM)
        NRec[:NDM,:NDM] *= treeinfo[info[0]][3]  # n leaves under left child
        NRec[NDM:,NDM:] *= treeinfo[info[1]][3]  # n leaves under right child
        
        meaning = tree[i]
        lchild = tree[info[0]]
        rchild = tree[info[1]]
        childvec = np.append(lchild,rchild)
        
        # calculate labels and reconstructions
        p = g(meaning,V)  # predicted label
        z = np.einsum('ij,j',U,meaning)  # reconstruction
        
        # calculate deltas for W,U,V
        deltaW1 = np.einsum('ik,k,ij',NRec,(z-childvec),U) * Dh(childvec,W)
        deltaW2 = -np.einsum('i,i,ij',(t/p),Dg(meaning,V),V) * Dh(childvec,W)
        deltaU = np.einsum('ij,j',NRec,(z-childvec))
        deltaV = -(t/p) * Dg(meaning,V)
        
        # add to derivatives over parameters
        DW1 = alpha * np.outer(deltaW1,childvec)
        DW2 = (1.0 - alpha) * np.outer(deltaW2,childvec)
        DW += DW1 + DW2
        DU += alpha * np.outer(deltaU,meaning)
        DV += (1.0-alpha) * np.outer(deltaV,meaning)
        
        # Previously, I believed the step below to be necessary. However, I
        # think the deltaW's calculated above are ALREADY for the hidden node.
        """
        # propagate W deltas to the hidden node
        #a_hn = np.einsum('ij,j',W,childvec)
        print deltaW1.shape,U.shape
        print deltaW2.shape,V.shape
        deltaW1_hn = Dh(childvec,W) * np.einsum('i,ji',deltaW1,U)
        deltaW2_hn = Dh(childvec,W) * np.einsum('i,ji',deltaW2,V)
        print deltaW1_hn.shape
        print deltaW2_hn.shape
        exit(0)
        """
        
        # now propagate down through the children
        # children are ordered from left to right
        children_idx = [info[0],info[1]]
        pdeltas = [[deltaW1,deltaW2],[deltaW1,deltaW2]]  # parents' deltas
        
        leafcounter = 0  # count leaves, for termination of loop
        
        while True:
            # First, test to see if we've looked at all possible (hidden)
            # children.  If so, break and move on to the next parent.
            if leafcounter == info[3]:
                break
            
            # Store deltas of all children. At the end of a loop iteration,
            # the children become parents (*tear*).
            cdeltas = []  # array to hold deltas of children
            newchildren_idx = []  # store indices of new children
            
            for pctr,cidx in enumerate(children_idx):
                # have we hit a leaf?
                if treeinfo[cidx,0]==(2*N-1):
                    leafcounter += 1
                    continue
                # OK, now is it a left or right child?
                if cidx == treeinfo[treeinfo[cidx,2],0]:
                    isleft = True
                if cidx == treeinfo[treeinfo[cidx,2],1]:
                    isright = True
                
                # parent deltas
                deltaW1 = pdeltas[pctr][0]
                deltaW2 = pdeltas[pctr][1]
                
                # the child's children (children^2, hence the notation)
                lchild2_idx = treeinfo[cidx,0]
                rchild2_idx = treeinfo[cidx,1]
                lchild2 = tree[lchild2_idx]
                rchild2 = tree[rchild2_idx]
                child2vec = np.append(lchild2,rchild2)
                
                # keep track of new childrens' indices
                newchildren_idx.append(lchild2_idx)
                newchildren_idx.append(rchild2_idx)
                
                # calculate the deltas
                if isleft == True:
                    c_deltaW1 = Dh(child2vec,W) * np.einsum('i,ij',deltaW1,Wleft)
                    c_deltaW2 = Dh(child2vec,W) * np.einsum('i,ij',deltaW2,Wleft)
                elif isright == True:
                    c_deltaW1 = Dh(child2vec,W) * np.einsum('i,ij',deltaW1,Wright)
                    c_deltaW2 = Dh(child2vec,W) * np.einsum('i,ij',deltaW2,Wright)
                cdeltas.append([c_deltaW1,c_deltaW2])
                cdeltas.append([c_deltaW1,c_deltaW2])
                
                # Now calculate contribution to derivatives
                DW1 = alpha * np.outer(c_deltaW1,child2vec)
                DW2 = (1.0 - alpha) * np.outer(c_deltaW2,child2vec)
                DW += DW1 + DW2
            
            # If you've reached this point, you've look at all the children of
            # the previous parent.
            pdeltas = cdeltas
            children_idx = newchildren_idx
    
    return DW,DU,DV
