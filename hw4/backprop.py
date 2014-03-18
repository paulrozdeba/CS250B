"""
backprop.py

Calculates derivatives of loss function in a neural network, using the back-
propagation algorithm.
"""
import tree_maker as tm
import numpy as np
import time

def full_grad(W1,b1,W2,b2,Wlabel,alpha,neg_list,pos_list,vocab,normalized):
    #set up some functions for array manipulation
    d = np.shape(W1)[0]
    W = np.zeros((d,(2*d)+1))
    W[:,:(2*d)] = W1
    W[:,(2*d)] = b1
    U = np.zeros((2*d,d+1))
    U[:,:d] = W2
    U[:,d] = b2
    k = np.shape(Wlabel)[0]
    V = np.zeros((k,d))
    V[:,:d] = Wlabel
    pars = (W,U,V)
    
    #set up arrays that will hold the derivatives
    dW = np.zeros(np.shape(W))
    dU = np.zeros(np.shape(U))
    dV = np.zeros(np.shape(Wlabel))
    
    #loop over neg_list
    label = np.array([0.0,1.0])
    for example in neg_list:
        tree_stuff = tm.build_tree(example,label,vocab,W1,W2,b1,b2,Wlabel,normalized)
        tree_meanings = tree_stuff[1]
        treeinfo = tree_stuff[0]
        #extra_ones = np.ones(np.shape(tree_meanings)[1])
        #tree = np.vstack((tree_meanings,extra_ones))
        Dlist = backprop(tree_meanings,treeinfo,label,pars,alpha,normalized)
        dW+=Dlist[0]
        dU+=Dlist[1]
        dV+=Dlist[2]
    #loop over pos_list
    label = np.array([1.0,0.0])
    for example in neg_list:
        tree_stuff = tm.build_tree(example,label,vocab,W1,W2,b1,b2,Wlabel,normalized)
        tree_meanings = tree_stuff[1]
        treeinfo = tree_stuff[0]
        #extra_ones = np.ones(np.shape(tree_meanings)[1])
        #tree = np.vstack((tree_meanings,extra_ones))
        Dlist = backprop(tree_meanings,treeinfo,label,pars,alpha,normalized)
        dW+=Dlist[0]
        dU+=Dlist[1]
        dV+=Dlist[2]
    
    #divide by size of training set
    dW /= float(len(neg_list) + len(pos_list))
    dU /= float(len(neg_list) + len(pos_list))
    dV /= float(len(neg_list) + len(pos_list))
    return dW[:,:(2*d)],dW[:,(2*d)],dU[:,:d],dU[:,d],dV

def full_grad_full(W1,b1,W2,b2,Wlabel,alpha,neg_list,pos_list,vocab,normalized):
    #set up some functions for array manipulation
    d = np.shape(W1)[0]
    W = np.zeros((d,(2*d)+1))
    W[:,:(2*d)] = W1
    W[:,(2*d)] = b1
    U = np.zeros((2*d,d+1))
    U[:,:d] = W2
    U[:,d] = b2
    k = np.shape(Wlabel)[0]
    V = np.zeros((k,d))
    V[:,:d] = Wlabel
    pars = (W,U,V)
    
    #set up arrays that will hold the derivatives
    dW = np.zeros(np.shape(W))
    dU = np.zeros(np.shape(U))
    dV = np.zeros(np.shape(Wlabel))
    dVocab = np.zeros(np.shape(vocab))
    
    #loop over neg_list
    label = np.array([0.0,1.0])
    for example in neg_list:
        tree_stuff = tm.build_tree(example,label,vocab,W1,W2,b1,b2,Wlabel,normalized)
        tree_meanings = tree_stuff[1]
        treeinfo = tree_stuff[0]
        #extra_ones = np.ones(np.shape(tree_meanings)[1])
        #tree = np.vstack((tree_meanings,extra_ones))
        Dlist = backprop_full(tree_meanings,treeinfo,label,pars,alpha,normalized)
        dW+=Dlist[0]
        dU+=Dlist[1]
        dV+=Dlist[2]
        dx = Dlist[3]
        for i in range(len(example)):
            voc_index = example[i]
            dVocab[voc_index,:] += dx[i,:]
        
    #loop over pos_list
    label = np.array([1.0,0.0])
    for example in pos_list:
        tree_stuff = tm.build_tree(example,label,vocab,W1,W2,b1,b2,Wlabel,normalized)
        tree_meanings = tree_stuff[1]
        treeinfo = tree_stuff[0]
        #extra_ones = np.ones(np.shape(tree_meanings)[1])
        #tree = np.vstack((tree_meanings,extra_ones))
        Dlist = backprop_full(tree_meanings,treeinfo,label,pars,alpha,normalized)
        dW+=Dlist[0]
        dU+=Dlist[1]
        dV+=Dlist[2]
        dx = Dlist[3]
        for i in range(len(example)):
            voc_index = example[i]
            dVocab[voc_index,:] += dx[i,:]
    
    #divide by size of training set
    dW /= float(len(neg_list) + len(pos_list))
    dU /= float(len(neg_list) + len(pos_list))
    dV /= float(len(neg_list) + len(pos_list))
    dVocab /= float(len(neg_list) + len(pos_list))
    return dW[:,:(2*d)],dW[:,(2*d)],dU[:,:d],dU[:,d],dV,dVocab

def h_norenorm(x,W):
    return np.tanh(np.einsum('ij,j',W,x))

def h_renorm(x,W):
    tanhact = np.tan(np.einsum('ij,j',W,x))
    return tanhact / np.sqrt(np.einsum('i,i',tanhact,tanhact))

def g(x,V):
    expact = np.exp(np.einsum('ij,j',V,x))
    return expact / np.sum(expact)

def Dh_norenorm(x,W):
    return 1.0 - h_norenorm(x,W)**2.0

def Dh_renorm(x,W):
    tanhact = np.tanh(np.einsum('ij,j',W,x))
    tanhact_norm2 = np.einsum('i,i',tanhact,tanhact)
    dtanhact = 1.0 - h_renorm(x,W)
    return (1.0/np.sqrt(tanhact_norm2)) * dtanhact * h_renorm(x,W) * \
        (1.0 - (h_renorm(x,W)/tanhact_norm2))

def Dg(x,V):
    gval = g(x,V)
    return gval * (1.0 - gval)

def backprop(tree, treeinfo, t, pars, alpha=0.2, renorm=False):
    """
    tree - Array of meanings for each node. Has shape=(d,num_nodes) as per
           output of build_tree. It is transposed in this function so that
           we can loop over nodes.
    treeinfo - Parameters describing tree structure. Has shape=(4,num_nodes)
               as per output of build_tree.  It is transposed in this function
               so that we can loop over information about individual nodes.
    t - True label.
    pars - List of parameter arrays in the order W,U,V.
    alpha - Hyperparameter for relative importance of E1, E2. Defaults to
            Socher's value of 0.2.
    renorm - Do you want to use renormalized transfer functions?
    """
    
    if renorm == False:
        DW,DU,DV = backprop_core(tree,treeinfo,t,h_norenorm,Dh_norenorm,g,Dg,
                                 pars,alpha)
    elif renorm == True:
        DW,DU,DV = backprop_core(tree,treeinfo,t,h_renorm,Dh_renorm,g,Dg,
                                 pars,alpha)
    else:
        print('Invalid input. Flag renorm must be either True or False.')
        exit(0)
    
    return DW,DU,DV

def backprop_full(tree, treeinfo, t, pars, alpha=0.2, renorm=False):
    """
    tree - Array of meanings for each node. Has shape=(d,num_nodes) as per
           output of build_tree. It is transposed in this function so that
           we can loop over nodes.
    treeinfo - Parameters describing tree structure. Has shape=(4,num_nodes)
               as per output of build_tree.  It is transposed in this function
               so that we can loop over information about individual nodes.
    t - True label.
    pars - List of parameter arrays in the order W,U,V.
    alpha - Hyperparameter for relative importance of E1, E2. Defaults to
            Socher's value of 0.2.
    renorm - Do you want to use renormalized transfer functions?
    """
    
    if renorm == False:
        DW,DU,DV,Dx = backprop_core_full(tree,treeinfo,t,h_norenorm,Dh_norenorm,
                                         g,Dg,pars,alpha)
    elif renorm == True:
        DW,DU,DV,Dx = backprop_core_full(tree,treeinfo,t,h_renorm,Dh_renorm,
                                         g,Dg,pars,alpha)
    else:
        print('Invalid input. Flag renorm must be either True or False.')
        exit(0)
    
    return DW,DU,DV,Dx

def backprop_core(tree, treeinfo, t, h, Dh, g, Dg, pars, alpha=0.2):
    """
    Core of the backpropagation.  Outputs derivatives of J wrt parameters
    for a single example.  This DOES NOT include derivatives over word
    meanings.  Look to backprop_core_full for this functionality.
    
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
    Wright = W[:,NDM:-1]  # this will include the extra column at the end
    
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
        nleft = float(treeinfo[info[0]][3])
        nright = float(treeinfo[info[1]][3])
        nrnl = float(nright + nleft)
        NRec[:NDM,:NDM] *= np.sqrt(nleft/nrnl)  # N scaling for left child
        NRec[NDM:,NDM:] *= np.sqrt(nright/nrnl)  # N scaling for right child
        
        meaning = tree[i]
        lchild = tree[info[0]]
        rchild = tree[info[1]]
        childvec = np.append(lchild,rchild)
        
        # extended vectors
        meaningX = np.append(meaning,1)
        childvecX = np.append(childvec,1)
        
        # calculate labels and reconstructions
        p = g(meaning,V)  # predicted label
        z = np.einsum('ij,j',U,meaningX)  # reconstruction
        
        # calculate deltas for W,U,V
        Ured = U[:,:-1]
        Wred = W[:,:-1]
        deltaW1 = np.einsum('ik,k,ij',NRec,(z-childvec),Ured) * Dh(childvecX,W)
        deltaW2 = -np.einsum('i,i,ij',(t/p),Dg(meaning,V),V) * Dh(childvecX,W)
        deltaU = np.einsum('ij,j',NRec,(z-childvec))
        deltaV = -(t/p) * Dg(meaning,V)
        
        # add to derivatives over parameters
        DW1 = alpha * np.outer(deltaW1,childvecX)
        DW2 = (1.0 - alpha) * np.outer(deltaW2,childvecX)
        DW += DW1 + DW2
        DU += alpha * np.outer(deltaU,meaningX)
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
                isleft = False
                isright = False
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
                child2vecX = np.append(child2vec,1)
                
                # keep track of new childrens' indices
                newchildren_idx.append(lchild2_idx)
                newchildren_idx.append(rchild2_idx)
                
                # calculate the deltas
                if isleft == True:
                    c_deltaW1 = Dh(child2vecX,W) * np.einsum('i,ij',deltaW1,Wleft)
                    c_deltaW2 = Dh(child2vecX,W) * np.einsum('i,ij',deltaW2,Wleft)
                elif isright == True:
                    c_deltaW1 = Dh(child2vecX,W) * np.einsum('i,ij',deltaW1,Wright)
                    c_deltaW2 = Dh(child2vecX,W) * np.einsum('i,ij',deltaW2,Wright)
                cdeltas.append([c_deltaW1,c_deltaW2])
                cdeltas.append([c_deltaW1,c_deltaW2])
                
                # Now calculate contribution to derivatives
                DW1 = alpha * np.outer(c_deltaW1,child2vecX)
                DW2 = (1.0 - alpha) * np.outer(c_deltaW2,child2vecX)
                DW += DW1 + DW2
            
            # If you've reached this point, you've look at all the children of
            # the previous parent.
            pdeltas = cdeltas
            children_idx = newchildren_idx
    
    return DW,DU,DV

def backprop_core_full(tree, treeinfo, t, h, Dh, g, Dg, pars, alpha=0.2):
    """
    Core of the backpropagation.  Outputs derivatives of J wrt parameters
    for a single example.  This DOES  include derivatives over word
    meanings.
    
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
    
    N = (tree.shape[0] + 1)/2  # length of sentence
    W,U,V = pars  # extract parameter arrays
    NDM = tree.shape[1]  # number of meaning dimensions
    NDL = V.shape[0]  # number of label dimensions
    Wleft = W[:,:NDM]
    Wright = W[:,NDM:-1]
    
    # need arrays for derivatives wrt parameters
    DW = np.zeros(shape=W.shape)
    DU = np.zeros(shape=U.shape)
    DV = np.zeros(shape=V.shape)
    Dx = np.zeros(shape=(N,NDM))
    
    # loop over parent nodes
    for i,(info,phrase) in enumerate(zip(treeinfo,tree)):
        if info[0] == (2*N-1) and info[1] == (2*N-1):
            continue  # is a leaf
        
        # this next array rescales the reconstruction error
        NRec = np.eye(2*NDM,2*NDM)
        nleft = float(treeinfo[info[0]][3])
        nright = float(treeinfo[info[1]][3])
        nrnl = float(nright + nleft)
        NRec[:NDM,:NDM] *= np.sqrt(nleft/nrnl)  # N scaling for left child
        NRec[NDM:,NDM:] *= np.sqrt(nright/nrnl)  # N scaling for right child
        
        meaning = tree[i]
        lchild = tree[info[0]]
        rchild = tree[info[1]]
        childvec = np.append(lchild,rchild)
        
        # extended vectors
        meaningX = np.append(meaning,1)
        childvecX = np.append(childvec,1)
        
        # calculate labels and reconstructions
        p = g(meaning,V)  # predicted label
        z = np.einsum('ij,j',U,meaningX)  # reconstruction
        
        # calculate deltas for W,U,V
        Ured = U[:,:-1]
        Wred = W[:,:-1]
        deltaW1 = np.einsum('ik,k,ij',NRec,(z-childvec),Ured) * Dh(childvecX,W)
        deltaW2 = -np.einsum('i,i,ij',(t/p),Dg(meaning,V),V) * Dh(childvecX,W)
        deltaU = np.einsum('ij,j',NRec,(z-childvec))
        deltaV = -(t/p) * Dg(meaning,V)
        
        # calculate delta for x variation
        # this is for variation of x for the top-level hidden node currently
        # under consideration
        # these vectors aren't used until the later loop, when we propagate
        # down to the leaves
        deltax1 = np.einsum('ij,j',NRec,(z-childvec))
        deltax2 = -(t/p) * Dg(meaning,V)
        # propagate to the hidden node
        deltax1 = np.einsum('i,ij',deltax1,Ured)
        deltax2 = np.einsum('i,ij',deltax2,V)
        
        # add to derivatives over parameters
        DW1 = alpha * np.outer(deltaW1,childvecX)
        DW2 = (1.0 - alpha) * np.outer(deltaW2,childvecX)
        DW += DW1 + DW2
        DU += alpha * np.outer(deltaU,meaningX)
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
        # now parents' deltas
        pdeltas = [[deltaW1,deltaW2,deltax1,deltax2],
                   [deltaW1,deltaW2,deltax1,deltax2]]
        
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
                isleft = False
                isright = False
                if cidx == treeinfo[treeinfo[cidx,2],0]:
                    isleft = True
                if cidx == treeinfo[treeinfo[cidx,2],1]:
                    isright = True
                # are the children leaves?
                leftLeaf = False
                rightLeaf = False
                if treeinfo[treeinfo[cidx,0],0] == (2*N-1):
                    leftLeaf = True
                if treeinfo[treeinfo[cidx,1],0] == (2*N-1):
                    rightLeaf = True
                
                # parent deltas
                deltaW1 = pdeltas[pctr][0]
                deltaW2 = pdeltas[pctr][1]
                deltax1 = pdeltas[pctr][2]
                deltax2 = pdeltas[pctr][3]
                
                # the child's children (children^2, hence the notation)
                lchild2_idx = treeinfo[cidx,0]
                rchild2_idx = treeinfo[cidx,1]
                lchild2 = tree[lchild2_idx]
                rchild2 = tree[rchild2_idx]
                child2vec = np.append(lchild2,rchild2)
                child2vecX = np.append(child2vec,1)
                
                # keep track of new childrens' indices
                newchildren_idx.append(lchild2_idx)
                newchildren_idx.append(rchild2_idx)
                
                # calculate the deltas
                if isleft == True:
                    c_deltaW1 = Dh(child2vecX,W) * np.einsum('i,ij',deltaW1,Wleft)
                    c_deltaW2 = Dh(child2vecX,W) * np.einsum('i,ij',deltaW2,Wleft)
                    c_deltax1 = Dh(child2vecX,W) * np.einsum('i,ij',deltax1,Wleft)
                    c_deltax2 = Dh(child2vecX,W) * np.einsum('i,ij',deltax2,Wleft)
                elif isright == True:
                    c_deltaW1 = Dh(child2vecX,W) * np.einsum('i,ij',deltaW1,Wright)
                    c_deltaW2 = Dh(child2vecX,W) * np.einsum('i,ij',deltaW2,Wright)
                    c_deltax1 = Dh(child2vecX,W) * np.einsum('i,ij',deltax1,Wright)
                    c_deltax2 = Dh(child2vecX,W) * np.einsum('i,ij',deltax2,Wright)
                cdeltas.append([c_deltaW1,c_deltaW2,deltax1,deltax2])
                cdeltas.append([c_deltaW1,c_deltaW2,deltax1,deltax2])
                
                # Now calculate contribution to derivatives
                DW1 = alpha * np.outer(c_deltaW1,child2vecX)
                DW2 = (1.0 - alpha) * np.outer(c_deltaW2,child2vecX)
                DW += DW1 + DW2
                if leftLeaf == True:
                    Dx1 = alpha * np.einsum('ij,j',Wleft,c_deltax1)
                    Dx2 = (1.0-alpha) * \
                          np.einsum('ij,j',Wleft,c_deltax2)
                    Dx[lchild2_idx] += Dx1 + Dx2
                if rightLeaf == True:
                    Dx1 = alpha * np.einsum('ij,j',Wright,c_deltax1)
                    Dx2 = (1.0-alpha) * \
                          np.einsum('ij,j',Wright,c_deltax2)
                    Dx[rchild2_idx] += Dx1 + Dx2
            
            # If you've reached this point, you've look at all the children of
            # the previous parent.
            pdeltas = cdeltas
            children_idx = newchildren_idx
            
    return DW,DU,DV,Dx
