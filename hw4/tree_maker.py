import numpy as np

def build_tree(sentence,t_label,vocab,W1,W2,b1,b2,Wlabel,normalized):
    """
    This function builds the RAE tree using the greedy algorithm and stores the
    result in arrays.
    
    Inputs:
        (W1,W2,b1,b2,Wlabel) -- the various weights at the current training state
        vocab -- the Vxd matrix of d-dimensional vectors corresponding to
        elements of the vocabulary.
        sentence -- the input sentence as a length N-list of integers corresponding
        to the vocab indices of the words.
        t_label -- the target distribution for this training example
        normalized -- boolean determining whether or not to normalize parent
        representations
    Outputs:
        tree_struct -- a 4x(2N -1) array of ints.  Each column corresponds to a
        node in the tree (starting from the N leaf nodes, and the last column is
        for the output node).  The first row corresponds to left child index,
        the second row corresponds to right child index, the 3rd is parent index
        If any entry = 2N-1 that is to indicate the empty set.  The 4th row is 
        the number of words under that node.
        tree_vectors -- a dx(2N-1) array of floats.  Eah column is the
        d-dimensional  parent representation at that node.
        tree_reconstruct -- a 2dx(2N-1) array of floats.  The first N columns
        are zero, then the next N-1 columns are the reconstruction vectors of
        the corresponding internal node.
        total_rec_error -- sum of weighted reconstruction error at all non-terminal
        nodes
        tree_predict -- the predicted distributions at every non-terminal node
        tree_reconstruct -- the pair of reconstruction vectors at every non
        terminal node
    """
    
    d = np.size(vocab[0,:])
    N = len(sentence)
    num_nodes = 2*N - 1
    label_size = np.size(t_label)
    
    #memory allocation
    tree_struct = np.zeros((4,num_nodes), dtype = 'int') + num_nodes
    tree_vectors = np.zeros((d,num_nodes))
    tree_reconstruct = np.zeros((2*d,num_nodes))
    tree_predict = np.zeros((label_size,num_nodes))
    total_rec_error = 0.0
    
    #initialize search level of tree
    tree_top = list(xrange(N))
    
    #initialize first N tree_vectors, and initial structural information of
    #the leaf nodes
    for i in range(N):
        vocab_index = sentence[i]
        tree_vectors[:,i] = vocab[vocab_index,:]
        tree_struct[3,i] = 1
    
    #initialize error-list of the leaf nodes
    error_list = []
    for i in range(N-1):
        left = i
        right = i+1
        c1 = tree_vectors[:,left]
        c2 = tree_vectors[:,right]
        n1 = tree_struct[3,left]
        n2 = tree_struct[3,right]
        error = recon_error(c1,c2,W1,W2,b1,b2,n1,n2,normalized)
        error_list.append(error)
    
    
    #main loop for building the tree
    for k in range(N,num_nodes):
        #find which pair of consecutive current top level nodes has lowest
        #recon error
        min_loc = np.argmin(error_list)
        total_rec_error += error_list[min_loc]
        #node indices of left and right children
        left = tree_top[min_loc]
        right = tree_top[min_loc + 1]
        #update parent entry for left and right
        tree_struct[2,left] = k
        tree_struct[2,right] = k
        #update tree_struct left and right children, and num_words for node k
        tree_struct[0,k] = left
        tree_struct[1,k] = right
        tree_struct[3,k] = tree_struct[3,left] + tree_struct[3,right]
        #update tree_vector entry for node k
        p = make_p(tree_vectors[:,left],tree_vectors[:,right],W1,b1,normalized)
        tree_vectors[:,k] = p
        #update reconstruction 
        g = np.dot(W2,p) + b2
        tree_reconstruct[:,k] = g
        
        if(k==(num_nodes-1)):
            break
        #update error_list, three cases
        if(min_loc==0):
            #first pair of nodes was selected
            error = recon_error(p,tree_vectors[:,min_loc+2],W1,W2,b1,b2,tree_struct[3,k],tree_struct[3,min_loc+2],normalized)
            error_list[:2] = [error]
        elif(min_loc==(len(tree_top)-2)):
            #last pair of nodes was selected
            error = recon_error(tree_vectors[:,min_loc-1],p,W1,W2,b1,b2,tree_struct[3,min_loc-1],tree_struct[3,k],normalized)
            error_list[-2:] = [error]
        else:
            #interior pair selected, have to replace two elements of the list
            error1 = recon_error(tree_vectors[:,min_loc-1],p,W1,W2,b1,b2,tree_struct[3,min_loc-1],tree_struct[3,k],normalized)
            error2 = recon_error(p,tree_vectors[:,min_loc+2],W1,W2,b1,b2,tree_struct[3,k],tree_struct[3,min_loc+2],normalized)
            error_list[min_loc-1:min_loc+2] = [error1,error2]
        #update tree_top
        tree_top[min_loc:min_loc+2] = [k]
    
    total_pred_error = 0.0
    #now calculate prediction vectors and  prediction errors at all non-terminal
    #nodes
    for j in range(N,num_nodes):
        prediction = make_d(tree_vectors[:,j],Wlabel)
        tree_predict[:,j] = prediction
        total_pred_error += pred_error(prediction,t_label)
    
    return tree_struct,tree_vectors,tree_reconstruct,tree_predict,total_rec_error,total_pred_error

def make_p(c1,c2,W1,b1,normalized):
    """
    Constructs the parent representation of c1 and c2.
    """
    p = np.tanh(np.dot(W1,np.hstack((c1,c2)))+ b1)
    if(normalized==1):
        pnorm = np.linalg.norm(p)
        p /= pnorm
    return p

def make_d(tree_vector,Wlabel):
    """
    calculates the prediction distribution for a given node
    """
    prediction = np.exp(np.dot(Wlabel,tree_vector))
    norm = np.sum(prediction)
    prediction /= norm
    return prediction

def recon_error(c1,c2,W1,W2,b1,b2,n1,n2,normalized):
    """
    calculates the weighted reconstruction error
    """
    d = np.size(c1)
    p = make_p(c1,c2,W1,b1,normalized)
    #print p,W2,b2
    g = np.dot(W2,p) + b2
    
    rec_error = (float(n1)/float(n1+n2))*(np.linalg.norm(c1 - g[:d])**2)
    rec_error += (float(n2)/float(n1+n2))*(np.linalg.norm(c2 - g[d:])**2)
    return rec_error

def pred_error(prediction,t_label):
    """
    calculates the cross entropy error for a given node
    """
    pred_error = 0.0
    pred_error = -np.dot(t_label,np.log(prediction))
    return pred_error

def whole_error(neg_list,pos_list,vocab,W1,b1,W2,b2,Wlabel,normalized):
    """
    calculates the prediction and reconstruction error over the whole corpus
    """
    whole_rec = 0.0
    whole_pred = 0.0
    
    num_neg = len(neg_list)
    num_pos = len(pos_list)
    
    #calculate over the neg_list
    t_label = np.array([0.0,1.0])
    for i in range(num_neg):
        (ignore1,ignore2,ignore3,ignore4,rec_error,pred_error) = build_tree(neg_list[i],t_label,vocab,W1,W2,b1,b2,Wlabel,normalized)
        whole_rec += rec_error
        whole_pred += pred_error
    #calculate over the pos_list
    t_label = np.array([1.0,0.0])
    for i in range(num_pos):
        (ignore1,ignore2,ignore3,ignore4,rec_error,pred_error) = build_tree(pos_list[i],t_label,vocab,W1,W2,b1,b2,Wlabel,normalized)
        whole_rec += rec_error
        whole_pred += pred_error
    
    return whole_rec,whole_pred
