"""
backprop_tests.py

Just some tests for the backprop function.
"""
import numpy as np
from backprop import backprop,backprop_full

def h(x,W):
    return np.tanh(np.einsum('ij,j',W,x))
def g(x,V):
    beta = np.einsum('ij,j',V,x)
    etothea = np.exp(beta)
    return etothea / np.sum(etothea)
def Dh(x,W):
    return 1.0 - h(x,W)**2.0
def Dg(x,V):
    gval = g(x,V)
    return gval * (1.0 - gval)

def runtest():
    # invent dummy tree and stuff
    DM = 5
    DL = 2
    Nw = 42
    Nn = 2*Nw - 1
    """
    # two-word tree
    tree = np.random.randn(Nn,DM).T
    treeinfo = np.array([[Nn, Nn, 2, 1],
                         [Nn, Nn, 2, 1],
                         [0, 1, Nn, 2]]).T
    #treeinfo = np.array([[1, 2, Nn, 2],
    #                     [Nn, Nn, 0, 1],
    #                     [Nn, Nn, 0, 1]]).T
    """
    """
    # three-word tree
    tree = np.random.randn(Nn,DM).T
    treeinfo = np.array([[Nn, Nn, 3, 1],
                         [Nn, Nn, 3, 1],
                         [Nn, Nn, 4, 1],
                         [0, 1, 4, 2],
                         [3, 2, Nn, 3]]).T
    """
    # four-word tree
    """
    tree = np.random.randn(Nn,DM).T
    treeinfo = np.array([[Nn,Nn,4,1],
                         [Nn,Nn,4,1],
                         [Nn,Nn,5,1],
                         [Nn,Nn,5,1],
                         [0,1,6,2],
                         [2,3,6,2],
                         [4,5,Nn,4]]).T
    """
    tree = np.load('meanings.npy')
    treeinfo = np.load('structure.npy')
    print tree.T
    print treeinfo.T
    t = np.random.randn(DL)

    # initialize parameters
    W = np.random.randn(DM,2*DM+1)
    U = np.random.randn(2*DM,DM+1)
    V = np.random.randn(DL,DM)
    pars = [W,U,V]
    
    # run the bastard
    DW,DU,DV,Dx = backprop_full(tree,treeinfo,t,pars,renorm=True)
    print DW,DU,DV,Dx

if __name__ == '__main__':
    runtest()
