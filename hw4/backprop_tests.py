"""
backprop_tests.py

Just some tests for the backprop function.
"""
import numpy as np
from backprop import backprop

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
    DM = 20
    DL = 2
    Nw = 2
    Nn = 2*Nw - 1
    
    # start with a two-word tree
    tree = np.random.randn(Nn,DM).T
    treeinfo = np.array([[1, 2, Nn, 2],
                         [Nn, Nn, 0, 1],
                         [Nn, Nn, 0, 1]]).T
    t = np.random.randn(DL)

    # initialize parameters
    W = np.random.randn(DM,2*DM)
    U = np.random.randn(2*DM,DM)
    V = np.random.randn(DL,DM)
    pars = [W,U,V]
    
    # run the bastard
    backprop(tree,treeinfo,t,h,Dh,g,Dg,pars)

if __name__ == '__main__':
    runtest()
