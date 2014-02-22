"""
dataproc.py

Load & process matlab data.
"""

import scipy.io

def loadsparse(fname):
    """
    Loads sparse matrix form matlab data.
    """
    
    datadict = scipy.io.loadmat(fname)
