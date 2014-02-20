"""
matlab_convert.py

Converts matlab data into numpy array, then saves them as npy archives.
Specifically, the Classic400 data is a sparse matrix that has been 
saved in a matlab data file.  This is to be extracted, and stored 
in a "sparse" format but in a numpy array.
"""

import numpy as np
import scipy.io

# first load the dict
mat_file = 'data/classic400.mat'
datadict = scipy.io.loadmat(mat_file)

# load the data in dense format
dense_data = datadict['classic400'].todense().tolist()

# iterate over the dense data, and dump nonzero entries into a new array
sparse_data = []
V = 6205  # size of vocabulary
M = 400   # number of documents

for i,doc in enumerate(dense_data):
    for j,count in enumerate(doc):
        if count != 0:
            sparse_data.append([i,j,count])

# now save in an npy archive
np.save('data/classic400.npy',np.array(sparse_data,dtype='int'))
