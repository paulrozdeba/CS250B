import numpy as np

# import doc,vocab indices
kosdata_full = np.loadtxt('data/docword.kos.txt', dtype='float', skiprows=3)
kosdata_full[:,:2] -= 1  # docs and vocs were originally numbered starting with 1

M = 3430  # number of documents
V = 6906  # size of vocabulary

# select only enough documents st S is approximately 50,000
for d in range(M):
    if d == 0:
        kosdata = kosdata_full[kosdata_full[:,0]==0.0]
        c = np.sum(kosdata[:,2])
    else:
        kosdata = np.vstack((kosdata,kosdata_full[kosdata_full[:,0]==float(d)]))
        c = np.sum(kosdata[:,2])
        if c > 25000:
            break

# now save the data
np.savetxt('data/kosdata.dat', kosdata)
