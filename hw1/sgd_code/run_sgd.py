"""
run_sgd.py

Runs the SGD procedure over the data for CS250B hw1, including automatic 
convergence detection in SGD optimization over the training data and a grid 
search over values of the regularization scale (which optimizes the regularized 
parameters over the validation set).
"""

import numpy as np
import matplotlib.pyplot as plt
import data_routines as dr
import sgd

# load the data and preprocess it
train_data = np.loadtxt('../dataset/1571/train_npcomp.dat', dtype='float')
test_data = np.loadtxt('../dataset/1571/test_npcomp.dat', dtype='float')

# save first half as training data, second half as validation data
D = train_data.shape[1] - 1
N_trainex = train_data.shape[0]
if N_trainex%2 == 0:
    N_trainex = int(N_trainex/2)
else:
    N_trainex = int(N_trainex/2)+1
valid_data = train_data[N_trainex:]
train_data = train_data[:N_trainex]

N_trainex = train_data.shape[0]
N_validex = valid_data.shape[0]
N_testex = test_data.shape[0]

train_data, tdmean, tdstd = dr.preprocess(train_data, full_output=True)
valid_data = dr.preprocess(valid_data, rescale=False)
test_data = dr.preprocess(test_data, rescale=False)

# rescale validation and test sets same as training data
valid_data[:,1:-1] -= np.resize(tdmean, (N_validex,D))
valid_data[:,1:-1] /= tdstd
test_data[:,1:-1] -= np.resize(tdmean, (N_testex,D))
test_data[:,1:-1] /= tdstd

# initiate the grid search over mu
trained_pars, mu_max, lr_max, LCL_max = sgd.mu_gridsearch(train_data, valid_data, N_mutestiter=1)

# calculate error rate on test data set
errors = 0
for ind_ex in range(N_testex):
    if sgd.logistic(trained_pars, test_data[ind_ex,1:]) >= 0.5:
        errors += 1 - test_data[ind_ex,0]
    else:
        errors += test_data[ind_ex,0]
error_rate = errors/N_testex
print 'Error rate on test data = ' + str(error_rate*100) + '%\n'

"""
# saving and plotting arrays
# plot some parameter trajectory
print 'plotting parameter trajectory'
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(pars_traj[:,0])
fig.savefig('testtraj_par.png')

# plot LCL trajectory
print 'calculating LCL time series'
LCL_ts = np.zeros(pars_traj.shape[0]/N_trainex + 1)
ns = 0
for step in pars_traj[::N_trainex]:
    LCL_ts[ns] = LCL(step, train_data)
    ns += 1

print 'plotting LCL time series'
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(LCL_ts)
fig.savefig('testtraj_LCL.png')
plt.close(fig)
"""