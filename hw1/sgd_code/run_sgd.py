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
# shuffle the training set before splitting it
randstate1 = np.random.get_state()
np.random.shuffle(train_data)

if N_trainex%2 == 0:
    N_trainex = int(N_trainex/2)
else:
    N_trainex = int(N_trainex/2)+1
valid_data = train_data[N_trainex:]
train_data = train_data[:N_trainex]

N_trainex = train_data.shape[0]
N_validex = valid_data.shape[0]
N_testex = test_data.shape[0]

# preprocess the data
train_data, tdmean, tdstd = dr.preprocess(train_data, full_output=True)
valid_data = dr.preprocess(valid_data, rescale=False)
test_data = dr.preprocess(test_data, rescale=False)

# rescale validation and test sets same as training data
valid_data[:,1:-1] -= np.resize(tdmean, (N_validex,D))
valid_data[:,1:-1] /= tdstd
test_data[:,1:-1] -= np.resize(tdmean, (N_testex,D))
test_data[:,1:-1] /= tdstd

# initiate the grid search over mu
trained_pars, mu_max, lr_max, LCL_max = sgd.mu_gridsearch(train_data, valid_data)

# calculate error rate on test data set
errors = 0
for ind_ex in range(N_testex):
    if sgd.logistic(trained_pars, test_data[ind_ex,1:]) >= 0.5:
        errors += 1 - test_data[ind_ex,0]
    else:
        errors += test_data[ind_ex,0]
error_rate = errors/N_testex
print 'Error rate on test data = ' + str(error_rate*100) + '%\n'

# save results to text file
fname = 'opt_results.txt'
f = open(fname, 'a')
f.write('rand state: ' + str(randstate1) + '\n')
f.write('\tmu = ' + str(mu_max) + '    lr = ' + str(lr_max) + 
        '    error = ' + str(error_rate) + '\n\n')
f.close()

# generate sample run using mu_max and lr_max, to plot sample parameter 
# trajectory and LCL trajectory
sample_traj = sgd.sgd_train(train_data, lr_max, mu_max, full_output=True)

# plot some parameter trajectory per time step
print 'plotting parameter trajectory over time steps'
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(1,1,1)
ax.plot(sample_traj[:,0])
ax.set_xlim((0,sample_traj.shape[0]))
ax.set_title('Sample parameter trajectory')
ax.set_xlabel('Step number')
ax.set_ylabel(r'$\beta_0$')
fig.savefig('beta0_traj_ts.pdf')

# plot some parameter trajectory per epoch number
print 'plotting parameter trajectory'
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(1,1,1)
ax.plot(sample_traj[::N_trainex,0])
ax.set_xlim((0,sample_traj[::N_trainex].shape[0]))
ax.set_title('Sample parameter trajectory')
ax.set_xlabel('Epoch number')
ax.set_ylabel(r'$\beta_0$')
fig.savefig('beta0_traj_ep.pdf')

# plot some parameter trajectory per time step
print 'plotting parameter trajectory over time steps'
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(1,1,1)
ax.plot(sample_traj[:,47])
ax.set_xlim((0,sample_traj.shape[0]))
ax.set_title('Sample parameter trajectory')
ax.set_xlabel('Step number')
ax.set_ylabel(r'$\beta_{47}$')
fig.savefig('beta47_traj_ts.pdf')

# plot some parameter trajectory per epoch number
print 'plotting parameter trajectory'
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(1,1,1)
ax.plot(sample_traj[::N_trainex,47])
ax.set_xlim((0,sample_traj[::N_trainex].shape[0]))
ax.set_title('Sample parameter trajectory')
ax.set_xlabel('Epoch number')
ax.set_ylabel(r'$\beta_{47}$')
fig.savefig('beta47_traj_ep.pdf')

# plot some parameter trajectory per time step
print 'plotting parameter trajectory over time steps'
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(1,1,1)
ax.plot(sample_traj[:,337])
ax.set_xlim((0,sample_traj.shape[0]))
ax.set_title('Sample parameter trajectory')
ax.set_xlabel('Step number')
ax.set_ylabel(r'$\beta_{337}$')
fig.savefig('beta139_traj_ts.pdf')

# plot some parameter trajectory per epoch number
print 'plotting parameter trajectory'
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(1,1,1)
ax.plot(sample_traj[::N_trainex,337])
ax.set_xlim((0,sample_traj[::N_trainex].shape[0]))
ax.set_title('Sample parameter trajectory')
ax.set_xlabel('Epoch number')
ax.set_ylabel(r'$\beta_{337}$')
fig.savefig('beta139_traj_ep.pdf')

# plot LCL trajectory
print 'calculating LCL time series'
LCL_ts = np.zeros(sample_traj.shape[0]/N_trainex + 1)
ns = 0
for step in sample_traj[::N_trainex]:
    LCL_ts[ns] = sgd.LCL(step, train_data)
    ns += 1

print 'plotting LCL time series'
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(LCL_ts)
ax.set_xlim((0,sample_traj[::N_trainex].shape[0]))
ax.set_title('Sample LCL trajectory')
ax.set_xlabel('Epoch number')
ax.set_ylabel('LCL')
fig.savefig('LCL_traj_ep.pdf')