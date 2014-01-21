This folder contains all of the code necessary to run the SGD optimization on 
the data provided for hw1.  `sgd.py` contains the core functionality of the 
procedure:

`sgd_train` runs the SGD optimization for the regularized logistic model LCL 
with automatic convergence checking.  `mu_gridsearch` searches over values of 
mu, the regularization scale, for a value that maximizes the LCL when measured 
on the validation set.

`run_sgd.py` is what you want to use to actually load the data and run the 
procedure.  Either use this script or make copies for different experiments you 
want to run.