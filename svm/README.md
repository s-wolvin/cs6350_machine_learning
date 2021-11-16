# HW 4: Support Vector Machine
Implement the support vector machine learning algoithms for the bank note dataset.
## Stochastic Sub-Gradient Descent for Primal Domain SVM
### primal_svm_partA.py
Implement the stochastic sub-gradient descent for SVM using the schedule of learning rate r = r0 / (1 + (r0/a) * t). <br />
run_primalSVM_partA.sh - shell script which runs primal_svm_partA.py using python3.8

### primal_svm_partB.py
Implement the stochastic sub-gradient descent for SVM using the schedule of learning rate r = r0 / (1 + t). <br />
run_primalSVM_partB.sh - shell script which runs primal_svm_partB.py using python3.8

## Dual SVM Learning Algorithm
### dual_svm.py
Implement the dual SVM with C = {100/873, 500/873, 700/873}. <br />
run_dualSVM.sh - shell script which runs dual_svm.py using python3.8

### dual_svm_gaussKernel.py
Starting with a dual SVM, implement a non-linear SVM using a Gaussian Kernel. Loops through values of C = {100/873, 500/873, 700/873} and Gamma = {0.1, 0.5, 1, 5, 100}.  <br />
run_dualSVM_gaussKernel.sh - shell script which runs dual_svm_gaussKernel.py using python3.8
