# HW 2: Ensemble Learning
Implemented the boosting and bagging algorithms based on decision
trees for the bank marketing dataset in HW1.

## Adaboost Decision Tree
### adaBoost_decisionTree_numerical.py <br />
Implement adaboost decision trees for the bank dataset. This python script outputs plots 'test_train_error_adaboost_stumps.png' and 'test_train_error_adaboost_all.png' which shows the training and test errors at each stump and at each iteration. <br />

run_adaboost.sh - shell script which runs adaBoost_decisionTree_numerical.py using python3.8

## Bagged Decision Trees
### baggedTrees_decisionTree.py <br />
Implement bagged decision trees for the bank dataset. This python script outputs a figure 'test_train_error.png' which plots the training and test errors with each iteration. In additon, it outputs CSV files which contains all predicitons for the training and test trees. <br />

run_baggedTrees.sh - shell script which runs baggedTrees_decisionTree.py using python3.8

### baggedTrees_vs_singleTree_decisonTree.py <br />
Implement bagged decision trees for the bank dataset by sampling 1000 of the 5000 data vectors and solving for 100 decision trees using the subset of the data. This python script outputs the predicted values at each iteration in a new file for each 1000-sample. To solve for the bias, variance, and general squared error of the training and test datasets, run the python script calc_bias_var_baggedTree.py. <br />

run_sampled_baggedTrees.sh - shell script which runs baggedTrees_vs_singleTree_decisonTree.py using python3.8

#### calc_bias_var_baggedTree.py
Solves for the bias, variance, and general squared error of the training and test datasets from the output from baggedTrees_vs_singleTree_decisonTree.py. <br />

run_calc_bias_var_baggedTree.sh - shell script which runs calc_bias_var_baggedTree.py using python3.8

## Random Forest Decision Trees
### randomForests_bagggedTrees_decisionTree.py <br />
Implement random forests decision trees for the bank dataset using varying subsets of attributes. This python script outputs two CSV files with contains the predictions for the training and test datasets. To change the number of attributes used at each level, change the 'num_subset' variable. <br />

run_forest_baggedTrees.sh - shell script which runs randomForests_bagggedTrees_decisionTree.py using python3.8

#### calc_error_bias_var_forest.py
Plots the training and test errors at each iteration with the output from randomForests_bagggedTrees_decisionTree.py using a subset of {2, 4, 6} of attributes. Solves for the bias, variance, and general squared error of the training and test datasets from the output from baggedTrees_vs_singleTree_decisonTree.py.

run_calc_error_bias_var_forest.sh - shell script which runs calc_error_bias_var_forest.py using python3.8


