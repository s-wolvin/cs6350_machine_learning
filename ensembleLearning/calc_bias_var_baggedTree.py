## Savanna Wolvin
# Created: Oct. 17th, 2021


# SUMMARY
# •Now you have 100 bagged predictors in hand. For comparison, pick the first
# tree in each run to get 100 fully expanded trees (i.e. single trees).
# •For each of the test example, compute the predictions of the 100 single 
# trees.Take the average, subtract the ground-truth label, and take square to 
# compute the bias term (see the lecture slides). Use all the predictions to 
# compute the sample variance as the approximation to the variance term (if 
# you forget what the sample variance is, check it out here). You now obtain 
# the bias and variance terms of a single tree learner for one test example. 
# You will need to compute them for all the test examples and then take 
# average as your final estimate of the bias and variance terms for the single 
# decision tree learner. You can add the two terms to obtain the estimate of 
# the general squared error (that is, expected error w.r.t test examples). Now 
# use your 100 bagged predictors to do the same thing and estimate the general  
# bias and variance terms, as well as the general squared error. Comparing the 
# results of the single tree learner and the bagged trees, what can you 
# conclude? What causes the difference?


#%% Global Imports
import pandas as pd 
import numpy as np




#%% Presets

# Data set
data_file_name = 'test'
data_file = 'bank-1/' + data_file_name + '.csv'

# number of iterations
T = 100

# Number of sampled datasets
S = 11


#%% Load varibles

print('Load data and attributes...')
trainData = pd.read_csv(data_file, sep=',', header=None)
trainData = trainData.to_numpy()

trainPrd = trainData[:, np.shape(trainData)[1]-1]
trainPrd[trainPrd == 'no'] = 0
trainPrd[trainPrd == 'yes'] = 1

single_trees = np.zeros([np.shape(trainData)[0], S], dtype=object)

# load in data
for sx in range(S):
    btData = pd.read_csv('bank_test_baggedTrees_outcomes_' + str(sx) + '_1000samples.csv', sep=',', header=0, index_col=0)
    btData = btData.to_numpy()
    
    single_trees[:,sx] = btData[:,0]
    
single_trees[single_trees == 'no'] = 0
single_trees[single_trees == 'yes'] = 1


#%% Bias and variance of the single trees

# find most predicted single tree
st_prediction = np.zeros([np.shape(single_trees)[0],1], dtype=float)
for idx in range(np.shape(single_trees)[0]):
    st_prediction[idx,0] = np.mean(single_trees[idx,:])
    
# bias term
bias_st = 0
for idx in range(np.shape(st_prediction)[0]):
    bias_st += ( (trainPrd[idx] - st_prediction[idx,0])**2 )
        
bias_st = (bias_st /  np.shape(st_prediction)[0])

# variance term
avg_prd = np.mean(single_trees, axis=1)
var_st = 0
for idx in range(np.shape(single_trees)[0]):
   var_st += (1/(S-1)) * np.sum((single_trees[idx,:] - avg_prd[idx]) **2)
    
var_st = var_st / np.shape(single_trees)[0]

gnrlSqrdError_st = var_st + bias_st




#%% load all bagged trees

# load in data
bagged_trees = np.zeros([np.shape(trainData)[0], S], dtype=object)
for sx in range(S):
    btData = pd.read_csv('bank_test_baggedTrees_outcomes_' + str(sx) + '_1000samples.csv', sep=',', header=0, index_col=0)
    btData = btData.to_numpy()
    
    btData[btData == 'no'] = 0
    btData[btData == 'yes'] = 1
    
    bagged_trees[:,sx] = np.mean(btData, axis=1)




#%% Bias and variance of the bagged trees

# find most predicted single tree
bt_prediction = np.zeros([np.shape(bagged_trees)[0],1], dtype=float)
for idx in range(np.shape(bagged_trees)[0]):
    bt_prediction[idx,0] = np.mean(bagged_trees[idx,:])
    
# bias term
bias_bt = 0
for idx in range(np.shape(bt_prediction)[0]):
    bias_bt += ( (trainPrd[idx] - bt_prediction[idx,0])**2 )
        
bias_bt = (bias_bt /  np.shape(bt_prediction)[0])

# variance term
avg_prd = np.mean(bagged_trees, axis=1)
var_bt = 0
for idx in range(np.shape(bagged_trees)[0]):
   var_bt += (1/(S-1)) * np.sum((bagged_trees[idx,:] - avg_prd[idx]) **2)
    
var_bt = var_bt / np.shape(bagged_trees)[0]

gnrlSqrdError_bt = var_bt + bias_bt

























