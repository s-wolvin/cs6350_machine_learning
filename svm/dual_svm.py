## Savanna Wolvin
# Created: Nov. 8th, 2021
# Edited: 
    
# SUMMARY
# Now let us implement SVM in the dual domain. We use the same dataset, 
# “bank-note.zip”. You can utilize existing constrained optimization libraries.
# For Python, we recommend using “scipy.optimize.minimize”, and you can learn 
# how to use this API from the document at 
# https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.minimize.html. 
# We recommend using SLSQP to incorporate the equality constraints.

# INPUT 


# OUTPUT



#%% Global Imports
import pandas as pd 
import numpy as np
import scipy.optimize as spo
from datetime import datetime




#%% Variable Presets

C = [100, 500, 700]

# Data set
data_file_train = 'train'
data_file_test = 'test'

data_folder = 'bank-note/'




#%% Load Data

print('Load Data...')

trainData = pd.read_csv(data_folder + data_file_train + '.csv', sep=',', header=None)
trainData = trainData.to_numpy()
trainData[np.where(trainData[:,4] == 0), 4] = -1

x = trainData[:,range(4)]
y = trainData[:,4]

testData = pd.read_csv(data_folder + data_file_test + '.csv', sep=',', header=None)
testData = testData.to_numpy()
testData[np.where(testData[:,4] == 0), 4] = -1




#%% Functions

def objective_func(a):
    summation1 = np.sum(np.array([(a[i]*y[i]*x[i,:]) for i in range(np.shape(a)[0])]), axis=0)
    summation2 = np.sum(np.array([(a[j]*y[j]*x[j,:]) for j in range(np.shape(a)[0])]), axis=0)
    component1 = np.dot(summation1, summation2)
    component2 = np.sum(a)
    
    return component1 - component2


def equality_constraint(a):
    iteration = [(a[i]*y[i]) for i in range(np.shape(a)[0])]
    
    return np.sum(iteration)

constraint1 = {'type': 'eq', 'fun': equality_constraint}




#%% Calculate Minimization, Weight Vector, and Bias

for Cx in C:
    # Bounds
    bnds = [(0, (Cx/873))] * np.shape(trainData)[0]
    
    
    # Calculate Minimization
    print('Calculate Alpha Values for C = ' + str(Cx) + '/' + str(873) + '...')
    
    a0 = [0] * np.shape(x)[0]
    
    start_time = datetime.now()
    result = spo.minimize(objective_func, a0, method='SLSQP', bounds=bnds, constraints=[constraint1])
    end_time = datetime.now()
    
    print(result.message + ': ' + 'Duration: ' + str(end_time - start_time))
    
    
    # Calculate Weighted Vector and Bias
    a = result.x
    
    weightedVector = np.array([(a[i]*y[i]*x[i,:]) for i in range(np.shape(x)[0])])
    weightedVector = np.sum(weightedVector, axis=0)
    print('Weight Vector: ' + str(weightedVector))
    
    bias = [(y[j] - np.dot(weightedVector, x[j,:])) for j in range(np.shape(x)[0])]
    bias = np.mean(bias)
    print('Bias: ' + str(bias))
    print('')



