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
from scipy.optimize import minimize, Bounds, NonlinearConstraint



#%% Variable Presets

C = 100/873

# Data set
data_file_train = 'train'
data_file_test = 'test'

data_folder = 'bank-note/'

weightedVector = np.array([0,0,0,0,0])

# if C == (100/873):
#     lr = 0.00095
# elif C == (500/873):
#     lr = 0.005
# elif C == (700/873):
#     lr = 0.0075



#%% Load Data
    
trainData = pd.read_csv(data_folder + data_file_train + '.csv', sep=',', header=None)
trainData = trainData.to_numpy()
trainData[np.where(trainData[:,4] == 0), 4] = -1

x = trainData[:,range(4)]
y = trainData[:,4]

testData = pd.read_csv(data_folder + data_file_test + '.csv', sep=',', header=None)
testData = testData.to_numpy()
testData[np.where(testData[:,4] == 0), 4] = -1



#%% 

def fun(a, y, x):
    value1 = [a[i]*y[i]*x[i,:] for i in range(np.shape(x)[0])]
    value2 = [a[j]*y[j]*x[j,:] for j in range(np.shape(x)[0])]
    value12 = np.sum([np.dot(value1[ij], value2[ij]) for ij in range(np.shape(x)[0])])
    
    value3 = np.sum(a)
    
    return ((1/2)*np.sum(value12)) - value3

# def fun(a0, y, x):
#     value1 = np.sum(np.array([y[i]*x[i,:] for i in range(np.shape(x)[0])]), axis=0)
#     value2 = np.sum(np.array([y[j]*x[j,:] for j in range(np.shape(x)[0])]), axis=0)
#     value12 = np.dot(value1, value2)
    
#     value3 = np.sum(a0)
    
#     return ((1/2)*np.sum(value12)) - value3



#%%

def constraint(a):
    return np.sum([a[k]*y[k] for k in range(np.shape(a)[0])])



#%% Minimize problem 
    
a0 = np.ones([np.shape(x)[0], 1])*C
print(fun(a0, y, x))

# bnds = (0, C)
bound = Bounds(0, C)

con = {'type':'eq', 'fun': constraint}

# eq_cons = ({'type':'eq', 'fun' : lambda a0: np.sum([a0[k]*y[k] for k in range(np.shape(a0)[0])])})

# nlc = NonlinearConstraint(const, )

# res = minimize(fun, a0, args=(y, x), method='SLSQP', bounds=bound)

res = minimize(fun, a0, args=(y, x), method='SLSQP', bounds=bound, constraints=con)

























