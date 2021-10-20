## Savanna Wolvin
# Created: Oct. 11th, 2021
# Edited: Oct. 20th, 2021

# SUMMARY
# We have discussed how to calculate the optimal weight vector with an
# analytical form. Please calculate the optimal weight vector in this way. 
# Comparing with the weight vectors learned by batch gradient descent and 
# stochastic gradient descent, what can you conclude? Why?

# INPUT
# data_file - file location that contains the training data to create the 
#                   decision tree

# OUTPUT
# w_matrix      - The analytically solved weight vector
# valueTrain    - Cost Function Value with the Training Data
# valueTest     - Cost Function Value with the Test Data




#%% Global Imports
import pandas as pd 
import numpy as np


#%% Variable Presets

# Data set
data_file_name = 'train'
data_file = 'concrete/' + data_file_name + '.csv'


#%% Load in Variables

print('Load data and attributes...')
trainData = pd.read_csv(data_file, sep=',', header=None)
trainData = trainData.to_numpy()
data_lngth = np.shape(trainData)[1]-1

X = trainData[:,0:7].T
Y = trainData[:,7]


#%% Calculate W

w_matrix = np.dot( np.linalg.inv(np.dot( X, X.T )), np.dot(X, Y ) )

print(w_matrix)




#%% Training Data

data_lngth = np.shape(trainData)[1]-1
    
value = 0

for ex in range(np.shape(trainData)[0]): # each example
    value += ( trainData[ex, data_lngth] - np.dot(w_matrix[:], trainData[ex,range(0,data_lngth)]) )**2

valueTrain = value * (0.5)

print('Training Cost Function Value: ' + str(valueTrain))




#%% Test Data

testData = pd.read_csv('concrete/test.csv', sep=',', header=None)
testData = testData.to_numpy()
data_lngth = np.shape(testData)[1]-1
    
value = 0

for ex in range(np.shape(testData)[0]): # each example
    value += ( testData[ex, data_lngth] - np.dot(w_matrix[:], testData[ex,range(0,data_lngth)]) )**2

valueTest = value * (0.5)

print('Test Cost Function Value: ' + str(valueTest))



