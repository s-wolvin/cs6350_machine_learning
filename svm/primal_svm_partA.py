## Savanna Wolvin
# Created: Nov. 7th, 2021
# Edited: Oct. 20th, 2021

# SUMMARY
# We will first implement SVM in the primal domain with stochastic 
# sub-gradient descent. We will reuse the dataset for Perceptron 
# implementation, namely, “bank-note.zip” in Canvas. The features and labels 
# are listed in the file “classification/data-desc.txt”. The training data are 
# stored in the file “classification/train.csv”, consisting of 872 examples. 
# The test data are stored in “classification/test.csv”, and comprise of 500 
# examples. In both the training and test datasets, feature values and labels 
# are separated by commas. Set the maximum epochs T to 100. Don’t forget to 
# shuffle the training examples at the start of each epoch. Use the curve of 
# the objective function (along with the number of updates) to diagnosis the 
# convergence. Try the hyperparameter C from {100/873, 500/873, 700/873}. 
# Don’t forget to convert the labels to be in {1,−1}.

# INPUT

# OUTPUT


#%% Global Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd




#%% Variable Presets

# hyperparameter
C = 700/873
C_name = '700-873'

# max epochs
maxEpoch = 100

# Data set
data_file_train = 'train'
data_file_test = 'test'

data_folder = 'bank-note/'

weightedVector = np.array([0,0,0,0,0])

if C == (100/873):
    lr = 0.00095
    a = 0.000082   
elif C == (500/873):
    lr = 0.005
    a = 0.00007  
elif C == (700/873):
    lr = 0.0075
    a = 0.0001



#%% Load Data
    
trainData = pd.read_csv(data_folder + data_file_train + '.csv', sep=',', header=None)
trainData = trainData.to_numpy()
trainData[np.where(trainData[:,4] == 0), 4] = -1

testData = pd.read_csv(data_folder + data_file_test + '.csv', sep=',', header=None)
testData = testData.to_numpy()
testData[np.where(testData[:,4] == 0), 4] = -1




#%% determine 

objectFuncValues = []
updates = 0

for epochx in range(maxEpoch):
    # calculate learning rate
    lrt = (lr) / (1 + (lr/a) * epochx)    
    
    # shuffle dataset
    N = np.shape(trainData)[0]
    rand_idx = rd.sample(range(0, N), N)
    trainDataX = trainData[rand_idx, :]
    
    for ex in range(N):
        error = trainDataX[ex, 4] * np.dot(weightedVector, np.append(trainDataX[ex, range(4)],1))
        
        if error <= 1:
            weightedVector = weightedVector - (lrt*weightedVector) + (lrt * C * N * trainDataX[ex, 4] * np.append(trainDataX[ex, range(4)],1))
            updates += 1
        else:
            weightedVector = (1 - lrt) * weightedVector
    
    empiricalLoss = [max([0, 1-trainData[ex,4]*(np.dot(weightedVector, np.append(trainData[ex, range(4)],1)))]) for ex in range(N)]
    regularizationTerm = (1/2) * np.dot(weightedVector.T, weightedVector)
    objectFuncValueX = regularizationTerm + C * np.sum(empiricalLoss)
    objectFuncValues.append(objectFuncValueX)
print(objectFuncValues[np.shape(objectFuncValues)[0]-1])



#%% Plot Object Function Values

fig = plt.figure()
plt.plot(range(maxEpoch), objectFuncValues)
plt.xlabel('Epoch')
plt.ylabel('Object Function Value')
plt.title('Object Function Value VS Epoch for C = ' + str(round(C, 4)) + ', \n a = ' + 
          str(a) + ', and \u03B3\u2080 = ' + str(lr) + ', with ' + str(updates) + ' updates')
    
figName = 'svm_partA_C_' + C_name + '_a_' + str(a*100000) + '_gamma_' + str(round(lr*100000)) + '.png'
# fig.savefig(figName, dpi=300)


#%% training and test error

train_error = 0

for ex in range(np.shape(trainData)[0]):
    sgn = np.sign(np.dot(weightedVector,np.append(trainData[ex, range(4)],1)))
    if sgn != np.sign(trainData[ex,4]):
        train_error += 1

test_error = 0

for ex in range(np.shape(testData)[0]):
    sgn = np.sign(np.dot(weightedVector,np.append(testData[ex, range(4)],1)))
    if sgn != np.sign(testData[ex,4]):
        test_error += 1

train_error = train_error / np.shape(trainData)[0]
test_error = test_error / np.shape(testData)[0]

print('Train Error: ' + str(train_error))
print('Test Error: ' + str(test_error))

print(weightedVector)









