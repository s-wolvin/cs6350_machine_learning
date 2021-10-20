## Savanna Wolvin
# Created: Oct. 20th, 2021
# Edited: Oct. 20th, 2021

# SUMMARY

# INPUT

# OUTPUT



#%% Global Imports
import numpy as np
import pandas as pd
import sys




#%% Variable Presets

# Type of Perceptron Algorithm: 'Standard', 'Voted', 'Averaged'
prcptrnAlgorithm = 'Averaged'

# maximum loops
maxEpoch = 10

# learning rate
rate = 0.1

# Data set
data_file_name1 = 'train'
data_file_name2 = 'test'

data_folder = 'bank-note/'




#%% Load Data
    
trainData = pd.read_csv(data_folder + data_file_name1 + '.csv', sep=',', header=None)
trainData = trainData.to_numpy()
trainData[np.where(trainData[:,4] == 0), 4] = -1

testData = pd.read_csv(data_folder + data_file_name2 + '.csv', sep=',', header=None)
testData = testData.to_numpy()
testData[np.where(testData[:,4] == 0), 4] = -1



   
#%% Main

def main():
    ### Pick Algorithm
    if prcptrnAlgorithm == 'Standard':
        prdctnsTrain, prdctnsTest   = standardPrcptrn()
               
    elif prcptrnAlgorithm == 'Voted':    
        prdctnsTrain, prdctnsTest, wVec, c  = votedPrcptrn()
    
    elif prcptrnAlgorithm == 'Averaged':    
        prdctnsTrain, prdctnsTest   = averagedPrcptrn()
    else:
        sys.error('Incorrect Perceptron Algorithm')
        
    ### calculate error
    prdError = np.zeros([np.shape(testData)[0]])
    prdError = np.count_nonzero((prdctnsTest - testData[:,4]) != 0) / np.shape(testData)[0]
    print('Average Prediction Error, Test Data: ' + str(prdError))    
    
    
    
    
#%%

def standardPrcptrn():
    # Variables
    wVec            = np.zeros([np.shape(trainData)[1]-1])
    prdctnsTrain    = np.zeros([np.shape(trainData)[0]])
    prdctnsTest     = np.zeros([np.shape(testData)[0]])
    
    for epochx in range(maxEpoch):
        for ex in range(np.shape(trainData)[0]):
            error = trainData[ex,4]*np.dot(wVec, trainData[ex,0:4])
            
            if error <= 0:
                wVec = wVec + rate * trainData[ex,4] * trainData[ex,0:4]
                
    for ex in range(np.shape(trainData)[0]):
        prdctnsTrain[ex]    = np.sign(np.dot(wVec, trainData[ex,0:4]))
        
    for ex in range(np.shape(testData)[0]):
        prdctnsTest[ex]     = np.sign(np.dot(wVec, testData[ex,0:4]))
    
    return prdctnsTrain, prdctnsTest    
    



#%%

def votedPrcptrn():
    # Variables
    wVec = np.zeros([np.shape(trainData)[1]-1, 1])
    m = 0
    c = []
    prdctnsTrain    = np.zeros([np.shape(trainData)[0]])
    prdctnsTest     = np.zeros([np.shape(testData)[0]])
    
    for epochx in range(maxEpoch):
        for ex in range(np.shape(trainData)[0]):
            error = trainData[ex,4]*np.dot(wVec[:,m], trainData[ex,0:4])
            
            if error <= 0:
                wVec = np.append(wVec, np.array(wVec[:,m] + rate * trainData[ex,4] * trainData[ex,0:4]).reshape(4,1), axis=1)
                m = m + 1
                c.append(1)
                
            else:
                c[m-1] += 1
    
    for ex in range(np.shape(trainData)[0]):
        summation = 0
        for wx in range(len(c)):
            summation += np.sign(np.dot(wVec[:,wx], trainData[ex,0:4])) * c[wx]
        prdctnsTrain[ex]    = np.sign(summation)
        
    for ex in range(np.shape(testData)[0]):
        summation = 0
        for wx in range(len(c)):
            summation += np.sign(np.dot(wVec[:,wx], testData[ex,0:4])) * c[wx]
        prdctnsTest[ex]    = np.sign(summation)

    return prdctnsTrain, prdctnsTest, wVec, c




#%%

def averagedPrcptrn():
    # Variables
    wVec = np.zeros([np.shape(trainData)[1]-1, 1])
    a = np.zeros([4])
    m = 0
    prdctnsTrain    = np.zeros([np.shape(trainData)[0]])
    prdctnsTest     = np.zeros([np.shape(testData)[0]])
    
    for epochx in range(maxEpoch):
        for ex in range(np.shape(trainData)[0]):
            error = trainData[ex,4]*np.dot(wVec[:,m], trainData[ex,0:4])
            
            if error <= 0:
                wVec = np.append(wVec, np.array(wVec[:,m] + rate * trainData[ex,4] * trainData[ex,0:4]).reshape(4,1), axis=1)
                a = a[:] + wVec[:,m+1]
                m = m + 1
                
    for ex in range(np.shape(trainData)[0]):
        prdctnsTrain[ex]    = np.sign(np.dot(a, trainData[ex,0:4]))
        
    for ex in range(np.shape(testData)[0]):
        prdctnsTest[ex]     = np.sign(np.dot(a, testData[ex,0:4]))

    return prdctnsTrain, prdctnsTest





#%%

main()



