## Savanna Wolvin
# Created: Oct. 20th, 2021
# Edited: Oct. 20th, 2021

# SUMMARY
# We will implement Perceptron for a binary classification task — bank-note 
# authentication. Please download the data “bank-note.zip” from Canvas. The 
# features and labels are listed in the file “bank-note/data-desc.txt”. The 
# training data are stored in the file “bank-note/train.csv”, consisting of 
# 872 examples. The test data are stored in “bank-note/test.csv”, and comprise 
# of 500 examples. In both the training and testing datasets, feature values 
# and labels are separated by commas.
# (a) [16 points] Implement the standard Perceptron. Set the maximum number of
    # epochs T to 10. Report your learned weight vector, and the average 
    # prediction error on the test dataset.
# (b) [16 points] Implement the voted Perceptron. Set the maximum number of 
    # epochs T to 10. Report the list of the distinct weight vectors and their 
    # counts — the number of correctly predicted training examples. Using this 
    # set of weight vectors to predict each test example. Report the average 
    # test error.
# (c) [16 points] Implement the average Perceptron. Set the maximum number of
    # epochs T to 10. Report your learned weight vector. Comparing with the 
    # list of weight vectors from (b), what can you observe? Report the 
    # average prediction error on the test data.
    
# INPUT
# prcptrnAlgorithm  - Choose what perceptron algorithm to use
# maxEpoch          - Choose the number of loops through the dataset
# rate              - Choose the learning rate
# data_file_train   - Name of the training datafile
# data_file_test    - Name of the test datafile
# data_folder       - Name of the folder containing the training and test data

# OUTPUT
# 


#%% Global Imports
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import random as rd




#%% Variable Presets

# Type of Perceptron Algorithm: 'Standard', 'Voted', 'Averaged'
prcptrnAlgorithm = 'Voted'

# maximum loops
maxEpoch = 10

# learning rate
rate = 0.1

# Data set
data_file_train = 'train'
data_file_test = 'test'

data_folder = 'bank-note/'




#%% Load Data
    
trainData = pd.read_csv(data_folder + data_file_train + '.csv', sep=',', header=None)
trainData = trainData.to_numpy()
trainData[np.where(trainData[:,4] == 0), 4] = -1

testData = pd.read_csv(data_folder + data_file_test + '.csv', sep=',', header=None)
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
        rand_idx = rd.sample(range(0, np.shape(trainData)[0]), np.shape(trainData)[0])
        trainDataX = trainData[rand_idx, :]
        
        for ex in range(np.shape(trainDataX)[0]):
            error = trainDataX[ex,4]*np.dot(wVec, trainDataX[ex,0:4])
            
            if error <= 0:
                wVec = wVec + rate * trainDataX[ex,4] * trainDataX[ex,0:4]
                
    for ex in range(np.shape(trainData)[0]):
        prdctnsTrain[ex]    = np.sign(np.dot(wVec, trainData[ex,0:4]))
        
    for ex in range(np.shape(testData)[0]):
        prdctnsTest[ex]     = np.sign(np.dot(wVec, testData[ex,0:4]))
    
    print('Weight Vector:')
    print(wVec)
    
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
    
    c.append(1)
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
        
    print('Counts:')
    print(c)
    print('Weight Vectors:')
    print(wVec)
    
    wd = 4
    ##
    fig, axs = plt.subplots(2,1, figsize=(8,5), sharex=True)
    axs[0].set_title('Weight Vector Values')
    im = axs[0].imshow(wVec, cmap='rainbow', interpolation=None, aspect='auto')
    axs[0].spines['top'].set_linewidth(wd)
    axs[0].spines['bottom'].set_linewidth(wd)
    axs[0].spines['right'].set_linewidth(wd)
    axs[0].spines['left'].set_linewidth(wd)
    axs[0].plot(range(len(c)), [2.5]* len(c), 'k', linewidth=wd)
    axs[0].plot(range(len(c)), [1.5]* len(c), 'k', linewidth=wd)
    axs[0].plot(range(len(c)), [0.5]* len(c), 'k', linewidth=wd)
    axs[0].set_yticks(ticks=[0,1,2,3])
    axs[0].set_yticklabels(['$w_1$','$w_2$','$w_3$','$w_4$'])
    axs[0].set_ylabel('Weighted Vector')
    fig.colorbar(im, ax=[axs[0], axs[1]])
    
    
    axs[1].set_title('Counts')
    axs[1].plot(range(len(c)), c, 'k')
    axs[1].set_xlabel('Count of Distinct Weight Vectors')
    axs[1].set_ylabel('Num. of Correctly Predicted \n Training Examples', fontsize='small')
    plt.show()
    fig.savefig('votedPerceptron.png', dpi=300)
      
    
    
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

    print('Weight Vector:')
    print(wVec[:,m])

    return prdctnsTrain, prdctnsTest





#%%

main()



