## Savanna Wolvin
# Created: Sep. 10th, 2021
# Edited: Sep. 10th, 2021

# SUMMARY
# We will implement a decision tree learning algorithm for car evaluation task.
# The dataset is from UCI repository(https://archive.ics.uci.edu/ml/datasets/
# car+evaluation). Please download the processed dataset (car.zip) from Canvas. 
# In this task, we have 6 car attributes, and the label is the evaluation of 
# the car. The attribute and label values are listed in the file 
# “data-desc.txt”. All the attributes are categorical. The training data are 
# stored in the file “train.csv”, consisting of 1,000 examples. The test data 
# are stored in “test.csv”, and comprise 728 examples. In both training and 
# test datasets, attribute values are separated by commas; the file 
# “datadesc.txt” lists the attribute names in each column.

# INPUT
# maxTreeDepth  - maximum number of levels on the decision tree
# algorithmType - choose between three algorithm types to create the decision 
#                   tree
# data_file     - file location that contains the training data to create the 
#                   decision tree
# labels        - list of column labels used by the data_file

# OUTPUT
# Boolean Function???


#%% Global Imports
import pandas as pd 
import numpy as np
import sys


#%% Variable Presets

# 1 through 6
maxTreeDepth = 2

# 'Entropy', 'GiniIndex', 'MajorityError'
algorithmType = 'Entropy'

# Data set
data_file = 'car/train.csv'

# column labels
labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']


#%% Main Function

def main():
    ### Load Data
    print('Load data and attributes...')
    
    trainData = pd.read_csv(data_file, sep=',', header=None)
    trainData = trainData.to_numpy()
    
    attr_dict = {}
    for idx in np.arange(0, np.shape(trainData)[1]):
        attr_dict.update({labels[idx]: np.unique(trainData[:,idx]).tolist()})


    ### Begin Loop to Create Decision Tree
    print('Begin decision tree loop...')
    avail_attributes = np.arange(0, len(labels)-1)
    level = 0
    while level < maxTreeDepth:
        ### Determine Head Node
        headNode = pickAttribute(trainData, avail_attributes)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # ### Calculate Total Entropy/GiniIndex/MajorityError
        # label_attr, count = np.unique(trainData[:,len_labels], return_counts=1)
        # total = calcInformationGain(count, sum(count))
        # dataLen = np.shape(trainData)[0]
        
        # ### Calculate Entropy/GiniIndex/MajorityError for Each Label
        # attr_informationGain = np.zeros((len_labels,1))
        # for attrx in np.arange(0, len_labels):   # loop through each attribute
        #     print(labels[attrx])
        #     attrSubset, attrCount = np.unique(trainData[:,attrx], return_counts=1)
        #     attrSubset_informationLoss = np.zeros((len(attrSubset), 1))
            
        #     for attrSubsetx in np.arange(0, len(attrSubset)):          # loop through each sub attribute
        #         # print(attrSubset[attrSubsetx])
        #         subsetIdx = [i for i in range(dataLen) if trainData[i, attrx] == attrSubset[attrSubsetx]]
        #         labelSubset, labelCount = np.unique(trainData[subsetIdx, len_labels], return_counts=1)
                
        #         attrSubset_informationLoss[attrSubsetx] = calcInformationGain(labelCount, attrCount[attrSubsetx]) * (attrCount[attrSubsetx]/dataLen)
                
        #     ### Calculate Expected Value 
        #     attr_informationGain[attrx] = total - sum(attrSubset_informationLoss)
            
        # ### Information Loss
        # print(labels[int(np.argmax(attr_informationGain, axis = 0))])

        
                
            
            
        

            
            
    

#%% Pick Attribute that Best Splits Data

def pickAttribute(trainData, avail_attributes):
    total_attributes = len(avail_attributes)
        
    ### Calculate Total Entropy/GiniIndex/MajorityError
    label_attr, count = np.unique(trainData[:,total_attributes], return_counts=1)
    total = calcInformationGain(count, sum(count))
    dataLen = np.shape(trainData)[0]
    
    ### Calculate Entropy/GiniIndex/MajorityError for Each Label
    attr_informationGain = np.zeros((total_attributes,1))
    for attrx in np.arange(0, total_attributes):   # loop through each attribute
        print(labels[attrx])
        attrSubset, attrCount = np.unique(trainData[:,attrx], return_counts=1)
        attrSubset_informationLoss = np.zeros((len(attrSubset), 1))
        
        for attrSubsetx in np.arange(0, len(attrSubset)):          # loop through each sub attribute
            # print(attrSubset[attrSubsetx])
            subsetIdx = [i for i in range(dataLen) if trainData[i, attrx] == attrSubset[attrSubsetx]]
            labelSubset, labelCount = np.unique(trainData[subsetIdx, total_attributes], return_counts=1)
            
            attrSubset_informationLoss[attrSubsetx] = calcInformationGain(labelCount, attrCount[attrSubsetx]) * (attrCount[attrSubsetx]/dataLen)
            
        ### Calculate Expected Value 
        attr_informationGain[attrx] = total - sum(attrSubset_informationLoss)
        
    ### Information Loss
    print(labels[int(np.argmax(attr_informationGain, axis = 0))])
    
    return int(np.argmax(attr_informationGain, axis = 0))


#%% Function Containing the Alorithms

def calcInformationGain(counts, total):
    xx = 0
    length = len(counts)
    
    if algorithmType == 'Entropy':
        for idx in np.arange(0, length): 
            if counts[idx] != 0 and total != 0:
                xx = xx - (counts[idx]/total)*np.log(counts[idx]/total)
        
    elif algorithmType == 'GiniIndex':
        for idx in np.arange(0, length): 
            if total != 0:
                xx = xx + (counts[idx]/total)**2
        xx = 1 - xx
        
    elif algorithmType == 'MajorityError':
        for idx in np.arange(0, length): 
            xx
            
    else:
        sys.exit('My error message')
        
    
    return xx


#%% MAIN
main()











































